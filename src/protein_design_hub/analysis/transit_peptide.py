"""Plant transit peptide and subcellular localisation prediction.

Pure-Python, no external dependencies.  Implements rule-based models for:

* Chloroplast transit peptides (cTP)  – TargetP / ChloroP heuristics
* Mitochondrial targeting peptides (mTP) – amphipathic helix, positive charge
* Classical signal peptides (SP)         – hydrophobic core + von Heijne rules
* Nuclear localisation signals (NLS)
* Nuclear export signals (NES)
* ER retention motifs (HDEL / KDEL family)
* Lipid anchor motifs (Myr, Palm, Prenyl)
* Vacuolar sorting signals (NPIR, C-terminal propeptide)
* Autophagy-interacting motifs (AIM / LIR)

References
----------
* Emanuelsson et al. 2000 (TargetP)
* von Heijne & Nishikawa 1991 (cTP composition rules)
* Hua & Sun 2001 (SignalP-like rules)
* Lange et al. 2007 (NLS/NES rules)
* Lo Conte et al. 1999 (ATG8 AIM)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Amino-acid property tables
# ---------------------------------------------------------------------------

_HYDROPHOBIC = set("AVILMFWP")
_CHARGED_POS = set("KR")
_CHARGED_NEG = set("DE")
_POLAR = set("STNQYHC")
_TINY = set("AGST")

# Kyte-Doolittle hydrophobicity (window-averaged)
_KD: Dict[str, float] = {
    "A":  1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C":  2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I":  4.5,
    "L":  3.8, "K": -3.9, "M":  1.9, "F":  2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V":  4.2,
}


def _kd_window(seq: str, window: int = 9) -> List[float]:
    """Return per-residue Kyte-Doolittle score using a sliding window."""
    n = len(seq)
    scores: List[float] = []
    for i in range(n):
        lo = max(0, i - window // 2)
        hi = min(n, i + window // 2 + 1)
        chunk = seq[lo:hi]
        scores.append(sum(_KD.get(aa, 0.0) for aa in chunk) / len(chunk))
    return scores


def _composition(seq: str) -> Dict[str, float]:
    """Fractional amino-acid composition."""
    n = len(seq)
    if n == 0:
        return {}
    counts: Dict[str, int] = {}
    for aa in seq.upper():
        counts[aa] = counts.get(aa, 0) + 1
    return {aa: cnt / n for aa, cnt in counts.items()}


def _charge(seq: str) -> int:
    """Net charge at pH 7 (simplified)."""
    return sum(1 for aa in seq if aa in _CHARGED_POS) - sum(1 for aa in seq if aa in _CHARGED_NEG)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TransitPeptidePrediction:
    """Organelle targeting prediction for the N-terminal region."""

    tp_type: str           # "cTP" | "mTP" | "SP" | "none"
    cleavage_site: int     # 1-indexed last position of TP (0 = no TP)
    confidence: float      # 0–1
    mature_start: int      # 1-indexed start of mature protein (= cleavage_site + 1)
    rationale: str

    @property
    def has_tp(self) -> bool:
        return self.tp_type != "none"

    def mature_sequence(self, full_sequence: str) -> str:
        if not self.has_tp:
            return full_sequence
        return full_sequence[self.cleavage_site:]


@dataclass
class LocalisationSignal:
    """A detected targeting / retention signal in a protein."""

    signal_type: str   # "NLS" | "NES" | "ER_retention" | "Myr" | "Palm" | "Prenyl" | "NPIR" | "AIM"
    start: int         # 1-indexed
    end: int
    sequence: str
    confidence: float
    description: str


@dataclass
class LocalisationReport:
    """Full subcellular localisation prediction for a plant protein."""

    primary_location: str            # Best-guess compartment
    primary_confidence: float
    alternative_locations: List[str]
    transit_peptide: TransitPeptidePrediction
    signals: List[LocalisationSignal] = field(default_factory=list)
    mutation_warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Transit peptide detection
# ---------------------------------------------------------------------------


def predict_transit_peptide(sequence: str) -> TransitPeptidePrediction:
    """Predict the N-terminal organelle targeting peptide.

    Implements a combined rule-based approach (TargetP + ChloroP heuristics):

    1. Chloroplast cTP: Ser/Thr/Ala-rich, hydroxylated, low acidic content,
       no or few charged residues in first 20 aa, positive Arg bias in middle.
    2. Mitochondrial mTP: Amphipathic helix with positive residues (Arg/Lys),
       typically 20–40 aa, ends with hydrophobic cleavage motif.
    3. Signal peptide: Three-part (n-region, h-region, c-region) per von Heijne.
    """
    seq = sequence.upper().replace("*", "")
    n = len(seq)
    window = min(80, n)
    nterminal = seq[:window]

    ctp_score = _score_ctp(nterminal)
    mtp_score = _score_mtp(nterminal)
    sp_score  = _score_sp(nterminal)

    best = max(ctp_score, mtp_score, sp_score)
    threshold = 0.45  # minimum confidence to call a TP

    if best < threshold:
        return TransitPeptidePrediction(
            tp_type="none", cleavage_site=0, confidence=1.0 - best,
            mature_start=1, rationale="No organelle targeting signal detected."
        )

    if ctp_score >= mtp_score and ctp_score >= sp_score:
        cleavage = _predict_ctp_cleavage(nterminal)
        return TransitPeptidePrediction(
            tp_type="cTP",
            cleavage_site=cleavage,
            confidence=ctp_score,
            mature_start=cleavage + 1,
            rationale=(
                f"Chloroplast transit peptide predicted (score {ctp_score:.2f}). "
                f"Ser/Thr/Ala-rich N-terminus, low acidic residues, positive charge bias. "
                f"Predicted cleavage after position {cleavage} "
                f"(C-motif: {seq[max(0,cleavage-3):cleavage+2]})."
            ),
        )
    elif mtp_score >= sp_score:
        cleavage = _predict_mtp_cleavage(nterminal)
        return TransitPeptidePrediction(
            tp_type="mTP",
            cleavage_site=cleavage,
            confidence=mtp_score,
            mature_start=cleavage + 1,
            rationale=(
                f"Mitochondrial targeting peptide predicted (score {mtp_score:.2f}). "
                f"Amphipathic helix with positive charge, ends in hydrophobic cleavage motif. "
                f"Predicted cleavage after position {cleavage}."
            ),
        )
    else:
        cleavage = _predict_sp_cleavage(nterminal)
        return TransitPeptidePrediction(
            tp_type="SP",
            cleavage_site=cleavage,
            confidence=sp_score,
            mature_start=cleavage + 1,
            rationale=(
                f"Classical signal peptide predicted (score {sp_score:.2f}). "
                f"Three-part structure (n-h-c region). "
                f"Predicted cleavage after position {cleavage} "
                f"(AXA motif: {seq[max(0,cleavage-3):cleavage+2]})."
            ),
        )


def _score_ctp(seq: str) -> float:
    """Score chloroplast transit peptide characteristics (0–1)."""
    if len(seq) < 15:
        return 0.0
    comp = _composition(seq[:60])
    score = 0.0
    # Ser/Thr/Ala richness: cTPs typically >25% STA
    sta = comp.get("S", 0) + comp.get("T", 0) + comp.get("A", 0)
    score += min(1.0, sta / 0.35) * 0.25
    # Low acidic residue content in first 20 aa
    acid_n20 = sum(1 for aa in seq[:20] if aa in "DE") / 20.0
    score += (1.0 - min(1.0, acid_n20 / 0.10)) * 0.20
    # Positive charge bias (Arg preferred over Lys in cTPs)
    arg_count = seq[:60].count("R")
    score += min(1.0, arg_count / 4.0) * 0.20
    # Low or no hydrophobic stretch in first 10 aa (unlike mTP/SP)
    kd = _kd_window(seq[:20], 5)
    max_kd = max(kd) if kd else 0.0
    score += (1.0 - min(1.0, max_kd / 3.0)) * 0.15
    # Length: cTPs are 40–80 aa; penalise very short
    n = len(seq)
    if n >= 40:
        score += 0.10
    elif n >= 25:
        score += 0.05
    # No Cys in first 30 aa (typical)
    cys_n30 = seq[:30].count("C")
    score += 0.10 * (1.0 if cys_n30 == 0 else 0.5)
    return min(1.0, score)


def _score_mtp(seq: str) -> float:
    """Score mitochondrial targeting peptide characteristics (0–1)."""
    if len(seq) < 15:
        return 0.0
    score = 0.0
    # Net positive charge in first 30 aa
    charge = _charge(seq[:30])
    score += min(1.0, charge / 4.0) * 0.30
    # Amphipathic helix: alternating hydrophobic and charged residues
    # Simple heuristic: >3 Arg in first 40 aa
    arg = seq[:40].count("R")
    lys = seq[:40].count("K")
    score += min(1.0, (arg + lys) / 5.0) * 0.20
    # Hydrophobic residues present but not clustered (unlike SP)
    kd = _kd_window(seq[:30], 5)
    mean_kd = sum(kd) / len(kd) if kd else 0.0
    max_kd = max(kd) if kd else 0.0
    # mTP: moderate hydrophobicity, not extreme
    if 0.5 <= mean_kd <= 2.5:
        score += 0.20
    elif mean_kd > 0:
        score += 0.10
    # Ends with hydrophobic motif (MPP cleavage: R-X↓ or R-X-S|A|T)
    # Check for Arg-containing cleavage motif in positions 20–50
    motif = re.search(r"R[^DE]{0,2}[ILVMFYA]", seq[15:50])
    if motif:
        score += 0.20
    # Low Trp (rare in mTP)
    w_count = seq[:40].count("W")
    score += 0.10 * (1.0 if w_count == 0 else 0.5)
    return min(1.0, score)


def _score_sp(seq: str) -> float:
    """Score classical signal peptide characteristics (0–1)."""
    if len(seq) < 10:
        return 0.0
    score = 0.0
    # n-region: 1–5 aa, 1–2 positive charges
    n_region = seq[:5]
    n_charge = sum(1 for aa in n_region if aa in "KR")
    if n_charge >= 1:
        score += 0.20
    # h-region: 7–15 aa hydrophobic core
    for start in range(2, 10):
        h = seq[start:start+10]
        hydro = sum(1 for aa in h if aa in "LVIAFM")
        if hydro >= 7:
            score += 0.35
            break
    # c-region + cleavage: AXA motif
    for i in range(15, min(35, len(seq) - 3)):
        window = seq[i:i+3]
        if window[0] in _TINY and window[2] in _TINY:
            score += 0.30
            break
    # Penalise sequences starting with very Ser/Thr rich (cTP-like)
    sta = sum(1 for aa in seq[:20] if aa in "STA") / 20.0
    if sta > 0.30:
        score -= 0.15
    return max(0.0, min(1.0, score))


def _predict_ctp_cleavage(seq: str) -> int:
    """Predict cTP cleavage site (1-indexed position of last TP residue).

    The transit peptide ends at a region enriched in Ala, Val, Ala
    (the C-motif recognised by the stromal processing peptidase, SPP).
    Typically 40–80 aa from N-terminus.
    """
    # Look for a region where STA-rich segment transitions to less polar
    best_pos = min(50, len(seq) - 1)
    best_score = -999.0
    for i in range(20, min(80, len(seq) - 5)):
        # SPP cleavage sites: VAXA, IAXA, AAXA (small-x-small motif)
        window = seq[i - 1:i + 3]
        if len(window) < 3:
            continue
        motif_score = 0.0
        if window[0] in "VA":
            motif_score += 1.0
        if window[2] in "AVST":
            motif_score += 0.5
        # Ser/Thr richness in preceding 10 aa (typical cTP tail)
        pre = seq[max(0, i - 10):i]
        sta = sum(1 for aa in pre if aa in "STA") / max(1, len(pre))
        motif_score += sta * 2.0
        # Hydrophobic transition after cleavage
        post = seq[i + 1:i + 6] if i + 6 <= len(seq) else ""
        if post:
            kd_post = sum(_KD.get(aa, 0.0) for aa in post) / len(post)
            if kd_post > 1.0:
                motif_score += 1.0
        if motif_score > best_score:
            best_score = motif_score
            best_pos = i
    return best_pos


def _predict_mtp_cleavage(seq: str) -> int:
    """Predict mTP cleavage site (MPP cleaves after R-x, R-2 rule)."""
    # Scan for R-X↓S/A/T or simply R in positions 15–45
    for i in range(15, min(50, len(seq) - 2)):
        if seq[i] == "R" and seq[i + 1] not in "DE":
            # Additional: look for small residue at +2
            if i + 2 < len(seq) and seq[i + 2] in "SAGIT":
                return i + 1  # cleavage after R
    # Fallback: fixed 30 aa
    return min(30, len(seq))


def _predict_sp_cleavage(seq: str) -> int:
    """Predict signal peptide cleavage site via AXA motif."""
    # Look for A-x-A around positions 15–35
    for i in range(14, min(35, len(seq) - 3)):
        window = seq[i:i + 3]
        if window[0] in "AG" and window[2] in "AGST":
            return i  # cleavage between position i and i+1 (after i, 0-indexed)
    return min(20, len(seq))


# ---------------------------------------------------------------------------
# Subcellular localisation signals
# ---------------------------------------------------------------------------

# Classical NLS patterns
_NLS_PATTERNS = [
    (r"[KR]{4,}",                        "Monopartite NLS (KKKRK-like)"),
    (r"[KR]{2}[^KR]{10,12}[KR]{3,}",    "Bipartite NLS (classical)"),
    (r"K[KR][^KR]{0,2}K",               "Short NLS"),
]

# Nuclear export signal (leucine-rich, CRM1-binding)
_NES_PATTERN = r"L[^LI]{1,3}L[^LI]{1,3}L[^LI]{1,3}[LI]"

# ER retention
_ER_RETENTION = {
    "HDEL": "ER lumenal retention (HDEL)",
    "KDEL": "ER lumenal retention (KDEL)",
    "DDEL": "ER lumenal retention (DDEL)",
    "ADEL": "ER lumenal retention (ADEL)",
    "RDEL": "ER lumenal retention (RDEL)",
}

# Vacuolar sorting (NPIR, bp-VSP)
_VACUOLAR_PATTERNS = [
    (r"NPIR",          "Vacuolar sorting signal (NPIR, bp-VSP)"),
    (r"[ILV][ILV]$",   "C-terminal propeptide for lytic vacuole"),
]

# Lipid anchors
_MYR_PATTERN  = r"^MG[IVLFA]"       # N-terminal myristoylation
_PALM_PATTERN = r"CC|CXC|CCXX|CAAX"  # Palmitoylation / CaaX prenylation

# ATG8-interacting motif (AIM / LIR) – [W/F/Y]-x-x-[L/I/V]
_AIM_PATTERN = r"[WFY]..[LIV]"


def predict_localisation_signals(sequence: str) -> List[LocalisationSignal]:
    """Detect all targeting / retention signals in the sequence."""
    seq = sequence.upper()
    signals: List[LocalisationSignal] = []

    # NLS
    for pattern, desc in _NLS_PATTERNS:
        for m in re.finditer(pattern, seq):
            signals.append(LocalisationSignal(
                signal_type="NLS",
                start=m.start() + 1, end=m.end(),
                sequence=m.group(),
                confidence=0.70,
                description=desc,
            ))
    # NES
    for m in re.finditer(_NES_PATTERN, seq):
        signals.append(LocalisationSignal(
            signal_type="NES",
            start=m.start() + 1, end=m.end(),
            sequence=m.group(),
            confidence=0.65,
            description="Nuclear export signal (leucine-rich, CRM1-dependent)",
        ))
    # ER retention (C-terminal)
    for motif, desc in _ER_RETENTION.items():
        if seq.endswith(motif):
            pos = len(seq) - len(motif)
            signals.append(LocalisationSignal(
                signal_type="ER_retention",
                start=pos + 1, end=len(seq),
                sequence=motif,
                confidence=0.90,
                description=desc,
            ))
    # Vacuolar
    for pattern, desc in _VACUOLAR_PATTERNS:
        for m in re.finditer(pattern, seq):
            signals.append(LocalisationSignal(
                signal_type="vacuolar",
                start=m.start() + 1, end=m.end(),
                sequence=m.group(),
                confidence=0.60,
                description=desc,
            ))
    # Myristoylation
    if re.match(_MYR_PATTERN, seq):
        signals.append(LocalisationSignal(
            signal_type="Myr",
            start=1, end=6,
            sequence=seq[:6],
            confidence=0.75,
            description="N-terminal myristoylation signal (plasma membrane anchor)",
        ))
    # CaaX prenylation
    caax = re.search(r"C[AVLIMFWP][AVLIMFWP][XQHST]$", seq[-4:] if len(seq) >= 4 else seq)
    if caax:
        signals.append(LocalisationSignal(
            signal_type="Prenyl",
            start=len(seq) - 3, end=len(seq),
            sequence=seq[-4:],
            confidence=0.70,
            description="CaaX prenylation motif (farnesyl / geranylgeranyl)",
        ))
    # AIM
    for m in re.finditer(_AIM_PATTERN, seq):
        signals.append(LocalisationSignal(
            signal_type="AIM",
            start=m.start() + 1, end=m.end(),
            sequence=m.group(),
            confidence=0.55,
            description="ATG8-interacting motif (autophagy receptor)",
        ))

    # Deduplicate overlapping NLS signals
    seen: set = set()
    unique: List[LocalisationSignal] = []
    for s in signals:
        key = (s.signal_type, s.start, s.end)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def comprehensive_localisation_report(
    full_sequence: str,
    include_mature: bool = True,
) -> LocalisationReport:
    """Return a full localisation report for a plant protein sequence."""
    seq = full_sequence.upper().replace("*", "")
    tp = predict_transit_peptide(seq)
    mature = tp.mature_sequence(seq) if include_mature and tp.has_tp else seq
    signals = predict_localisation_signals(mature)

    # Determine primary location
    location, confidence, alternatives = _determine_primary_location(tp, signals)

    return LocalisationReport(
        primary_location=location,
        primary_confidence=confidence,
        alternative_locations=alternatives,
        transit_peptide=tp,
        signals=signals,
    )


def _determine_primary_location(
    tp: TransitPeptidePrediction,
    signals: List[LocalisationSignal],
) -> Tuple[str, float, List[str]]:
    """Derive the most likely compartment from TP + signals."""
    alternatives: List[str] = []
    # TP overrides signal-based prediction
    if tp.tp_type == "cTP":
        alternatives = ["Chloroplast envelope", "Thylakoid lumen"]
        return "Chloroplast stroma", tp.confidence, alternatives
    if tp.tp_type == "mTP":
        return "Mitochondrial matrix", tp.confidence, ["Mitochondrial inner membrane"]
    if tp.tp_type == "SP":
        # SP → ER → check for retention or secretory default
        er_ret = [s for s in signals if s.signal_type == "ER_retention"]
        vac = [s for s in signals if s.signal_type == "vacuolar"]
        if er_ret:
            return "Endoplasmic reticulum (lumen)", 0.85, ["Golgi apparatus"]
        if vac:
            return "Vacuole", 0.75, ["Extracellular (secreted)"]
        return "Extracellular / secreted", tp.confidence, ["Golgi apparatus", "Plasma membrane"]

    # No TP → cytoplasmic default, modified by signals
    nls = [s for s in signals if s.signal_type == "NLS"]
    nes = [s for s in signals if s.signal_type == "NES"]
    myr = [s for s in signals if s.signal_type in ("Myr", "Prenyl")]
    aim = [s for s in signals if s.signal_type == "AIM"]

    if nls and not nes:
        return "Nucleus", 0.72, ["Cytoplasm"]
    if nes and not nls:
        return "Cytoplasm (NES-exported)", 0.65, ["Nucleus (transient)"]
    if myr:
        return "Plasma membrane (lipid-anchored)", 0.70, ["Cytoplasm"]
    if aim:
        alternatives = ["Cytoplasm"]
        return "Autophagosome / selective autophagy", 0.55, alternatives
    return "Cytoplasm", 0.60, ["Nucleus (NLR shuttling possible)"]


# ---------------------------------------------------------------------------
# Mutation impact on targeting signals
# ---------------------------------------------------------------------------


def assess_mutation_localisation_impact(
    position: int,          # 1-indexed
    wt_aa: str,
    mut_aa: str,
    report: LocalisationReport,
) -> List[str]:
    """Return warning strings if a point mutation disrupts a targeting signal."""
    warnings: List[str] = []
    tp = report.transit_peptide
    # Mutation inside TP
    if tp.has_tp and 1 <= position <= tp.cleavage_site:
        warnings.append(
            f"Position {position} is inside the {tp.tp_type} transit peptide "
            f"({wt_aa}→{mut_aa}). Mutation may disrupt organelle targeting."
        )
    # Mutation at NLS
    for sig in report.signals:
        if sig.start <= position <= sig.end:
            if sig.signal_type == "NLS" and wt_aa in "KR" and mut_aa not in "KR":
                warnings.append(
                    f"Position {position} ({wt_aa}→{mut_aa}) is in a {sig.description}. "
                    "Removing basic residue may ablate nuclear import."
                )
            elif sig.signal_type == "ER_retention" and wt_aa in "HKDE" and mut_aa not in "HKDE":
                warnings.append(
                    f"Position {position} ({wt_aa}→{mut_aa}) is in the {sig.description}. "
                    "Mutation may disrupt ER retention → protein secreted."
                )
            elif sig.signal_type == "NES" and wt_aa == "L" and mut_aa not in "LI":
                warnings.append(
                    f"Position {position} ({wt_aa}→{mut_aa}) disrupts a leucine in a "
                    "nuclear export signal. Protein may accumulate in nucleus."
                )
    return warnings
