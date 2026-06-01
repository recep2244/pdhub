"""Wet-Lab Readiness Advisor — expression system selection, purification
strategy, yield estimation, codon-risk flagging, and an integrated
go/no-go decision report.

All calculations are pure Python (no external tools required).
Thresholds and heuristics are grounded in published CMC/bioprocess
literature (see inline citations).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ExpressionSystemRec:
    system: str               # e.g. "E. coli periplasm"
    score: float              # 0–100 suitability
    estimated_yield_mgl: Tuple[float, float]   # (min, max) mg/L
    timeline_days: Tuple[int, int]             # (min, max) days to purified protein
    cost_per_mg_usd: Tuple[float, float]       # (min, max) $/mg
    rationale: List[str]
    caveats: List[str]
    protocol_notes: List[str]


@dataclass
class PurificationStep:
    step_number: int
    method: str          # e.g. "Protein A affinity"
    resin_or_column: str
    buffer_system: str
    expected_yield_pct: float   # % recovery
    expected_purity_pct: float
    notes: str


@dataclass
class PurificationStrategy:
    steps: List[PurificationStep]
    overall_recovery_pct: float
    final_purity_target: str
    qc_checklist: List[str]
    concentration_notes: str


@dataclass
class CodonRiskFlag:
    host: str
    rare_codons: List[str]       # codons to watch (AA-specific notes)
    risk_level: str              # "Low" | "Medium" | "High"
    recommendation: str


@dataclass
class WetLabReadinessReport:
    sequence: str
    mw_kda: float
    pi: float
    gravy: float
    instability_index: float
    n_cys: int
    n_met: int
    n_asn: int   # potential N-glycosylation Asn

    # Derived
    disulfide_pairs_predicted: int
    length: int
    has_signal_peptide_hint: bool   # N-terminal signal peptide heuristic
    secretory_hint: bool

    # Recommendations
    expression_systems: List[ExpressionSystemRec]  # ranked
    purification: PurificationStrategy
    codon_risks: List[CodonRiskFlag]

    # Overall verdict
    go_nogo: str              # "GO" | "CONDITIONAL" | "NO-GO"
    go_nogo_reason: str
    criteria_met: List[str]
    criteria_failed: List[str]
    criteria_watch: List[str]

    # Synopsis
    synopsis: str


# ---------------------------------------------------------------------------
# Constants / reference data
# ---------------------------------------------------------------------------

# Kyte-Doolittle hydropathy (for GRAVY)
_KD: Dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Amino-acid molecular weights (approximate, Da)
_AA_MW: Dict[str, float] = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.15,
    "E": 147.13, "Q": 146.15, "G": 75.07,  "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

# Instability index weights (DIWV, Guruprasad 1990)
_DIWV: Dict[str, Dict[str, float]] = {
    "A": {"A":1.0,"C":44.94,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "C": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":33.60,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "D": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":-7.49,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "E": {"A":1.0,"C":44.94,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "F": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":33.60},
    "G": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":13.34,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":13.34,"Y":1.0},
    "H": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":24.68,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "I": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":-7.49,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "K": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":33.60,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "L": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "M": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "N": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":1.0,"S":24.68,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "P": {"A":20.26,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":20.26,"W":1.0,"Y":1.0},
    "Q": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "R": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":58.28,"Y":1.0},
    "S": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "T": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "V": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":20.26,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "W": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":24.68,"N":13.34,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
    "Y": {"A":1.0,"C":1.0,"E":1.0,"D":1.0,"G":1.0,"F":1.0,"H":1.0,"I":1.0,"K":1.0,"L":1.0,"M":1.0,"N":1.0,"P":1.0,"Q":1.0,"R":1.0,"S":1.0,"T":1.0,"V":1.0,"W":1.0,"Y":1.0},
}

# Rare codon warnings per host
_RARE_CODONS: Dict[str, Dict[str, str]] = {
    "E. coli": {
        "Arg (AGG/AGA)": "AGG/AGA encode Arg; rare in E. coli BL21 — use Rosetta(DE3) pLysS or codon-optimise",
        "Ile (ATA)": "ATA encodes Ile; rare in E. coli — triggers ribosomal stalling at high density",
        "Leu (CTA)": "CTA encodes Leu; <1% frequency — causes truncation artefacts",
        "Pro (CCC)": "CCC can stall translation at runs ≥3; check Pro-rich sequences",
    },
    "HEK293/CHO": {
        "Arg (CGT/CGA)": "CGA is the rarest human Arg codon — reduce to <3% of Arg usage",
        "Leu (CTA)": "CTA frequency <5% in human — avoid in 5' coding region",
    },
    "Pichia pastoris": {
        "Arg (AGG)": "AGG very rare in Pichia; substitute with AGA or CGT",
        "Leu (CTA/CTG)": "CTA/CTG low-use; prefer TTA/TTG in Pichia context",
    },
    "N. benthamiana / Plant": {
        "Arg (CGG/CGA)": "Plant genomes prefer AGA/AGG — resynthesize CG-rich Arg stretches",
        "Leu (TTA)": "TTA rare in dicots; prefer CTG or TTG",
        "CpG dinucleotides": "High CpG density triggers plant post-transcriptional gene silencing (PTGS)",
    },
}

# pKa values for pI calculation
_PKA = {
    "N_TERM": 9.69, "C_TERM": 2.34,
    "D": 3.86, "E": 4.25, "C": 8.33, "Y": 10.07,
    "H": 6.00, "K": 10.53, "R": 12.48,
}


# ---------------------------------------------------------------------------
# Low-level sequence calculations
# ---------------------------------------------------------------------------

def _clean(seq: str) -> str:
    return "".join(c.upper() for c in seq if c.upper() in "ACDEFGHIKLMNPQRSTVWY")


def _mw(seq: str) -> float:
    return sum(_AA_MW.get(aa, 110.0) for aa in seq) - 18.015 * (len(seq) - 1)


def _gravy(seq: str) -> float:
    if not seq:
        return 0.0
    return sum(_KD.get(aa, 0.0) for aa in seq) / len(seq)


def _instability(seq: str) -> float:
    """Instability index (Guruprasad 1990). < 40 = stable."""
    if len(seq) < 2:
        return 0.0
    score = sum(
        _DIWV.get(seq[i], {}).get(seq[i + 1], 1.0)
        for i in range(len(seq) - 1)
    )
    return 10.0 / len(seq) * score


def _pi(seq: str) -> float:
    """Isoelectric point via binary search (Henderson-Hasselbalch)."""
    counts = {aa: seq.count(aa) for aa in "ACDEGHIKRTY"}

    def charge_at_ph(ph: float) -> float:
        c = 0.0
        c += 1.0 / (1.0 + 10 ** (ph - _PKA["N_TERM"]))
        c -= 1.0 / (1.0 + 10 ** (_PKA["C_TERM"] - ph))
        pos = {"K": _PKA["K"], "R": _PKA["R"], "H": _PKA["H"]}
        neg = {"D": _PKA["D"], "E": _PKA["E"], "C": _PKA["C"], "Y": _PKA["Y"]}
        for aa, pka in pos.items():
            c += counts.get(aa, 0) / (1.0 + 10 ** (ph - pka))
        for aa, pka in neg.items():
            c -= counts.get(aa, 0) / (1.0 + 10 ** (pka - ph))
        return c

    lo, hi = 0.0, 14.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if charge_at_ph(mid) > 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


def _has_signal_peptide_hint(seq: str) -> bool:
    """Heuristic: N-terminal ~25 aa with hydrophobic core typical of signal peptides."""
    if len(seq) < 10:
        return False
    window = seq[:25]
    hydrophobic = sum(1 for aa in window if _KD.get(aa, 0) > 1.5)
    return hydrophobic >= 7  # ≥7 hydrophobic residues in first 25 = signal peptide-like


def _count_potential_nglycosylation(seq: str) -> int:
    """Count NxS/T sequons (N-x-S/T where x != P)."""
    count = 0
    for i in range(len(seq) - 2):
        if seq[i] == "N" and seq[i + 1] != "P" and seq[i + 2] in "ST":
            count += 1
    return count


def _predict_disulfide_pairs(n_cys: int) -> int:
    """Rough estimate: even number → n/2 pairs; odd → (n-1)/2 + 1 free Cys."""
    if n_cys < 2:
        return 0
    return n_cys // 2


# ---------------------------------------------------------------------------
# Expression System Scorer
# ---------------------------------------------------------------------------

def _score_expression_systems(
    length: int,
    mw_kda: float,
    gravy: float,
    instability: float,
    n_cys: int,
    n_met: int,
    n_glyco: int,
    has_signal: bool,
    pi: float,
) -> List[ExpressionSystemRec]:
    recs: List[ExpressionSystemRec] = []

    n_ss = n_cys // 2   # predicted disulfide pairs

    # ── 1. E. coli periplasm ─────────────────────────────────────────────────
    ecoli_peri_score = 60.0
    ecoli_peri_rationale = []
    ecoli_peri_caveats = []
    ecoli_peri_protocol = [
        "Clone into pET-22b(+) or pHEN2 (scFv) with pelB/OmpA leader sequence",
        "Express BL21(DE3) at 30°C, 0.1 mM IPTG overnight (periplasmic export)",
        "Periplasmic extraction: osmotic shock (30 mM Tris pH 8, 20% sucrose, 1 mM EDTA)",
        "IMAC (His6) or Strep-Tactin purification, 1 step",
        "SEC-HPLC QC: expect >95% monomer for good scFv/nanobody",
    ]

    if mw_kda < 30:
        ecoli_peri_score += 15
        ecoli_peri_rationale.append(f"Small protein ({mw_kda:.0f} kDa) — periplasmic export efficient")
    elif mw_kda < 50:
        ecoli_peri_score += 5
        ecoli_peri_rationale.append(f"Medium protein ({mw_kda:.0f} kDa) — periplasmic export possible")
    else:
        ecoli_peri_score -= 20
        ecoli_peri_caveats.append(f"Large protein ({mw_kda:.0f} kDa) — poor periplasmic export efficiency; consider cytoplasmic or HEK293")

    if n_glyco > 0:
        ecoli_peri_score -= 25
        ecoli_peri_caveats.append(f"{n_glyco} N-X-S/T sequon(s) — E. coli cannot glycosylate; use HEK293/CHO if glycoform matters")

    if n_ss <= 2:
        ecoli_peri_score += 10
        ecoli_peri_rationale.append(f"≤2 disulfide pair(s) predicted — oxidising periplasm supports correct folding")
    elif n_ss <= 4:
        ecoli_peri_score += 0
        ecoli_peri_caveats.append(f"{n_ss} disulfide pairs — consider SHuffle T7 strain (cytoplasmic disulfide isomerase)")
    else:
        ecoli_peri_score -= 15
        ecoli_peri_caveats.append(f"{n_ss} disulfide pairs — high risk of misfolding; use baculovirus/HEK293")

    if gravy < 0:
        ecoli_peri_score += 5
        ecoli_peri_rationale.append("Hydrophilic sequence (GRAVY < 0) — good solubility in periplasm")
    elif gravy > 0.2:
        ecoli_peri_score -= 10
        ecoli_peri_caveats.append("Hydrophobic sequence (GRAVY > 0.2) — inclusion body risk; consider detergent solubilisation")

    if instability < 40:
        ecoli_peri_score += 5
        ecoli_peri_rationale.append("Stable sequence (II < 40)")
    else:
        ecoli_peri_score -= 5
        ecoli_peri_caveats.append("Unstable sequence (II ≥ 40) — add protease inhibitors, work on ice")

    ecoli_peri_score = max(0.0, min(100.0, ecoli_peri_score))
    yield_lo = 1.0 if ecoli_peri_score > 70 else 0.2
    yield_hi = 20.0 if ecoli_peri_score > 70 else (5.0 if ecoli_peri_score > 50 else 1.0)
    recs.append(ExpressionSystemRec(
        system="E. coli periplasm",
        score=ecoli_peri_score,
        estimated_yield_mgl=(yield_lo, yield_hi),
        timeline_days=(4, 7),
        cost_per_mg_usd=(5.0, 30.0),
        rationale=ecoli_peri_rationale,
        caveats=ecoli_peri_caveats,
        protocol_notes=ecoli_peri_protocol,
    ))

    # ── 2. E. coli cytoplasm (inclusion body + refolding) ────────────────────
    ecoli_cyto_score = 45.0
    ecoli_cyto_rationale = ["Inclusion body approach bypasses toxicity and solubility issues during expression"]
    ecoli_cyto_caveats = ["Refolding optimisation required (dilution, oxidative redox, additives)"]
    ecoli_cyto_protocol = [
        "Clone into pET-28a(+), express at 37°C 4 h high-IPTG (1 mM) for IBs",
        "IB wash: 1% Triton X-100, then H2O washes (×3)",
        "Denature in 8 M urea or 6 M Gdn-HCl + 50 mM DTT",
        "Refold by rapid dilution into 50 mM Tris pH 8, 150 mM NaCl, 0.4 M Arg, 0.5 mM GSSG/5 mM GSH",
        "IMAC from refold buffer; SEC polishing",
        "Validate by CD: α-helix 208/222 nm, β-sheet 218 nm",
    ]
    if n_glyco > 0:
        ecoli_cyto_caveats.append("No glycosylation — use only if aglycosyl form is acceptable")
    if gravy > 0.3 or instability > 50:
        ecoli_cyto_score += 15
        ecoli_cyto_rationale.append("High hydrophobicity/instability — IB approach may actually improve yield vs soluble expression")
    ecoli_cyto_score = max(0.0, min(100.0, ecoli_cyto_score))
    recs.append(ExpressionSystemRec(
        system="E. coli cytoplasm (IB refolding)",
        score=ecoli_cyto_score,
        estimated_yield_mgl=(5.0, 100.0),
        timeline_days=(7, 14),
        cost_per_mg_usd=(2.0, 15.0),
        rationale=ecoli_cyto_rationale,
        caveats=ecoli_cyto_caveats,
        protocol_notes=ecoli_cyto_protocol,
    ))

    # ── 3. HEK293 transient ──────────────────────────────────────────────────
    hek_score = 55.0
    hek_rationale = []
    hek_caveats = []
    hek_protocol = [
        "Clone into pTT5 or pcDNA3.1(+) with Kozak + signal peptide (e.g. IgG VH leader)",
        "PEI transfection of HEK293-6E or ExpiHEK293F (suspension)",
        "Harvest D7–D10; clarify 1000×g 10 min, 0.22 µm filter",
        "Protein A capture (IgG) or IMAC/Strep-Tactin; SEC polishing",
        "QC: SDS-PAGE (reducing + non-reducing), SEC-HPLC, DLS, SPR/BLI",
    ]
    if n_glyco > 0:
        hek_score += 20
        hek_rationale.append(f"{n_glyco} N-glycosylation site(s) — HEK293 provides human-like glycoforms")
    if n_ss >= 4:
        hek_score += 15
        hek_rationale.append(f"{n_ss} disulfide pairs — ER oxidative environment ensures correct pairing")
    if mw_kda > 50:
        hek_score += 10
        hek_rationale.append(f"Large protein ({mw_kda:.0f} kDa) — mammalian folding machinery handles complex folds")
    if length > 400:
        hek_score += 5
        hek_rationale.append("Long polypeptide — HEK293 chaperones support complex folding")
    if mw_kda < 15:
        hek_score -= 10
        hek_caveats.append("Very small proteins often lost during clarification/Protein A; add fusion tag (Fc, albumin)")
    hek_score = max(0.0, min(100.0, hek_score))
    yield_lo = 10.0 if hek_score > 70 else 2.0
    yield_hi = 100.0 if hek_score > 70 else 30.0
    recs.append(ExpressionSystemRec(
        system="HEK293 transient",
        score=hek_score,
        estimated_yield_mgl=(yield_lo, yield_hi),
        timeline_days=(7, 12),
        cost_per_mg_usd=(20.0, 150.0),
        rationale=hek_rationale,
        caveats=hek_caveats,
        protocol_notes=hek_protocol,
    ))

    # ── 4. CHO stable cell line ──────────────────────────────────────────────
    cho_score = 45.0  # lower default (higher cost/time)
    cho_rationale = ["Validated for cGMP therapeutic manufacturing; standard for IND-enabling studies"]
    cho_caveats = ["4–6 week selection + screening; not suitable for early-stage discovery"]
    cho_protocol = [
        "Clone into DHFR/GS-based vector or targeted integration vector (AAVS1/ROSA26)",
        "Stable selection: MTX pressure (DHFR) or MSX (GS)",
        "Single-cell cloning, fed-batch development, > 1 g/L achievable",
        "Protein A/G capture, viral inactivation (low pH), AEX/CEX polishing, UF/DF",
        "Full ICH Q6B characterisation: glycoform, charge variant, host cell protein, DNA",
    ]
    if n_glyco > 0:
        cho_score += 15
        cho_rationale.append("Glycoprotein — CHO provides consistent, regulatorily accepted glycoforms")
    if mw_kda > 100:
        cho_score += 10
        cho_rationale.append("Very large protein — CHO stable lines handle high MW better than transient")
    cho_score = max(0.0, min(100.0, cho_score))
    recs.append(ExpressionSystemRec(
        system="CHO stable cell line",
        score=cho_score,
        estimated_yield_mgl=(500.0, 5000.0),  # mg/L at scale
        timeline_days=(30, 60),
        cost_per_mg_usd=(1.0, 10.0),  # at scale
        rationale=cho_rationale,
        caveats=cho_caveats,
        protocol_notes=cho_protocol,
    ))

    # ── 5. Pichia pastoris ───────────────────────────────────────────────────
    pichia_score = 50.0
    pichia_rationale = ["Yeast secretion; cost-efficient scale-up; high-yield for small-medium proteins"]
    pichia_caveats = [
        "Hypermannosylation (high-mannose glycoforms) — immunogenic in humans for therapeutics",
        "Codon optimisation for Pichia essential (Kozak + methanol-inducible AOX1 promoter)",
    ]
    pichia_protocol = [
        "Clone into pPIC9K (secreted) with α-factor pre-pro signal; linearise for genomic integration",
        "Electroporate KM71H (aox1::ARG4) or X-33; select on MM/MD agar",
        "Buffered glycerol-complex medium (BMGY) growth, then methanol induction (BMMY)",
        "Harvest D3–D7; clarify; IMAC/SEC; validate glycoform by PNGase F digest + MS",
    ]
    if 15 <= mw_kda <= 60:
        pichia_score += 15
        pichia_rationale.append(f"Ideal size range for Pichia secretion ({mw_kda:.0f} kDa)")
    if n_glyco > 0 and n_glyco <= 2:
        pichia_score += 5
        pichia_rationale.append("Manageable glycosylation — Pichia can handle 1–2 N-glycan sites")
    elif n_glyco > 2:
        pichia_caveats.append("Many N-glycan sites — Pichia hypermannosylation may cause heterogeneity; use HEK293")
        pichia_score -= 15
    if n_ss > 4:
        pichia_caveats.append("Many disulfide bonds — ER PDI limited; validate by non-reducing SDS-PAGE")
    pichia_score = max(0.0, min(100.0, pichia_score))
    recs.append(ExpressionSystemRec(
        system="Pichia pastoris",
        score=pichia_score,
        estimated_yield_mgl=(10.0, 1000.0),
        timeline_days=(14, 28),
        cost_per_mg_usd=(3.0, 20.0),
        rationale=pichia_rationale,
        caveats=pichia_caveats,
        protocol_notes=pichia_protocol,
    ))

    # ── 6. Baculovirus / Sf9 ────────────────────────────────────────────────
    baculo_score = 40.0
    baculo_rationale = ["Handles multi-subunit complexes and large proteins with post-translational modifications"]
    baculo_caveats = [
        "Insect-type glycosylation (Man5 core only, no sialic acid) — not suitable for IgG Fc effector function studies",
        "BSL-2 baculovirus work required",
        "2–3 week bacmid generation before first expression",
    ]
    baculo_protocol = [
        "Clone into pFastBac; generate bacmid (DH10Bac); transfect Sf9 with Cellfectin II",
        "P1→P2→P3 virus amplification; MOI 2–5 PFU/cell for expression",
        "Harvest at 48–72 h (pre-cell-death); clarify 500×g 10 min",
        "IMAC (His tag at C-term preferred); SEC polishing",
        "Validate by Western (anti-His or specific Ab)",
    ]
    if n_ss > 4:
        baculo_score += 15
        baculo_rationale.append(f"Many disulfides ({n_ss} pairs) — insect ER better than E. coli for complex fold")
    if mw_kda > 80 or length > 600:
        baculo_score += 15
        baculo_rationale.append("Large/complex protein — baculovirus handles full-length membrane/multi-domain proteins")
    baculo_score = max(0.0, min(100.0, baculo_score))
    recs.append(ExpressionSystemRec(
        system="Baculovirus / Sf9",
        score=baculo_score,
        estimated_yield_mgl=(1.0, 20.0),
        timeline_days=(21, 35),
        cost_per_mg_usd=(30.0, 200.0),
        rationale=baculo_rationale,
        caveats=baculo_caveats,
        protocol_notes=baculo_protocol,
    ))

    # ── 7. Cell-free ─────────────────────────────────────────────────────────
    cellfree_score = 30.0
    cellfree_rationale = ["Rapid (<24 h); useful for toxic proteins, isotope labelling (NMR), or membrane proteins with detergent"]
    cellfree_caveats = [
        "Yield typically 0.1–2 mg/mL reaction; expensive at scale",
        "No in vivo PTMs (glycosylation, phosphorylation)",
        "Batch-to-batch variability; not suitable for GMP manufacturing",
    ]
    cellfree_protocol = [
        "Use PURExpress or wheat-germ (WEPRO) system; add DNA template directly",
        "Supplement with crowding agents (PEG), chaperones (DnaK/GroEL), redox buffer for disulfides",
        "Dialysis mode: 100 µL reaction + 1 mL feed buffer; harvest 16–24 h at 37°C",
        "IMAC purification; verify by silver stain (low µg amounts)",
    ]
    if n_met > 5:
        cellfree_score += 10
        cellfree_rationale.append(f"Many Met residues ({n_met}) — cell-free allows [13C/15N-Met] labelling for NMR")
    cellfree_score = max(0.0, min(100.0, cellfree_score))
    recs.append(ExpressionSystemRec(
        system="Cell-free (in vitro translation)",
        score=cellfree_score,
        estimated_yield_mgl=(0.1, 2.0),
        timeline_days=(1, 2),
        cost_per_mg_usd=(500.0, 5000.0),
        rationale=cellfree_rationale,
        caveats=cellfree_caveats,
        protocol_notes=cellfree_protocol,
    ))

    # Sort by score descending
    recs.sort(key=lambda r: -r.score)
    return recs


# ---------------------------------------------------------------------------
# Purification Strategy Builder
# ---------------------------------------------------------------------------

def _build_purification_strategy(
    mw_kda: float,
    pi: float,
    is_antibody: bool,
    n_ss: int,
    n_glyco: int,
    chain_type: Optional[str] = None,  # "VH", "VL_kappa", "VL_lambda", "VHH"
) -> PurificationStrategy:
    steps: List[PurificationStep] = []
    step = 1

    # ── Step 1: Affinity capture ─────────────────────────────────────────────
    if is_antibody or (chain_type and chain_type.startswith("V")):
        if chain_type in ("VH", None) or is_antibody:
            steps.append(PurificationStep(
                step_number=step,
                method="Protein A affinity chromatography",
                resin_or_column="MabSelect PrismA (Cytiva) or rProtein A Sepharose",
                buffer_system="Equilibration: PBS pH 7.4 | Elution: 0.1 M glycine pH 3.0 | Neutralise: 1 M Tris pH 8.0",
                expected_yield_pct=85.0,
                expected_purity_pct=95.0,
                notes="Neutralise immediately after elution to prevent acid-induced aggregation. Pool fractions above A280 ≥0.3.",
            ))
        elif chain_type in ("VL_kappa",):
            steps.append(PurificationStep(
                step_number=step,
                method="Protein L affinity (kappa light chain)",
                resin_or_column="CaptureSelect Kappa (Thermo Fisher)",
                buffer_system="Equilibration: 50 mM sodium phosphate, 150 mM NaCl pH 7.4 | Elution: 0.1 M glycine pH 2.7",
                expected_yield_pct=80.0,
                expected_purity_pct=90.0,
                notes="Protein L binds kappa VL domains regardless of heavy chain; useful for scFv/Fab purification.",
            ))
        elif chain_type in ("VHH",):
            steps.append(PurificationStep(
                step_number=step,
                method="IMAC (His6 tag) — nanobody",
                resin_or_column="HisTrap HP 1 mL (Cytiva) or TALON resin",
                buffer_system="Equilibration: 50 mM Tris pH 8, 300 mM NaCl, 20 mM imidazole | Elution: + 250 mM imidazole",
                expected_yield_pct=80.0,
                expected_purity_pct=85.0,
                notes="His6 C-terminal preferred for nanobodies (N-terminal may interfere with CDR3 loop). Remove tag with TEV/3C if functional assay requires.",
            ))
        step += 1
    else:
        # Generic His-tag IMAC as default
        steps.append(PurificationStep(
            step_number=step,
            method="IMAC — His6/His8 affinity",
            resin_or_column="HisTrap HP or Ni-NTA Superflow",
            buffer_system="Equilibration: 50 mM sodium phosphate pH 7.5, 300 mM NaCl, 10 mM imidazole | Elution: gradient to 500 mM imidazole",
            expected_yield_pct=75.0,
            expected_purity_pct=80.0,
            notes="Add 1 mM TCEP if > 2 Cys. Imidazole step gradient (50/150/300 mM) reduces co-eluting His-containing host proteins.",
        ))
        step += 1

    # ── Step 2: Tag removal (optional but recommended for functional assays) ──
    steps.append(PurificationStep(
        step_number=step,
        method="Affinity tag removal (optional)",
        resin_or_column="TEV protease (His6-TEV) + IMAC subtractive step, OR HRV 3C (His-3C) + Glutathione resin",
        buffer_system="TEV: 50 mM Tris pH 8, 0.5 mM EDTA, 1 mM DTT, 16°C overnight | 3C: 50 mM HEPES pH 7, 150 mM NaCl, 4°C 4 h",
        expected_yield_pct=85.0,
        expected_purity_pct=95.0,
        notes="Required before animal dosing (His-tag immunogenicity) and high-resolution structural studies. Skip for early-stage biophysical screening.",
    ))
    step += 1

    # ── Step 3: Ion exchange polishing ──────────────────────────────────────
    if pi < 6.5:
        # Anion exchange: protein negative at pH 7–8, binds to AEX
        steps.append(PurificationStep(
            step_number=step,
            method="Anion exchange chromatography (AEX)",
            resin_or_column="Q Sepharose HP or Capto Q (Cytiva)",
            buffer_system=f"Equilibration: 20 mM Tris pH 8.0 | Elution: 0–1 M NaCl linear gradient over 20 CV | (pI ≈ {pi:.1f}; protein binds at pH > pI)",
            expected_yield_pct=80.0,
            expected_purity_pct=98.0,
            notes="Load at ≤2 mS/cm conductivity. Aggregates and HCP typically do not bind; flow-through mode also possible.",
        ))
    elif pi > 7.5:
        # Cation exchange: protein positive at pH 5–6, binds to CEX
        steps.append(PurificationStep(
            step_number=step,
            method="Cation exchange chromatography (CEX)",
            resin_or_column="SP Sepharose HP or Capto SP ImpRes",
            buffer_system=f"Equilibration: 50 mM acetate pH 5.0 | Elution: 0–1 M NaCl gradient | (pI ≈ {pi:.1f}; binds below pI)",
            expected_yield_pct=80.0,
            expected_purity_pct=98.0,
            notes="Excellent for mAbs (pI 7–9). Also resolves charge variants (deamidated, C-terminal Lys) critical for lot release.",
        ))
    else:
        steps.append(PurificationStep(
            step_number=step,
            method="Hydrophobic interaction chromatography (HIC) — alternative polishing",
            resin_or_column="Butyl Sepharose HP or Phenyl Sepharose 6 FF",
            buffer_system=f"Equilibration: 50 mM phosphate pH 7, 1.5 M (NH4)2SO4 | Elution: reverse salt gradient to water",
            expected_yield_pct=75.0,
            expected_purity_pct=95.0,
            notes=f"pI ≈ {pi:.1f} (near-neutral) — HIC separates by hydrophobicity independent of charge. Useful for aggregated/misfold removal.",
        ))
    step += 1

    # ── Step 4: SEC polishing ────────────────────────────────────────────────
    steps.append(PurificationStep(
        step_number=step,
        method="Size-exclusion chromatography (SEC) — analytical + preparative",
        resin_or_column=f"Superdex {'75' if mw_kda < 50 else '200'} 10/300 GL (analytical) / HiLoad 16/600 (preparative)",
        buffer_system="PBS pH 7.4 or 10 mM HEPES pH 7.5, 150 mM NaCl — match intended storage/assay buffer",
        expected_yield_pct=90.0,
        expected_purity_pct=99.0,
        notes="PDI (polydispersity index by DLS) < 0.15 = monodisperse. Collect main peak only; discard leading shoulder (aggregates). Final QC: A280 conc., SDS-PAGE, SEC-HPLC.",
    ))

    # Overall recovery
    overall = 1.0
    for s in steps:
        overall *= (s.expected_yield_pct / 100.0)

    qc = [
        "SDS-PAGE (reducing + non-reducing) — check MW and disulfide bond formation",
        "SEC-HPLC — PDI < 0.15; ≥95% monomer",
        "A280 concentration (ε from sequence) + BCA for dilute samples",
        "DSF (Tm) — baseline thermal stability before proceeding to binding assays",
        "DLS — Rh < 10 nm (monomer); Z-average PDI < 0.2",
    ]
    if is_antibody:
        qc += [
            "SPR/BLI — confirm antigen binding (KD, kon, koff)",
            "Octet (BLI) — Fc gamma receptor binding (effector function confirmation)",
            "Endotoxin LAL assay — < 1 EU/mg for in vivo studies",
        ]

    conc_note = (
        "Concentrate with Amicon Ultra-4 (10 kDa cut-off) to ≥1 mg/mL. "
        "Add 0.02% Tween-20 if aggregation observed during concentration. "
        "Flash-freeze 20–50 µL aliquots in liquid nitrogen; store –80°C."
        if mw_kda > 15
        else "Small protein: use 3 kDa Amicon cut-off; verify no loss through membrane."
    )

    return PurificationStrategy(
        steps=steps,
        overall_recovery_pct=round(overall * 100, 1),
        final_purity_target=">99% by SEC-HPLC" if is_antibody else ">95% by SEC-HPLC",
        qc_checklist=qc,
        concentration_notes=conc_note,
    )


# ---------------------------------------------------------------------------
# Codon risk assessment
# ---------------------------------------------------------------------------

def _codon_risks(seq: str, hosts: Optional[List[str]] = None) -> List[CodonRiskFlag]:
    if hosts is None:
        hosts = ["E. coli", "HEK293/CHO"]

    flags: List[CodonRiskFlag] = []
    aa_counts: Dict[str, int] = {}
    for aa in seq:
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    for host in hosts:
        rare = _RARE_CODONS.get(host, {})
        if not rare:
            continue

        # Heuristic: flag if sequence is Arg-rich or Pro-rich (common problematic AAs)
        arg_pct = aa_counts.get("R", 0) / max(len(seq), 1) * 100
        pro_pct = aa_counts.get("P", 0) / max(len(seq), 1) * 100
        ile_pct = aa_counts.get("I", 0) / max(len(seq), 1) * 100

        risk_level = "Low"
        active_rare: List[str] = []

        if host == "E. coli":
            if arg_pct > 8:
                active_rare.append(f"Arg (AGG/AGA): {arg_pct:.1f}% Arg content — high rare-codon risk")
                risk_level = "High"
            elif arg_pct > 4:
                active_rare.append(f"Arg (AGG/AGA): {arg_pct:.1f}% Arg content — moderate rare-codon risk")
                risk_level = "Medium"
            if ile_pct > 7:
                active_rare.append(f"Ile (ATA): {ile_pct:.1f}% Ile content — check for ATA codons")
                if risk_level == "Low":
                    risk_level = "Medium"
            if pro_pct > 8:
                active_rare.append(f"Pro (CCC): {pro_pct:.1f}% Pro — Pro-rich stretches may stall ribosomes")
                if risk_level == "Low":
                    risk_level = "Medium"
            rec_text = (
                "Use Rosetta(DE3) pLysS or BL21-CodonPlus(DE3)-RIL to supplement rare tRNAs. "
                "Alternatively, codon-optimise with IDT/Twist/GenScript codon optimisation tools."
            )

        elif host in ("HEK293/CHO",):
            if arg_pct > 10:
                active_rare.append(f"Arg (CGA): {arg_pct:.1f}% Arg — check CGA frequency in synthetic gene")
                risk_level = "Medium"
            rec_text = "Optimise for human codon usage (IDT CodonOpt or Benchling). CGA replacement reduces truncation risk."

        elif host == "Pichia pastoris":
            if arg_pct > 6:
                active_rare.append(f"Arg (AGG): {arg_pct:.1f}% Arg — Pichia AGG very rare")
                risk_level = "Medium"
            rec_text = "Use Pichia codon usage table from NCBI (taxon 4922). Avoid CpG dinucleotides for best expression."

        else:  # plant
            # CpG estimate: rough CpG dinucleotide frequency from AA composition
            cg_aas = aa_counts.get("R", 0) + aa_counts.get("P", 0)  # Arg/Pro = CG-rich codons
            if cg_aas / max(len(seq), 1) > 0.12:
                active_rare.append("High CG-rich AA content (Arg+Pro > 12%) — may trigger plant PTGS")
                risk_level = "High"
            rec_text = (
                "Codon-optimise for Nicotiana benthamiana/Arabidopsis using PlantGenie or IDT. "
                "Remove CpG dinucleotides and rare Arg codons. Validate with transient expression assay."
            )

        if not active_rare:
            active_rare = ["No major rare-codon flags detected for " + host]
            risk_level = "Low"
            rec_text = "Standard codon optimisation recommended for any synthetic gene; no specific rare-codon concerns."

        flags.append(CodonRiskFlag(
            host=host,
            rare_codons=active_rare,
            risk_level=risk_level,
            recommendation=rec_text,
        ))

    return flags


# ---------------------------------------------------------------------------
# Go/No-Go decision
# ---------------------------------------------------------------------------

def _go_nogo(
    mw_kda: float,
    gravy: float,
    instability: float,
    pi: float,
    n_glyco: int,
    n_ss: int,
    top_system_score: float,
) -> Tuple[str, str, List[str], List[str], List[str]]:
    met: List[str] = []
    failed: List[str] = []
    watch: List[str] = []

    # Stability
    if instability < 40:
        met.append(f"Instability index {instability:.1f} < 40 (stable)")
    elif instability < 55:
        watch.append(f"Instability index {instability:.1f} (40–55 = borderline; optimise thermal stability)")
    else:
        failed.append(f"Instability index {instability:.1f} ≥ 55 (likely unstable in solution)")

    # Hydrophilicity
    if gravy < 0:
        met.append(f"GRAVY {gravy:.3f} < 0 (hydrophilic; expected soluble)")
    elif gravy < 0.3:
        watch.append(f"GRAVY {gravy:.3f} (0–0.3 = borderline; add solubilising tags or additives)")
    else:
        failed.append(f"GRAVY {gravy:.3f} ≥ 0.3 (hydrophobic; high aggregation risk)")

    # pI
    if 4.5 <= pi <= 9.5:
        met.append(f"pI {pi:.1f} in workable range (4.5–9.5)")
    else:
        watch.append(f"pI {pi:.1f} outside 4.5–9.5 — may aggregate near physiological pH; consider buffer additives")

    # Size
    if mw_kda <= 150:
        met.append(f"MW {mw_kda:.0f} kDa — standard expression system compatible")
    else:
        watch.append(f"MW {mw_kda:.0f} kDa — very large; limited expression system options; check multi-subunit assembly")

    # Disulfide complexity
    if n_ss == 0:
        met.append("No disulfide bonds — simple reducing buffer conditions throughout purification")
    elif n_ss <= 3:
        met.append(f"{n_ss} disulfide pair(s) — manageable; use oxidising periplasmic or mammalian expression")
    else:
        watch.append(f"{n_ss} disulfide pairs — complex; requires dedicated disulfide isomerase strain or mammalian expression")

    # Glycosylation
    if n_glyco == 0:
        met.append("No N-X-S/T sequons — aglycosyl form expressible in E. coli")
    elif n_glyco <= 2:
        watch.append(f"{n_glyco} potential N-glycosylation site(s) — mammalian expression required for correct glycoform")
    else:
        failed.append(f"{n_glyco} N-glycosylation sites — glycoprotein requires mammalian expression; E. coli/Pichia not suitable")

    # Expression system feasibility
    if top_system_score >= 70:
        met.append(f"Expression system suitability score {top_system_score:.0f}/100 — strong candidate")
    elif top_system_score >= 50:
        watch.append(f"Expression system suitability score {top_system_score:.0f}/100 — feasible with optimisation")
    else:
        failed.append(f"Expression system suitability score {top_system_score:.0f}/100 — challenging; extensive optimisation needed")

    # Overall verdict
    if len(failed) == 0 and len(watch) <= 2:
        verdict = "GO"
        reason = "All critical criteria met; proceed to gene synthesis and expression screening."
    elif len(failed) == 0:
        verdict = "CONDITIONAL"
        reason = f"{len(watch)} watch item(s) require attention before advancing to large-scale expression."
    elif len(failed) == 1:
        verdict = "CONDITIONAL"
        reason = f"1 failing criterion: {failed[0][:80]}... — address before committing to expression campaign."
    else:
        verdict = "NO-GO"
        reason = f"{len(failed)} criteria failed — sequence engineering required before wet lab advancement."

    return verdict, reason, met, failed, watch


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_wet_lab_report(
    sequence: str,
    is_antibody: bool = False,
    chain_type: Optional[str] = None,  # "VH", "VL_kappa", "VL_lambda", "VHH"
    codon_hosts: Optional[List[str]] = None,
) -> WetLabReadinessReport:
    """Generate a comprehensive wet-lab readiness report for a given sequence."""
    seq = _clean(sequence)
    if not seq:
        raise ValueError("Empty or non-standard sequence provided.")

    mw = _mw(seq)
    mw_kda = mw / 1000.0
    grav = _gravy(seq)
    inst = _instability(seq)
    pi_val = _pi(seq)
    n_cys = seq.count("C")
    n_met = seq.count("M")
    n_asn = seq.count("N")
    n_glyco = _count_potential_nglycosylation(seq)
    n_ss = _predict_disulfide_pairs(n_cys)
    has_sp = _has_signal_peptide_hint(seq)
    secretory = has_sp or is_antibody

    expr_systems = _score_expression_systems(
        length=len(seq),
        mw_kda=mw_kda,
        gravy=grav,
        instability=inst,
        n_cys=n_cys,
        n_met=n_met,
        n_glyco=n_glyco,
        has_signal=has_sp,
        pi=pi_val,
    )

    top_score = expr_systems[0].score if expr_systems else 0.0

    purification = _build_purification_strategy(
        mw_kda=mw_kda,
        pi=pi_val,
        is_antibody=is_antibody,
        n_ss=n_ss,
        n_glyco=n_glyco,
        chain_type=chain_type,
    )

    codon_risks = _codon_risks(seq, codon_hosts)

    verdict, reason, met, failed, watch = _go_nogo(
        mw_kda=mw_kda,
        gravy=grav,
        instability=inst,
        pi=pi_val,
        n_glyco=n_glyco,
        n_ss=n_ss,
        top_system_score=top_score,
    )

    # Synopsis
    top_sys_name = expr_systems[0].system if expr_systems else "unknown"
    top_yield = expr_systems[0].estimated_yield_mgl if expr_systems else (0, 0)
    synopsis = (
        f"{verdict}: {mw_kda:.0f} kDa protein (pI {pi_val:.1f}, GRAVY {grav:.2f}, II {inst:.0f}). "
        f"Top recommended expression system: **{top_sys_name}** "
        f"(suitability {top_score:.0f}/100, estimated yield {top_yield[0]:.0f}–{top_yield[1]:.0f} mg/L). "
        f"Purification: {len(purification.steps)}-step strategy, est. {purification.overall_recovery_pct:.0f}% recovery. "
        f"{len(met)} criteria met, {len(watch)} to watch, {len(failed)} failed."
    )

    return WetLabReadinessReport(
        sequence=seq,
        mw_kda=mw_kda,
        pi=pi_val,
        gravy=grav,
        instability_index=inst,
        n_cys=n_cys,
        n_met=n_met,
        n_asn=n_asn,
        disulfide_pairs_predicted=n_ss,
        length=len(seq),
        has_signal_peptide_hint=has_sp,
        secretory_hint=secretory,
        expression_systems=expr_systems,
        purification=purification,
        codon_risks=codon_risks,
        go_nogo=verdict,
        go_nogo_reason=reason,
        criteria_met=met,
        criteria_failed=failed,
        criteria_watch=watch,
        synopsis=synopsis,
    )
