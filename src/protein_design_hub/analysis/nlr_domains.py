"""NLR (Nucleotide-binding Leucine-rich Repeat) domain annotation for plants.

Pure-Python, no external dependencies.

Annotates TIR, CC, NBS-ARC, and LRR domains using motif patterns and
positional rules derived from:

* Meyers et al. 2003 (Plant NLR classification)
* Takken & Goverse 2012 (NBS-ARC activation cycle)
* Maekawa et al. 2011 (TIR domain structure)
* Padmanabhan et al. 2009 (LRR specificity)
* Baggs et al. 2017 (NLR network)

Supports:
  - TIR-NLR (TNL) and CC-NLR (CNL) discrimination
  - NBS motif detection: P-loop, RNBS-A, RNBS-B, RNBS-D, MHD
  - LRR repeat counting and specificity position mapping
  - Critical mutation flagging (P-loop, ADP-ribosylase, MHD disruption)
  - NLR-aware pLDDT / quality interpretation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NLRMotif:
    """A functional motif within an NLR domain."""

    name: str              # e.g. "P-loop", "MHD", "W-box"
    start: int             # 1-indexed
    end: int
    sequence: str
    is_critical: bool      # mutations here likely LOF
    note: str


@dataclass
class NLRDomainAnnotation:
    """A single annotated structural domain."""

    domain_type: str       # "TIR" | "CC" | "NBS" | "LRR" | "unknown"
    subtype: str           # e.g. "TIR-NLR" or "CC-NLR", empty for sub-domains
    start: int
    end: int
    confidence: float
    motifs: List[NLRMotif] = field(default_factory=list)
    lrr_count: int = 0     # filled if domain_type == "LRR"
    specificity_positions: List[int] = field(default_factory=list)


@dataclass
class NLRAnnotation:
    """Full NLR architecture annotation for a protein sequence."""

    is_nlr: bool
    nlr_class: str          # "TNL" | "CNL" | "NL" | "unknown"
    confidence: float
    domains: List[NLRDomainAnnotation] = field(default_factory=list)
    all_motifs: List[NLRMotif] = field(default_factory=list)
    critical_positions: List[int] = field(default_factory=list)
    summary: str = ""


@dataclass
class MutationImpactAnnotation:
    """Impact prediction for a single point mutation in an NLR."""

    position: int
    wt_aa: str
    mut_aa: str
    domain: Optional[str]
    motif: Optional[str]
    is_critical_site: bool
    predicted_effect: str    # "LOF" | "altered_specificity" | "neutral" | "uncertain"
    risk_level: str          # "high" | "medium" | "low"
    rationale: str


# ---------------------------------------------------------------------------
# Motif patterns (regex, 1-letter AA code)
# ---------------------------------------------------------------------------

# TIR domain signatures
_TIR_MOTIFS = {
    # W-box: conserved Trp near centre of TIR fold (Trp-x(5)-Glu)
    "W-box":          (r"W[^*]{4,6}E",   True,  "TIR W-box; required for TIR-TIR interaction and cell death signalling"),
    # ADP-ribosylase-like catalytic triad
    "ADP-rib-1":      (r"[ED][ED]G[IVLA]",False, "TIR ADP-ribosylase motif 1 (NAD+-cleaving)"),
    "ADP-rib-2":      (r"G[ST]G[XAVILS]", False, "TIR ADP-ribosylase motif 2"),
    # TIR linker to NBS: usually disordered; flagged automatically by pLDDT
}

# NBS-ARC domain motifs (TIR or CC can precede NBS)
_NBS_MOTIFS = {
    "P-loop":   (r"G[AVLIST]G[^DE]{2}GK[TS]", True,
                 "P-loop (Walker-A); GxxxxGKT required for ATP/ADP binding; "
                 "any mutation (esp. K/T) = non-functional NLR"),
    "RNBS-A":   (r"[FYILM][DENT][LVIM][SAT]XX[QEDK]",  True,
                 "RNBS-A: required for NBS-ARC activation state"),
    "RNBS-B":   (r"[KRHST][LIVMF]{2}[TS]T[RKHQ]",     False,
                 "RNBS-B: structural role in NBS pocket"),
    "RNBS-D":   (r"[RKHQ]FLY[ILVF]{2}",                True,
                 "RNBS-D: required for effector-triggered conformational change"),
    "MHD":      (r"M[HKRQ][DE]",                       True,
                 "MHD motif: Triad at end of ARC2; gain-of-function in autoactive alleles; "
                 "D→A = constitutive activation"),
    "GLPL":     (r"GLPL",                               False,
                 "GLPL motif: conserved surface-exposed strand in ARC subdomain"),
}

# LRR repeat unit (canonical ~20–24 aa, with conserved xxLxLxx core)
_LRR_UNIT = re.compile(r"[ACILVMF]{1,3}[^P]{2}L[^P]{2}L[^P]{5,10}")
# LRR specificity positions: 11, 12, 14 in each repeat (0-indexed within repeat)
_LRR_SPEC_OFFSETS = [10, 11, 13]  # 0-indexed within repeat → positions 11, 12, 14

# Coiled-coil CC pattern (heptad repeat: [LVIM]xx[LVIM]xxx)
_CC_PATTERN = re.compile(r"([LVIM][^LVIM]{2}[LVIM][^LVIM]{3}){2,}")


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _find_motif(
    seq: str,
    pattern: str,
    name: str,
    is_critical: bool,
    note: str,
    search_start: int = 0,
    search_end: Optional[int] = None,
) -> Optional[NLRMotif]:
    """Search seq[search_start:search_end] for pattern, return first hit."""
    region = seq[search_start:search_end]
    m = re.search(pattern, region, re.IGNORECASE)
    if m is None:
        return None
    abs_start = search_start + m.start() + 1  # 1-indexed
    abs_end   = search_start + m.end()
    return NLRMotif(
        name=name, start=abs_start, end=abs_end,
        sequence=m.group(), is_critical=is_critical, note=note,
    )


def _score_tir_region(seq: str) -> Tuple[float, List[NLRMotif]]:
    """Return (confidence, motifs) for a TIR domain in seq."""
    motifs: List[NLRMotif] = []
    score = 0.0
    for name, (pat, crit, note) in _TIR_MOTIFS.items():
        hit = _find_motif(seq, pat, name, crit, note)
        if hit:
            motifs.append(hit)
            score += 0.30 if crit else 0.15
    # Composition: TIR domains are ~180 aa, Trp-rich
    w_frac = seq.count("W") / max(1, len(seq))
    if w_frac > 0.015:
        score += 0.15
    # TIR fold has distinctive small size and alpha-beta composition
    if 100 <= len(seq) <= 200:
        score += 0.10
    return min(1.0, score), motifs


def _score_nbs_region(seq: str) -> Tuple[float, List[NLRMotif]]:
    """Return (confidence, motifs) for an NBS-ARC domain in seq."""
    motifs: List[NLRMotif] = []
    score = 0.0
    for name, (pat, crit, note) in _NBS_MOTIFS.items():
        hit = _find_motif(seq, pat, name, crit, note)
        if hit:
            motifs.append(hit)
            score += 0.20 if crit else 0.10
    # NBS-ARC is ~300 aa
    if 200 <= len(seq) <= 400:
        score += 0.10
    return min(1.0, score), motifs


def _count_lrr_repeats(seq: str) -> Tuple[int, List[int], List[NLRMotif]]:
    """Count LRR repeats and identify specificity positions.

    Returns (n_repeats, specificity_positions_1indexed, lrr_motifs).
    """
    motifs: List[NLRMotif] = []
    spec_positions: List[int] = []
    matches = list(_LRR_UNIT.finditer(seq))
    for i, m in enumerate(matches):
        motifs.append(NLRMotif(
            name=f"LRR-{i+1}",
            start=m.start() + 1, end=m.end(),
            sequence=m.group(),
            is_critical=False,
            note=f"LRR repeat {i+1} (~{m.end()-m.start()} aa)",
        ))
        # Specificity positions 11, 12, 14 of each repeat
        for off in _LRR_SPEC_OFFSETS:
            pos = m.start() + off + 1  # 1-indexed
            if pos <= len(seq):
                spec_positions.append(pos)
    return len(matches), spec_positions, motifs


def _detect_cc(seq: str, region_end: int = 200) -> Optional[NLRDomainAnnotation]:
    """Detect a coiled-coil domain in the first `region_end` residues."""
    region = seq[:region_end]
    m = _CC_PATTERN.search(region)
    if m is None:
        return None
    return NLRDomainAnnotation(
        domain_type="CC",
        subtype="N-terminal CC",
        start=m.start() + 1,
        end=m.end(),
        confidence=0.60,
    )


# ---------------------------------------------------------------------------
# Positional scanning for full NLR sequence
# ---------------------------------------------------------------------------


def _partition_nlr(seq: str) -> Dict[str, Tuple[int, int]]:
    """Partition sequence into rough domain windows.

    Typical TNL/CNL layout:
      - N-terminal module (TIR or CC): 0 – ~25–30% of length
      - NBS-ARC:                       ~25 – ~60%
      - LRR solenoid:                  ~60 – 100%
    """
    n = len(seq)
    return {
        "N-module": (0, int(n * 0.30)),
        "NBS":      (int(n * 0.25), int(n * 0.62)),
        "LRR":      (int(n * 0.55), n),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_nlr(sequence: str) -> NLRAnnotation:
    """Annotate NLR domain architecture of an input protein sequence.

    Parameters
    ----------
    sequence:
        Amino-acid sequence (1-letter, may include *).

    Returns
    -------
    NLRAnnotation with detected domains, motifs, and critical positions.
    """
    seq = sequence.upper().replace("*", "").replace("-", "")
    n = len(seq)
    if n < 150:
        return NLRAnnotation(is_nlr=False, nlr_class="unknown", confidence=0.0,
                             summary="Sequence too short for NLR analysis (< 150 aa).")

    windows = _partition_nlr(seq)
    domains: List[NLRDomainAnnotation] = []
    all_motifs: List[NLRMotif] = []
    nlr_class = "unknown"
    total_confidence = 0.0

    # --- N-terminal module: TIR or CC ---
    nm_start, nm_end = windows["N-module"]
    n_seq = seq[nm_start:nm_end]
    tir_score, tir_motifs = _score_tir_region(n_seq)

    # Offset motif positions to absolute
    for m in tir_motifs:
        m.start += nm_start
        m.end   += nm_start

    cc_dom = _detect_cc(seq, region_end=nm_end + 50)

    if tir_score >= 0.35:
        nlr_class = "TNL"
        tir_dom = NLRDomainAnnotation(
            domain_type="TIR",
            subtype="TIR-NLR",
            start=nm_start + 1, end=nm_end,
            confidence=tir_score,
            motifs=tir_motifs,
        )
        domains.append(tir_dom)
        all_motifs.extend(tir_motifs)
        total_confidence += tir_score * 0.30
    elif cc_dom is not None:
        nlr_class = "CNL"
        domains.append(cc_dom)
        total_confidence += cc_dom.confidence * 0.20

    # --- NBS-ARC domain ---
    nbs_start, nbs_end = windows["NBS"]
    nbs_seq = seq[nbs_start:nbs_end]
    nbs_score, nbs_motifs = _score_nbs_region(nbs_seq)
    for m in nbs_motifs:
        m.start += nbs_start
        m.end   += nbs_start
    if nbs_score >= 0.20:
        nbs_dom = NLRDomainAnnotation(
            domain_type="NBS",
            subtype="NBS-ARC",
            start=nbs_start + 1, end=nbs_end,
            confidence=nbs_score,
            motifs=nbs_motifs,
        )
        domains.append(nbs_dom)
        all_motifs.extend(nbs_motifs)
        total_confidence += nbs_score * 0.35

    # --- LRR domain ---
    lrr_start, lrr_end = windows["LRR"]
    lrr_seq = seq[lrr_start:lrr_end]
    lrr_count, spec_positions, lrr_motifs = _count_lrr_repeats(lrr_seq)
    # Offset
    for m in lrr_motifs:
        m.start += lrr_start
        m.end   += lrr_start
    spec_abs = [p + lrr_start for p in spec_positions]

    if lrr_count >= 3:
        lrr_conf = min(1.0, lrr_count / 10.0)
        lrr_dom = NLRDomainAnnotation(
            domain_type="LRR",
            subtype="LRR-solenoid",
            start=lrr_start + 1, end=lrr_end,
            confidence=lrr_conf,
            motifs=lrr_motifs,
            lrr_count=lrr_count,
            specificity_positions=spec_abs,
        )
        domains.append(lrr_dom)
        all_motifs.extend(lrr_motifs)
        total_confidence += lrr_conf * 0.35

    # Is it an NLR?
    has_nbs = any(d.domain_type == "NBS" for d in domains)
    has_lrr = any(d.domain_type == "LRR" for d in domains)
    is_nlr  = has_nbs and has_lrr

    if nlr_class == "unknown" and is_nlr:
        nlr_class = "NL"

    # Collect critical positions
    critical = sorted({
        m.start for m in all_motifs if m.is_critical
    } | {
        m.end for m in all_motifs if m.is_critical
    })

    # Summary
    domain_str = " – ".join(d.domain_type for d in domains) or "none detected"
    lrr_note = f" ({lrr_count} LRR repeats, {len(spec_abs)} specificity positions)" if has_lrr else ""
    summary = (
        f"NLR class: {nlr_class}.  Architecture: {domain_str}{lrr_note}.  "
        f"NBS motifs: {len([m for m in all_motifs if m.name in _NBS_MOTIFS])}.  "
        f"Confidence: {total_confidence:.2f}."
    )

    return NLRAnnotation(
        is_nlr=is_nlr,
        nlr_class=nlr_class,
        confidence=min(1.0, total_confidence),
        domains=domains,
        all_motifs=all_motifs,
        critical_positions=critical,
        summary=summary,
    )


def flag_mutation_impact(
    position: int,     # 1-indexed
    wt_aa: str,
    mut_aa: str,
    nlr_ann: NLRAnnotation,
) -> MutationImpactAnnotation:
    """Predict the functional impact of a point mutation in an NLR.

    Checks:
    * Is position inside a critical NBS motif (P-loop, MHD, RNBS-A/D)?
    * Is it in LRR specificity-determining positions?
    * Is it in TIR W-box or ADP-ribosylase?

    Returns
    -------
    MutationImpactAnnotation with predicted effect and rationale.
    """
    # Find enclosing domain
    containing_domain: Optional[str] = None
    for d in nlr_ann.domains:
        if d.start <= position <= d.end:
            containing_domain = d.domain_type

    # Find enclosing motif
    containing_motif: Optional[NLRMotif] = None
    for m in nlr_ann.all_motifs:
        if m.start <= position <= m.end:
            containing_motif = m

    # Check specificity positions (LRR)
    is_spec = position in [
        p
        for d in nlr_ann.domains if d.domain_type == "LRR"
        for p in d.specificity_positions
    ]

    if containing_motif and containing_motif.is_critical:
        # P-loop K/T: any substitution = LOF
        if containing_motif.name == "P-loop":
            if wt_aa in "KT" and mut_aa not in "KT":
                return MutationImpactAnnotation(
                    position=position, wt_aa=wt_aa, mut_aa=mut_aa,
                    domain=containing_domain,
                    motif=containing_motif.name,
                    is_critical_site=True,
                    predicted_effect="LOF",
                    risk_level="high",
                    rationale=(
                        f"P-loop {wt_aa}{position}{mut_aa}: Walker-A lysine/threonine required "
                        "for ATP binding. This substitution likely abolishes NLR function."
                    ),
                )
            # Glycine mutations in P-loop also critical
            if wt_aa == "G":
                return MutationImpactAnnotation(
                    position=position, wt_aa=wt_aa, mut_aa=mut_aa,
                    domain=containing_domain, motif=containing_motif.name,
                    is_critical_site=True, predicted_effect="LOF",
                    risk_level="high",
                    rationale=(
                        f"P-loop Gly{position}{mut_aa}: conserved Gly required for backbone "
                        "flexibility at phosphate-binding loop. Likely LOF."
                    ),
                )
        # MHD: D→V/A = GOF autoactive; D→K/R = structural disruption
        if containing_motif.name == "MHD":
            if wt_aa == "D":
                if mut_aa in "AV":
                    return MutationImpactAnnotation(
                        position=position, wt_aa=wt_aa, mut_aa=mut_aa,
                        domain=containing_domain, motif="MHD",
                        is_critical_site=True, predicted_effect="GOF_autoactive",
                        risk_level="high",
                        rationale=(
                            f"MHD Asp{position}{mut_aa}: classic autoactivating mutation "
                            "(D→A in MHD = constitutive immune signalling in multiple NLRs). "
                            "Useful for gain-of-function studies; may trigger autoimmunity."
                        ),
                    )
                if mut_aa in "KR":
                    return MutationImpactAnnotation(
                        position=position, wt_aa=wt_aa, mut_aa=mut_aa,
                        domain=containing_domain, motif="MHD",
                        is_critical_site=True, predicted_effect="LOF",
                        risk_level="high",
                        rationale=(
                            f"MHD Asp{position}{mut_aa}: charge reversal at MHD aspartate "
                            "typically disrupts ARC2 structure. Predicted LOF."
                        ),
                    )
        # W-box
        if containing_motif.name == "W-box" and wt_aa == "W":
            return MutationImpactAnnotation(
                position=position, wt_aa=wt_aa, mut_aa=mut_aa,
                domain="TIR", motif="W-box",
                is_critical_site=True, predicted_effect="LOF",
                risk_level="high",
                rationale=(
                    f"W-box Trp{position}{mut_aa}: conserved tryptophan in TIR W-box "
                    "required for TIR-TIR self-association and cell-death signalling. "
                    "Likely abolishes TIR function and ETI."
                ),
            )
        # Other critical motifs
        return MutationImpactAnnotation(
            position=position, wt_aa=wt_aa, mut_aa=mut_aa,
            domain=containing_domain, motif=containing_motif.name,
            is_critical_site=True, predicted_effect="uncertain",
            risk_level="medium",
            rationale=(
                f"Position {position} ({wt_aa}→{mut_aa}) is in the critical motif "
                f"'{containing_motif.name}'. {containing_motif.note}"
            ),
        )

    if is_spec:
        return MutationImpactAnnotation(
            position=position, wt_aa=wt_aa, mut_aa=mut_aa,
            domain="LRR", motif="specificity-determining position",
            is_critical_site=False, predicted_effect="altered_specificity",
            risk_level="medium",
            rationale=(
                f"Position {position} ({wt_aa}→{mut_aa}) is at an LRR specificity-determining "
                "position (11/12/14 within repeat). Mutations here can change effector "
                "recognition specificity without disrupting overall structure."
            ),
        )

    return MutationImpactAnnotation(
        position=position, wt_aa=wt_aa, mut_aa=mut_aa,
        domain=containing_domain, motif=None,
        is_critical_site=False, predicted_effect="neutral",
        risk_level="low",
        rationale=(
            f"Position {position} ({wt_aa}→{mut_aa}) is in "
            f"{'domain ' + containing_domain if containing_domain else 'an unannotated region'}. "
            "No critical motif detected. Structural impact depends on local environment."
        ),
    )


def interpret_plddt_for_nlr(
    plddt_per_residue: List[float],
    nlr_ann: NLRAnnotation,
) -> Dict[str, object]:
    """Adjust pLDDT quality interpretation given NLR domain context.

    Returns dict with:
    - adjusted_mean: mean pLDDT excluding known disordered linkers
    - flagged_regions: list of (start, end, reason) for low-pLDDT concerns
    - suppressed_regions: list of (start, end, reason) for expected low pLDDT
    - quality_verdict: "good" | "acceptable" | "poor"
    """
    n = len(plddt_per_residue)
    if n == 0:
        return {"adjusted_mean": 0.0, "quality_verdict": "poor",
                "flagged_regions": [], "suppressed_regions": []}

    suppressed: List[Tuple] = []
    flagged:    List[Tuple] = []

    # TIR–NBS linker: residues between TIR end and NBS start are typically disordered
    tir_doms = [d for d in nlr_ann.domains if d.domain_type == "TIR"]
    nbs_doms = [d for d in nlr_ann.domains if d.domain_type == "NBS"]
    if tir_doms and nbs_doms:
        linker_start = tir_doms[0].end
        linker_end   = nbs_doms[0].start - 1
        if linker_end > linker_start:
            suppressed.append((
                linker_start, linker_end,
                "TIR–NBS linker: intrinsically disordered in plant NLRs; pLDDT < 70 is expected.",
            ))

    # LRR solenoid: globally somewhat lower pLDDT than globular domains
    lrr_doms = [d for d in nlr_ann.domains if d.domain_type == "LRR"]
    if lrr_doms:
        lrr_s, lrr_e = lrr_doms[0].start, lrr_doms[0].end
        lrr_plddt = plddt_per_residue[lrr_s - 1:lrr_e]
        lrr_mean = sum(lrr_plddt) / len(lrr_plddt) if lrr_plddt else 0.0
        if lrr_mean < 65:
            suppressed.append((
                lrr_s, lrr_e,
                f"LRR solenoid (mean pLDDT {lrr_mean:.1f}): solenoid structures often score "
                "lower with ESMFold; accept if LRR repeats are present and RMSD < 2 Å.",
            ))

    # Suppressed ranges (positions to exclude from adjusted mean)
    suppressed_idx: set = set()
    for s, e, _ in suppressed:
        suppressed_idx.update(range(max(0, s - 1), min(n, e)))

    active_scores = [
        v for i, v in enumerate(plddt_per_residue) if i not in suppressed_idx
    ]
    adjusted_mean = sum(active_scores) / len(active_scores) if active_scores else 0.0

    # Flag genuinely low-pLDDT regions outside suppressions
    i = 0
    while i < n:
        if plddt_per_residue[i] < 50 and i not in suppressed_idx:
            j = i
            while j < n and plddt_per_residue[j] < 50 and j not in suppressed_idx:
                j += 1
            if j - i >= 5:  # minimum 5 consecutive low-confidence residues
                region_mean = sum(plddt_per_residue[i:j]) / (j - i)
                flagged.append((
                    i + 1, j,
                    f"Unexpectedly low pLDDT ({region_mean:.1f}) outside known disordered linker. "
                    "May indicate structural problem.",
                ))
            i = j
        else:
            i += 1

    if adjusted_mean >= 70:
        verdict = "good"
    elif adjusted_mean >= 55:
        verdict = "acceptable"
    else:
        verdict = "poor"

    return {
        "adjusted_mean": round(adjusted_mean, 2),
        "quality_verdict": verdict,
        "flagged_regions": flagged,
        "suppressed_regions": suppressed,
    }
