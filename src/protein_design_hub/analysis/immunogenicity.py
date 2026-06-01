"""
immunogenicity.py — Immunogenicity risk analysis for therapeutic proteins and antibodies.

Predicts:
  - MHC-II T-cell epitopes via a simplified pan-DR PSSM (no external tools required)
  - Linear B-cell epitopes via the Parker hydrophilicity scale
  - Aggregation-prone regions (APR) via a TANGO-like beta-aggregation scale
  - An overall immunogenicity risk score (0-100)

All scoring is performed in pure Python using only numpy/standard library.
References:
  - Pan-DR PSSM anchors: Greenbaum et al. (2011) Immunome Research
  - Parker scale: Parker et al. (1986) Biochemistry
  - TANGO aggregation scale: Fernandez-Escamilla et al. (2004) Nature Biotechnology
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Standard amino acid set
# ---------------------------------------------------------------------------

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# MHC-II Pan-DR PSSM
# Per-position log-odds binding preferences (9 positions × 20 AAs).
# Anchor positions: P1 (pos 0), P4 (pos 3), P6 (pos 5), P9 (pos 8).
# Derived from Greenbaum 2011 pan-DR consensus.
# ---------------------------------------------------------------------------

MHC2_PSSM: Dict[int, Dict[str, float]] = {
    0: {  # P1 — primary anchor, strongly hydrophobic preferred
        "A": -0.2, "C":  0.1, "D": -2.1, "E": -2.3, "F":  2.5,
        "G": -0.8, "H":  0.2, "I":  1.8, "K": -2.3, "L":  2.8,
        "M":  2.1, "N": -0.7, "P": -3.0, "Q": -0.5, "R": -2.0,
        "S": -0.6, "T": -0.2, "V":  1.2, "W":  2.0, "Y":  2.2,
    },
    1: {  # P2 — secondary, relatively open
        "A":  0.3, "C":  0.0, "D": -0.5, "E": -0.8, "F":  0.5,
        "G":  0.2, "H":  0.1, "I":  0.8, "K": -0.5, "L":  0.8,
        "M":  0.5, "N":  0.0, "P": -2.0, "Q":  0.0, "R": -0.3,
        "S":  0.2, "T":  0.2, "V":  0.8, "W":  0.3, "Y":  0.3,
    },
    2: {  # P3 — open
        "A":  0.2, "C":  0.0, "D": -0.3, "E": -0.5, "F":  0.3,
        "G":  0.1, "H":  0.0, "I":  0.5, "K": -0.3, "L":  0.5,
        "M":  0.3, "N":  0.0, "P": -1.5, "Q":  0.0, "R": -0.2,
        "S":  0.1, "T":  0.1, "V":  0.5, "W":  0.2, "Y":  0.2,
    },
    3: {  # P4 — secondary anchor, small/hydrophobic preferred
        "A":  1.2, "C":  0.3, "D": -1.5, "E": -1.8, "F":  1.0,
        "G":  1.8, "H":  0.0, "I":  1.0, "K": -1.8, "L":  1.5,
        "M":  0.8, "N": -0.3, "P": -2.5, "Q": -0.3, "R": -1.5,
        "S":  0.5, "T":  0.5, "V":  1.2, "W":  0.5, "Y":  0.5,
    },
    4: {  # P5 — open
        "A":  0.2, "C":  0.0, "D": -0.2, "E": -0.3, "F":  0.3,
        "G":  0.1, "H":  0.0, "I":  0.5, "K": -0.2, "L":  0.5,
        "M":  0.3, "N":  0.0, "P": -1.0, "Q":  0.0, "R": -0.2,
        "S":  0.1, "T":  0.1, "V":  0.5, "W":  0.2, "Y":  0.2,
    },
    5: {  # P6 — auxiliary anchor
        "A":  0.0, "C":  0.0, "D": -1.0, "E": -1.2, "F":  1.2,
        "G": -0.2, "H":  0.0, "I":  1.0, "K": -1.2, "L":  1.5,
        "M":  1.0, "N": -0.2, "P": -2.0, "Q": -0.2, "R": -1.0,
        "S":  0.2, "T":  0.2, "V":  1.0, "W":  1.0, "Y":  1.0,
    },
    6: {  # P7 — open
        "A":  0.1, "C":  0.0, "D": -0.3, "E": -0.5, "F":  0.3,
        "G":  0.1, "H":  0.0, "I":  0.5, "K": -0.3, "L":  0.5,
        "M":  0.3, "N":  0.0, "P": -1.5, "Q":  0.0, "R": -0.2,
        "S":  0.1, "T":  0.1, "V":  0.5, "W":  0.2, "Y":  0.2,
    },
    7: {  # P8 — open
        "A":  0.1, "C":  0.0, "D": -0.3, "E": -0.5, "F":  0.3,
        "G":  0.1, "H":  0.0, "I":  0.5, "K": -0.3, "L":  0.5,
        "M":  0.3, "N":  0.0, "P": -1.5, "Q":  0.0, "R": -0.2,
        "S":  0.1, "T":  0.1, "V":  0.5, "W":  0.2, "Y":  0.2,
    },
    8: {  # P9 — C-terminal anchor, small preferred
        "A":  1.0, "C":  0.3, "D": -1.0, "E": -1.3, "F":  0.5,
        "G":  1.5, "H":  0.0, "I":  0.5, "K": -1.2, "L":  0.8,
        "M":  0.5, "N": -0.2, "P": -2.0, "Q": -0.2, "R": -1.2,
        "S":  1.0, "T":  1.0, "V":  1.0, "W":  0.2, "Y":  0.2,
    },
}

# Normalisation constants derived empirically from the PSSM matrix.
# Computed as sum of per-position min/max values across the 9 anchor positions.
_PSSM_RAW_MIN: float = sum(min(v.values()) for v in MHC2_PSSM.values())
_PSSM_RAW_MAX: float = sum(max(v.values()) for v in MHC2_PSSM.values())

# Thresholds (normalised 0-100 scale)
TCELL_HIGH_THRESHOLD:   float = 60.0
TCELL_MEDIUM_THRESHOLD: float = 45.0

# ---------------------------------------------------------------------------
# Parker (1986) hydrophilicity scale — B-cell linear epitope prediction
# ---------------------------------------------------------------------------

PARKER_SCALE: Dict[str, float] = {
    "A": -0.5, "R":  3.0, "N":  0.2, "D":  3.0, "C": -1.0,
    "Q":  0.2, "E":  3.0, "G":  0.0, "H": -0.5, "I": -1.8,
    "L": -1.8, "K":  3.0, "M": -1.3, "F": -2.5, "P":  0.0,
    "S":  0.3, "T": -0.4, "W": -3.4, "Y": -2.3, "V": -1.5,
}

BCELL_WINDOW:     int   = 7     # sliding window for B-cell scoring
BCELL_THRESHOLD:  float = 55.0  # normalised threshold (0-100)
BCELL_MERGE_GAP:  int   = 3     # merge stretches within this many residues
BCELL_MIN_LENGTH: int   = 6     # minimum reportable B-cell epitope length

# ---------------------------------------------------------------------------
# TANGO-like beta-aggregation scale — APR detection
# Fernandez-Escamilla et al. (2004) Nature Biotechnology
# ---------------------------------------------------------------------------

TANGO_SCALE: Dict[str, float] = {
    "A":  0.06, "R": -0.39, "N": -0.28, "D": -0.56, "C": -0.15,
    "Q": -0.28, "E": -0.50, "G": -0.03, "H": -0.04, "I":  0.49,
    "L":  0.49, "K": -0.57, "M":  0.30, "F":  0.73, "P": -0.44,
    "S": -0.04, "T": -0.04, "W":  0.53, "Y":  0.44, "V":  0.44,
}

APR_WINDOW:          int   = 5     # sliding window for APR scoring
APR_MIN_LENGTH:      int   = 5     # minimum consecutive residues to flag
APR_HIGH_THRESHOLD:  float = 0.30
APR_MEDIUM_THRESHOLD: float = 0.15

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TCellEpitope:
    """Predicted MHC-II T-cell epitope (9-mer core)."""

    position:      int    # 1-indexed start of the 9-mer
    peptide:       str    # 9-mer sequence
    score:         float  # binding score 0-100 (higher = more immunogenic)
    risk:          str    # "High" | "Medium" | "Low"
    context_15mer: str    # 15-mer context window centred on the 9-mer


@dataclass
class BCellEpitope:
    """Predicted linear B-cell epitope."""

    start:      int    # 1-indexed start
    end:        int    # 1-indexed end
    sequence:   str    # epitope sequence
    score:      float  # average normalised score across the window
    peak_score: float  # maximum score within the window


@dataclass
class AggregationRegion:
    """Aggregation-prone region (APR)."""

    start:    int    # 1-indexed start
    end:      int    # 1-indexed end
    sequence: str
    score:    float  # average TANGO-scale window score
    risk:     str    # "High" | "Medium"


@dataclass
class ImmunogenicityResult:
    """Full immunogenicity analysis result."""

    sequence:                str
    t_cell_epitopes:         List[TCellEpitope]    = field(default_factory=list)
    b_cell_epitopes:         List[BCellEpitope]    = field(default_factory=list)
    apr_regions:             List[AggregationRegion] = field(default_factory=list)
    per_residue_tcell:       List[float]           = field(default_factory=list)
    per_residue_bcell:       List[float]           = field(default_factory=list)
    immunogenicity_score:    float                 = 0.0
    risk_label:              str                   = "Low"
    t_cell_epitope_density:  float                 = 0.0
    humanness_adjustment:    float                 = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _normalise_pssm(raw: float) -> float:
    """Map a raw PSSM sum onto [0, 100]."""
    span = _PSSM_RAW_MAX - _PSSM_RAW_MIN  # 25.0
    return _clamp((raw - _PSSM_RAW_MIN) / span * 100.0)


def _sliding_window_average(values: List[float], window: int) -> List[float]:
    """
    Return a list of the same length as *values* where each element is the
    average of the *window*-wide window centred on that element.

    Edges are handled by narrowing the window (no zero-padding).
    """
    n = len(values)
    result: List[float] = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        result.append(sum(chunk) / len(chunk))
    return result


def _normalise_list(values: List[float], lo: float, hi: float) -> List[float]:
    """Min-max normalise *values* to [0, 100] using provided *lo* and *hi*."""
    span = hi - lo
    if span == 0.0:
        return [50.0] * len(values)
    return [_clamp((v - lo) / span * 100.0) for v in values]


def _risk_label_from_score(score: float) -> str:
    """Convert a 0-100 immunogenicity score to a risk label."""
    if score <= 20.0:
        return "Low"
    if score <= 45.0:
        return "Moderate"
    if score <= 70.0:
        return "High"
    return "Very High"


def _tcell_risk(score: float) -> str:
    if score >= TCELL_HIGH_THRESHOLD:
        return "High"
    if score >= TCELL_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def _context_15mer(sequence: str, pos: int) -> str:
    """
    Return a 15-mer context window around a 9-mer starting at *pos* (0-indexed).

    The window is centred such that the 9-mer occupies positions 3-11 of the
    15-mer (3 flanking residues on each side where possible).
    """
    start = max(0, pos - 3)
    end   = min(len(sequence), pos + 9 + 3)
    return sequence[start:end]


# ---------------------------------------------------------------------------
# MHC-II T-cell epitope scoring
# ---------------------------------------------------------------------------


def _score_mhc2_9mer(peptide: str) -> float:
    """
    Score a 9-mer peptide against the pan-DR PSSM.

    Returns a normalised score in [0, 100].  Unknown amino acids contribute 0.
    """
    raw = 0.0
    for pos, aa in enumerate(peptide):
        raw += MHC2_PSSM[pos].get(aa, 0.0)
    return _normalise_pssm(raw)


def _predict_tcell_epitopes(
    sequence: str,
    top_k: int,
) -> Tuple[List[TCellEpitope], List[float]]:
    """
    Scan all 9-mer windows in *sequence* and return:
      - a sorted list of the top-k T-cell epitopes (score >= medium threshold)
      - per-residue scores (max over all windows the residue participates in)
    """
    n = len(sequence)
    per_residue = [0.0] * n
    all_epitopes: List[TCellEpitope] = []

    if n < 9:
        return all_epitopes, per_residue

    for i in range(n - 8):
        peptide = sequence[i : i + 9]
        score   = _score_mhc2_9mer(peptide)

        # Update per-residue max
        for j in range(i, i + 9):
            if score > per_residue[j]:
                per_residue[j] = score

        if score >= TCELL_MEDIUM_THRESHOLD:
            all_epitopes.append(
                TCellEpitope(
                    position      = i + 1,  # 1-indexed
                    peptide       = peptide,
                    score         = round(score, 2),
                    risk          = _tcell_risk(score),
                    context_15mer = _context_15mer(sequence, i),
                )
            )

    all_epitopes.sort(key=lambda e: e.score, reverse=True)
    return all_epitopes[:top_k], per_residue


# ---------------------------------------------------------------------------
# B-cell linear epitope prediction (Parker scale)
# ---------------------------------------------------------------------------


def _predict_bcell_epitopes(
    sequence: str,
    top_k: int,
) -> Tuple[List[BCellEpitope], List[float]]:
    """
    Predict linear B-cell epitopes using the Parker hydrophilicity scale.

    Returns:
      - a list of predicted B-cell epitopes (length >= BCELL_MIN_LENGTH)
      - per-residue normalised Parker scores (0-100)
    """
    n = len(sequence)
    if n == 0:
        return [], []

    # Raw Parker scores per residue
    raw_parker = [PARKER_SCALE.get(aa, 0.0) for aa in sequence]

    # Sliding window average
    windowed = _sliding_window_average(raw_parker, BCELL_WINDOW)

    # Normalise to 0-100 using fixed-scale anchors
    # Parker scale range: approx -3.4 (Trp) to +3.0 (Arg/Asp/Glu/Lys)
    per_residue_norm = _normalise_list(windowed, lo=-3.4, hi=3.0)

    # Identify stretches above threshold
    above = [score >= BCELL_THRESHOLD for score in per_residue_norm]

    # Collect contiguous segments
    segments: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            segments.append((i, j - 1))  # inclusive, 0-indexed
            i = j
        else:
            i += 1

    # Merge segments within BCELL_MERGE_GAP residues
    merged: List[Tuple[int, int]] = []
    for seg in segments:
        if merged and seg[0] - merged[-1][1] <= BCELL_MERGE_GAP:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    # Build epitope objects
    epitopes: List[BCellEpitope] = []
    for seg in merged:
        start_0, end_0 = seg[0], seg[1]
        length = end_0 - start_0 + 1
        if length < BCELL_MIN_LENGTH:
            continue
        scores_in_seg = per_residue_norm[start_0 : end_0 + 1]
        epitopes.append(
            BCellEpitope(
                start      = start_0 + 1,                    # 1-indexed
                end        = end_0 + 1,                      # 1-indexed
                sequence   = sequence[start_0 : end_0 + 1],
                score      = round(sum(scores_in_seg) / len(scores_in_seg), 2),
                peak_score = round(max(scores_in_seg), 2),
            )
        )

    epitopes.sort(key=lambda e: e.peak_score, reverse=True)
    return epitopes[:top_k], per_residue_norm


# ---------------------------------------------------------------------------
# Aggregation-prone region (APR) detection
# ---------------------------------------------------------------------------


def _predict_apr_regions(sequence: str) -> List[AggregationRegion]:
    """
    Detect aggregation-prone regions using a TANGO-like beta-aggregation scale.

    Flags stretches of >= APR_MIN_LENGTH consecutive residues whose
    APR_WINDOW-averaged TANGO score exceeds APR_MEDIUM_THRESHOLD.
    """
    n = len(sequence)
    if n < APR_WINDOW:
        return []

    raw_tango = [TANGO_SCALE.get(aa, 0.0) for aa in sequence]
    windowed  = _sliding_window_average(raw_tango, APR_WINDOW)

    # Identify contiguous stretches above medium threshold
    above = [w >= APR_MEDIUM_THRESHOLD for w in windowed]

    segments: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if (j - i) >= APR_MIN_LENGTH:
                segments.append((i, j - 1))
            i = j
        else:
            i += 1

    aprs: List[AggregationRegion] = []
    for start_0, end_0 in segments:
        seg_scores = windowed[start_0 : end_0 + 1]
        avg_score  = sum(seg_scores) / len(seg_scores)
        risk       = "High" if avg_score >= APR_HIGH_THRESHOLD else "Medium"
        aprs.append(
            AggregationRegion(
                start    = start_0 + 1,
                end      = end_0 + 1,
                sequence = sequence[start_0 : end_0 + 1],
                score    = round(avg_score, 4),
                risk     = risk,
            )
        )

    return aprs


# ---------------------------------------------------------------------------
# Overall immunogenicity score
# ---------------------------------------------------------------------------


def _compute_immunogenicity_score(
    t_cell_epitopes: List[TCellEpitope],
    b_cell_epitopes: List[BCellEpitope],
    apr_regions:     List[AggregationRegion],
    sequence_length: int,
    n_total_9mers:   int,
) -> Tuple[float, float, str]:
    """
    Compute:
      - overall immunogenicity score (0-100)
      - T-cell epitope density (fraction of 9-mers above medium threshold)
      - risk label string

    Returns (score, t_cell_density, risk_label).
    """
    n_high_tcell   = sum(1 for e in t_cell_epitopes if e.risk == "High")
    n_medium_tcell = sum(1 for e in t_cell_epitopes if e.risk == "Medium")

    n_high_bcell   = sum(1 for e in b_cell_epitopes if e.peak_score >= 70.0)
    n_medium_bcell = sum(1 for e in b_cell_epitopes if e.peak_score < 70.0)

    n_high_apr   = sum(1 for a in apr_regions if a.risk == "High")
    n_medium_apr = sum(1 for a in apr_regions if a.risk == "Medium")

    tcell_component  = min(100.0, n_high_tcell * 15.0 + n_medium_tcell * 5.0)
    bcell_component  = min(100.0, n_high_bcell * 10.0 + n_medium_bcell * 3.0)
    apr_component    = min(100.0, n_high_apr   * 15.0 + n_medium_apr   * 5.0)

    score = (
        tcell_component * 0.50
        + bcell_component * 0.20
        + apr_component   * 0.30
    )
    score = _clamp(score)

    density = (len(t_cell_epitopes) / n_total_9mers) if n_total_9mers > 0 else 0.0

    return round(score, 2), round(density, 4), _risk_label_from_score(score)


# ---------------------------------------------------------------------------
# Humanness adjustment
# ---------------------------------------------------------------------------

# Amino acid frequencies in the human proteome (UniProt reviewed human entries,
# approximate).  Used to penalise sequences with unusual AA compositions.
_HUMAN_AA_FREQ: Dict[str, float] = {
    "A": 0.070, "R": 0.056, "N": 0.036, "D": 0.047, "C": 0.023,
    "Q": 0.039, "E": 0.062, "G": 0.065, "H": 0.026, "I": 0.044,
    "L": 0.101, "K": 0.058, "M": 0.021, "F": 0.037, "P": 0.050,
    "S": 0.083, "T": 0.055, "W": 0.013, "Y": 0.028, "V": 0.060,
}


def _humanness_penalty(sequence: str) -> float:
    """
    Compute a penalty [0, 100] reflecting how non-human the AA composition is.

    Uses the sum of squared differences between observed and expected human
    AA frequencies, scaled so that a random peptide gets ~50 and a human-like
    sequence gets ~0.
    """
    n = len(sequence)
    if n == 0:
        return 0.0

    observed: Dict[str, float] = {aa: 0.0 for aa in STANDARD_AA}
    for aa in sequence:
        if aa in observed:
            observed[aa] += 1.0
    obs_freq = {aa: cnt / n for aa, cnt in observed.items()}

    ssd = sum(
        (obs_freq.get(aa, 0.0) - exp) ** 2
        for aa, exp in _HUMAN_AA_FREQ.items()
    )
    # Scale: maximum theoretical SSD ≈ 0.10 (all residues are one non-human AA)
    penalty = _clamp(ssd / 0.10 * 100.0)
    return round(penalty, 2)


# ---------------------------------------------------------------------------
# Main public class
# ---------------------------------------------------------------------------


class ImmunogenicityAnalyzer:
    """
    Predict immunogenicity risk for therapeutic proteins and antibodies.

    All computations are performed in pure Python — no external tools,
    no network calls, no biopython MHC utilities.

    Example
    -------
    >>> analyzer = ImmunogenicityAnalyzer()
    >>> result = analyzer.analyze("EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAR")
    >>> print(result.risk_label, result.immunogenicity_score)
    """

    def analyze(
        self,
        sequence: str,
        top_k_tcell: int = 10,
        top_k_bcell: int = 5,
    ) -> ImmunogenicityResult:
        """
        Run a full immunogenicity analysis on *sequence*.

        Parameters
        ----------
        sequence:
            Single-letter amino-acid sequence (standard 20 AAs).  The
            sequence is automatically upper-cased and stripped of whitespace.
        top_k_tcell:
            Maximum number of T-cell epitopes to return (sorted by score).
        top_k_bcell:
            Maximum number of B-cell epitopes to return (sorted by peak score).

        Returns
        -------
        ImmunogenicityResult
            Populated dataclass with all predicted epitopes, per-residue
            profiles, and an overall risk score.  If the sequence is empty
            or contains non-standard characters throughout, the result will
            have score=0 and empty epitope lists.
        """
        sequence = sequence.strip().upper()

        # Validate — allow sequences that contain at least one valid AA
        clean_seq = "".join(aa for aa in sequence if aa in STANDARD_AA)
        if len(clean_seq) == 0:
            return ImmunogenicityResult(sequence=sequence)

        # Work on cleaned sequence (unknown AAs mapped to gaps / score=0)
        # We keep positions mapped back to original for 1-indexed output.
        # For simplicity, if there are non-standard chars, replace with G
        # (glycine, lowest-impact neutral AA) so window sizes stay correct.
        working_seq = "".join(aa if aa in STANDARD_AA else "G" for aa in sequence)

        n = len(working_seq)
        n_9mers = max(0, n - 8)

        # T-cell epitopes
        t_epitopes, per_res_tcell = _predict_tcell_epitopes(working_seq, top_k_tcell)

        # B-cell epitopes
        b_epitopes, per_res_bcell = _predict_bcell_epitopes(working_seq, top_k_bcell)

        # APR regions
        aprs = _predict_apr_regions(working_seq)

        # Overall score
        imm_score, density, risk_label = _compute_immunogenicity_score(
            t_epitopes, b_epitopes, aprs, n, n_9mers
        )

        # Humanness adjustment (informational — not fed back into imm_score)
        humanness_adj = _humanness_penalty(working_seq)

        return ImmunogenicityResult(
            sequence               = sequence,
            t_cell_epitopes        = t_epitopes,
            b_cell_epitopes        = b_epitopes,
            apr_regions            = aprs,
            per_residue_tcell      = [round(v, 2) for v in per_res_tcell],
            per_residue_bcell      = [round(v, 2) for v in per_res_bcell],
            immunogenicity_score   = imm_score,
            risk_label             = risk_label,
            t_cell_epitope_density = density,
            humanness_adjustment   = humanness_adj,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def analyze_immunogenicity(
    sequence:     str,
    top_k_tcell:  int = 10,
    top_k_bcell:  int = 5,
) -> ImmunogenicityResult:
    """
    Module-level convenience wrapper around :class:`ImmunogenicityAnalyzer`.

    Parameters
    ----------
    sequence:    Protein sequence (standard 20 AAs, upper- or lower-case).
    top_k_tcell: Maximum T-cell epitopes to return.
    top_k_bcell: Maximum B-cell epitopes to return.
    """
    return ImmunogenicityAnalyzer().analyze(
        sequence, top_k_tcell=top_k_tcell, top_k_bcell=top_k_bcell
    )
