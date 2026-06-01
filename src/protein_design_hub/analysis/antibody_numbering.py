"""
antibody_numbering.py — Pure-Python antibody variable domain analysis.

Supports chain-type detection (VH, VL-kappa, VL-lambda, VHH/nanobody), CDR/framework
annotation (Chothia, IMGT, Kabat schemes), germline V-gene assignment, humanness
scoring, H3 kink detection, and paratope-candidate identification.

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CDRAnnotation:
    """Annotation for a single CDR loop."""

    name: str
    """e.g. 'CDR-H1'"""
    start: int
    """1-indexed, inclusive."""
    end: int
    """1-indexed, inclusive."""
    sequence: str
    """Extracted CDR sequence."""
    length: int
    """len(sequence)."""
    scheme: str
    """'chothia' | 'imgt' | 'kabat'"""
    canonical_class: str
    """e.g. 'H1-1', 'H3-kinked', or '' if unknown."""


@dataclass
class GermlineAssignment:
    """Closest human germline V-gene and identity metrics."""

    gene: str
    """e.g. 'IGHV3-23'"""
    family: str
    """e.g. 'IGHV3'"""
    identity: float
    """Percent identity (0–100) over first 90 residues."""
    n_mutations: int
    """Number of substitutions vs. germline."""
    chain_type: str
    """'VH' | 'VL_kappa' | 'VL_lambda' | 'VHH'"""


@dataclass
class AntibodyAnnotation:
    """Full annotation of an antibody variable domain."""

    sequence: str
    chain_type: str
    """'VH' | 'VL_kappa' | 'VL_lambda' | 'VHH' | 'Unknown'"""
    chain_confidence: float
    """Confidence score 0–1 for chain_type call."""
    cdrs: List[CDRAnnotation]
    """Chothia-scheme CDRs."""
    cdrs_imgt: List[CDRAnnotation]
    """IMGT-scheme CDRs."""
    cdrs_kabat: List[CDRAnnotation]
    """Kabat-scheme CDRs."""
    frameworks: Dict[str, str]
    """FR1, FR2, FR3, FR4 sequences (Chothia-based intervals)."""
    germline: Optional[GermlineAssignment]
    humanness_score: float
    """0–100; 100 = perfect germline identity."""
    framework_mutations: List[str]
    """Positional substitutions vs. closest germline, e.g. ['V37I']."""
    h3_kink_motif: bool
    """True when CDR-H3 shows kinked-loop determinants."""
    paratope_candidates: List[int]
    """1-indexed positions most likely to contact antigen."""


# ---------------------------------------------------------------------------
# Germline reference sequences
# ---------------------------------------------------------------------------

# First ~95 residues of each V gene (FR1–CDR1–FR2–CDR2–FR3 consensus).
GERMLINE_SEQUENCES: Dict[str, str] = {
    # Heavy chain VH
    "IGHV1-2": (
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMT"
        "RDTSISTAYMELSSLRSEDTAVYYCAR"
    ),
    "IGHV1-69": (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRD"
        "NAKNSLYLQMNSLRAEDTAVYYCAR"
    ),
    "IGHV3-23": (
        "EVQLLESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRD"
        "NSKNTLYLQMNSLRAEDTAVYYCAR"
    ),
    "IGHV3-30": (
        "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRD"
        "NAKNSLYLQMNSLRAEDTAVYYCAR"
    ),
    "IGHV4-34": (
        "QVQLQQSGPGLVKPSQTLSLTCAISGDSVSSNSAAWNWIRQSPSRGLEWLGRTYYRSKWYNDYAVSVKSRITIN"
        "PDTSKNQFSLQLNSVTPEDTAVYYCAR"
    ),
    "IGHV1-46": (
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTLTTD"
        "SSTTTAYMELKSLQFDDTAVYYCAR"
    ),
    "IGHV2-5": (
        "QVTLKESGPVLVKPTETLTLTCTVSGFSLSNARMGWSWIRQPPGKALEWLAHIFSNDEKSYSTSLKSRLTISKD"
        "TSKSQVVLTMTNMDPVDTATYYCAR"
    ),
    "IGHV3-7": (
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMSWVRQAPGKGLEWVANIKQDGSEKYYVDSVKGRFTISRDN"
        "AKNSLYLQMNSLRAEDTAVYYCAR"
    ),
    # Light chain kappa
    "IGKV1-39": (
        "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTL"
        "TISRLEPEDFAVYYC"
    ),
    "IGKV3-20": (
        "EIVMTQSPATLSVSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPARFSGSGSGTDFTL"
        "TISSLEPEDFAVYYC"
    ),
    "IGKV1-5": (
        "DIQMTQSPSSLSASVGDRVTITCRASQGISSWLAWYQQKPEKAPKSLIYAASSLQSGVPSRFSGSGSGTDFTLT"
        "ISSLQPEDFATYYYC"
    ),
    "IGKV2-28": (
        "DIVMTQSPLSLPVTPGEPASISCRSSQSLLHSNGYNYLDWYLQKPGQSPQLLIYLGSNRASGVPDRFSGSGSGT"
        "DFTLKISRVEAEDVGVYYC"
    ),
    # Light chain lambda
    "IGLV1-44": (
        "QSVLTQPPSVSGAPGQRVTISCTGSSSNIGAGYDVHWYQQLPGTAPKLLIYGNDNRPSGVPDRFSGSKSGTSAS"
        "LAISGLQSEDEADYYC"
    ),
    "IGLV2-14": (
        "QSVLTQPPSASGTPGQRVTISCSGSSSNIGSNYVYWYQQLPGTAPKLLIYDNNKRPSGVPDRFSGSKSGTSASL"
        "AISGLQSEDEADYYC"
    ),
    # VHH nanobody
    "IGHV3-VHH": (
        "QVQLQESGGGSVQAGGSLRLSCAASGYIFSSYAMGWFRQAPGKEREFVSAISGSGNSTYYADSVKGRFTISQDN"
        "AKNTVYLQMNSLKPEDTAVYYC"
    ),
    "IGHV4-VHH": (
        "QVQLQESGGGSVQTGGSLRLSCAASGYIFSSYAMGWFRQAPGKERELVAAISSAGDSTYYADSVKGRFTISRDN"
        "AKNTLYLQMNSLKPEDTAVYYC"
    ),
}

# Map gene name prefix → chain type category used for filtering
_GERMLINE_CHAIN: Dict[str, str] = {
    "IGHV1-2": "VH",
    "IGHV1-69": "VH",
    "IGHV3-23": "VH",
    "IGHV3-30": "VH",
    "IGHV4-34": "VH",
    "IGHV1-46": "VH",
    "IGHV2-5": "VH",
    "IGHV3-7": "VH",
    "IGKV1-39": "VL_kappa",
    "IGKV3-20": "VL_kappa",
    "IGKV1-5": "VL_kappa",
    "IGKV2-28": "VL_kappa",
    "IGLV1-44": "VL_lambda",
    "IGLV2-14": "VL_lambda",
    "IGHV3-VHH": "VHH",
    "IGHV4-VHH": "VHH",
}

# ---------------------------------------------------------------------------
# CDR boundary tables
# ---------------------------------------------------------------------------
# Format: (start, end) — 1-indexed, inclusive, in Chothia sequential numbering.
# All boundaries are clamped to actual sequence length at annotation time.

_CDR_BOUNDARIES: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {
    "chothia": {
        "VH": {
            "CDR-H1": (26, 32),
            "CDR-H2": (52, 56),
            "CDR-H3": (95, 102),
        },
        "VL": {
            "CDR-L1": (24, 34),
            "CDR-L2": (50, 56),
            "CDR-L3": (89, 97),
        },
        "VHH": {
            "CDR1": (26, 32),
            "CDR2": (52, 56),
            "CDR3": (95, 102),
        },
    },
    "imgt": {
        "VH": {
            "CDR-H1": (27, 38),
            "CDR-H2": (56, 65),
            "CDR-H3": (105, 117),
        },
        "VL": {
            "CDR-L1": (27, 38),
            "CDR-L2": (56, 65),
            "CDR-L3": (105, 117),
        },
        "VHH": {
            "CDR1": (27, 38),
            "CDR2": (56, 65),
            "CDR3": (105, 117),
        },
    },
    "kabat": {
        "VH": {
            "CDR-H1": (31, 35),
            "CDR-H2": (49, 65),  # Kabat 1989: positions 49–65 (17 residues)
            "CDR-H3": (95, 102),
        },
        "VL": {
            "CDR-L1": (24, 34),
            "CDR-L2": (50, 56),
            "CDR-L3": (89, 97),
        },
        "VHH": {
            "CDR1": (31, 35),
            "CDR2": (50, 65),
            "CDR3": (95, 102),
        },
    },
}

# Framework region boundaries (Chothia, 1-indexed, inclusive) relative to the
# chain class.  These flank the Chothia CDR boundaries above.
_FR_BOUNDARIES: Dict[str, Dict[str, Tuple[int, int]]] = {
    "VH": {
        "FR1": (1, 25),
        "FR2": (33, 51),
        "FR3": (57, 94),
        "FR4": (103, 113),
    },
    "VL": {
        "FR1": (1, 23),
        "FR2": (35, 49),
        "FR3": (57, 88),
        "FR4": (98, 109),
    },
    "VHH": {
        "FR1": (1, 25),
        "FR2": (33, 51),
        "FR3": (57, 94),
        "FR4": (103, 113),
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pct_identity(a: str, b: str) -> Tuple[float, int]:
    """Return (percent_identity, n_mutations) for two sequences of possibly
    different lengths.  Only the overlapping prefix is compared."""
    overlap = min(len(a), len(b))
    if overlap == 0:
        return 0.0, 0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    mutations = overlap - matches
    return 100.0 * matches / overlap, mutations


def _chain_category(chain_type: str) -> str:
    """Map full chain type string to the category key used in CDR tables."""
    if chain_type in ("VH",):
        return "VH"
    if chain_type in ("VL_kappa", "VL_lambda"):
        return "VL"
    if chain_type == "VHH":
        return "VHH"
    return "VH"  # fallback


def _extract_cdr(sequence: str, start: int, end: int) -> Tuple[int, int, str]:
    """Return (clamped_start, clamped_end, subsequence) for 1-indexed bounds."""
    n = len(sequence)
    s = max(1, min(start, n))
    e = max(1, min(end, n))
    return s, e, sequence[s - 1 : e]


# ---------------------------------------------------------------------------
# Chain-type detection
# ---------------------------------------------------------------------------


def detect_chain_type(sequence: str) -> Tuple[str, float]:
    """Detect the antibody variable domain chain type from sequence alone.

    Uses length heuristics, conserved Cys positions (Ig disulfide), conserved
    Trp positions (FR2/FR4), and VHH-specific hydrophilic substitutions at the
    former VH/VL interface (positions ~44 and ~47 of the Chothia numbering).

    Parameters
    ----------
    sequence:
        Single-letter amino-acid sequence of the putative variable domain.

    Returns
    -------
    (chain_type, confidence)
        chain_type is one of ``'VH'``, ``'VL_kappa'``, ``'VL_lambda'``,
        ``'VHH'``, or ``'Unknown'``.
        confidence is a float in [0, 1].
    """
    seq = sequence.upper().strip()
    n = len(seq)

    if n < 90:
        return "Unknown", 0.1

    # ---- conserved Cys positions (Ig disulfide) --------------------------
    # Expected near 20% and 80% of sequence length.
    cys_20 = seq[int(n * 0.15) : int(n * 0.27)]
    cys_80 = seq[int(n * 0.72) : int(n * 0.88)]
    has_cys_pair = ("C" in cys_20) and ("C" in cys_80)

    # ---- conserved FR2 Trp (~36-40% into sequence) -----------------------
    trp_fr2_region = seq[int(n * 0.33) : int(n * 0.43)]
    has_fr2_trp = "W" in trp_fr2_region

    # ---- conserved FR4 Trp (~last 15% of sequence) ----------------------
    trp_fr4_region = seq[int(n * 0.82) :]
    has_fr4_trp = "W" in trp_fr4_region

    # ---- VHH hydrophilic substitutions at former VH/VL interface --------
    # Chothia position 44 is Gly in VH (G44) and Glu/Gln in VHH.
    # Chothia position 47 is Trp in VH (W47) and Gly/Phe/Ser in VHH.
    # For a ~120-residue domain these are at 0-indexed positions ~43 and ~46.
    # Use a narrow 3-residue window centred on each to be robust to length variation.
    _p44_centre = max(0, min(int(n * 0.358), n - 1))  # ≈ position 43 in 120-aa domain
    _p47_centre = max(0, min(int(n * 0.383), n - 1))  # ≈ position 46 in 120-aa domain
    p44_window = seq[max(0, _p44_centre - 1) : _p44_centre + 2]
    p47_window = seq[max(0, _p47_centre - 1) : _p47_centre + 2]
    # VH signature: G at pos 44, W at pos 47
    vh_p44_gly = "G" in p44_window
    vh_p47_trp = "W" in p47_window
    # VHH signature: E or Q at pos 44 (hydrophilic substitution), no W at pos 47
    vhh_p44 = any(aa in p44_window for aa in "EQ") and not vh_p44_gly
    vhh_p47_nontrp = not vh_p47_trp

    # ---- FR4 kappa/lambda C-terminal motif detection --------------------
    tail = seq[-15:]
    kappa_motif = any(m in tail for m in ("KLEIK", "KVEIK", "KLVIK", "KLEIR"))
    lambda_motif = any(m in tail for m in ("KVEIK", "KVEVK", "KLEDK", "KVEEK"))

    # ---- Scoring --------------------------------------------------------
    scores: Dict[str, float] = {"VH": 0.0, "VL_kappa": 0.0, "VL_lambda": 0.0, "VHH": 0.0}

    # Hard length rejection: antibody V-domains are 90–140 aa; longer is very
    # unlikely to be a single V domain.
    if n > 150:
        return "Unknown", 0.1

    # Length priors
    if 105 <= n <= 135:
        scores["VH"] += 0.3
        scores["VHH"] += 0.3
    if 95 <= n <= 120:
        scores["VL_kappa"] += 0.3
        scores["VL_lambda"] += 0.3

    # Conserved Cys pair (all Ig variable domains)
    if has_cys_pair:
        scores["VH"] += 0.2
        scores["VHH"] += 0.2
        scores["VL_kappa"] += 0.2
        scores["VL_lambda"] += 0.2

    # FR2 Trp (VH / VHH / VL all share it; VHH sometimes Phe/Tyr)
    if has_fr2_trp:
        scores["VH"] += 0.2
        scores["VL_kappa"] += 0.15
        scores["VL_lambda"] += 0.15
        scores["VHH"] += 0.1  # less common in VHH

    # FR4 Trp (present in all Ig variable domains)
    if has_fr4_trp:
        scores["VH"] += 0.15
        scores["VHH"] += 0.15
        scores["VL_kappa"] += 0.1
        scores["VL_lambda"] += 0.1

    # Position-44 / position-47 discriminator (most reliable VH vs VHH signal)
    if vh_p44_gly and vh_p47_trp:
        # Classic VH interface: G44 + W47
        scores["VH"] += 0.35
        scores["VHH"] -= 0.2
    elif vhh_p44 and vhh_p47_nontrp:
        # VHH hydrophilic substitutions: E/Q44 + no W47
        scores["VHH"] += 0.35
        scores["VH"] -= 0.2
    elif vhh_p47_nontrp:
        # Missing W47 alone is a weak VHH indicator
        scores["VHH"] += 0.1

    # C-terminal motif for kappa/lambda
    if kappa_motif:
        scores["VL_kappa"] += 0.3
    if lambda_motif:
        scores["VL_lambda"] += 0.3

    # Shorter sequences lean VL
    if n < 115:
        scores["VL_kappa"] += 0.1
        scores["VL_lambda"] += 0.1
        scores["VH"] -= 0.05
        scores["VHH"] -= 0.05

    # ---- Decision -------------------------------------------------------
    best = max(scores, key=lambda k: scores[k])
    best_score = scores[best]

    if best_score < 0.15:
        return "Unknown", 0.1

    # Normalize confidence to 0–1 range
    total = sum(max(v, 0) for v in scores.values()) or 1.0
    confidence = min(1.0, best_score / total * 2.0)

    return best, round(confidence, 3)


# ---------------------------------------------------------------------------
# CDR annotation
# ---------------------------------------------------------------------------


def _annotate_cdrs_for_scheme(
    sequence: str,
    chain_type: str,
    scheme: str,
) -> List[CDRAnnotation]:
    """Extract CDR annotations for a single numbering scheme.

    Parameters
    ----------
    sequence:
        Variable domain amino-acid sequence.
    chain_type:
        One of ``'VH'``, ``'VL_kappa'``, ``'VL_lambda'``, ``'VHH'``.
    scheme:
        ``'chothia'``, ``'imgt'``, or ``'kabat'``.

    Returns
    -------
    List of :class:`CDRAnnotation` objects, one per CDR.
    """
    cat = _chain_category(chain_type)
    if cat not in _CDR_BOUNDARIES[scheme]:
        return []

    boundaries = _CDR_BOUNDARIES[scheme][cat]
    cdrs: List[CDRAnnotation] = []

    for name, (raw_start, raw_end) in boundaries.items():
        if raw_start > len(sequence):
            continue  # CDR entirely beyond sequence end
        s, e, seq_fragment = _extract_cdr(sequence, raw_start, raw_end)
        canonical = ""
        if scheme == "chothia":
            canonical = _canonical_class(name, seq_fragment, chain_type)
        cdrs.append(
            CDRAnnotation(
                name=name,
                start=s,
                end=e,
                sequence=seq_fragment,
                length=len(seq_fragment),
                scheme=scheme,
                canonical_class=canonical,
            )
        )
    return cdrs


# ---------------------------------------------------------------------------
# Canonical class detection
# ---------------------------------------------------------------------------


def _canonical_class(cdr_name: str, cdr_seq: str, chain_type: str) -> str:
    """Return a Chothia canonical class label based on loop length.

    Parameters
    ----------
    cdr_name:
        CDR name as used in the Chothia boundary table (e.g. ``'CDR-H1'``).
    cdr_seq:
        Extracted CDR sequence.
    chain_type:
        Chain type string.

    Returns
    -------
    Canonical class string, e.g. ``'H1-1'``, ``'H3-kinked'``, or ``''``.
    """
    length = len(cdr_seq)

    # ----- VH canonical classes ------------------------------------------
    if cdr_name in ("CDR-H1", "CDR1") and chain_type in ("VH", "VHH"):
        if length == 5:
            return "H1-1"
        if length == 6:
            return "H1-2"
        if length == 7:
            return "H1-3"
        return "H1-?"

    if cdr_name in ("CDR-H2", "CDR2") and chain_type in ("VH", "VHH"):
        if length == 3:
            return "H2-1"
        if length == 4:
            return "H2-2"
        return "H2-?"

    if cdr_name in ("CDR-H3", "CDR3") and chain_type in ("VH", "VHH"):
        if length <= 6:
            return "H3-short"
        if length <= 12:
            return "H3-medium"
        return "H3-long"

    # ----- VL canonical classes ------------------------------------------
    if cdr_name == "CDR-L1" and chain_type in ("VL_kappa", "VL_lambda"):
        if 10 <= length <= 11:
            return "L1-1"
        if length == 12:
            return "L1-2"
        if 13 <= length <= 14:
            return "L1-3"
        return "L1-?"

    if cdr_name == "CDR-L2" and chain_type in ("VL_kappa", "VL_lambda"):
        return "L2-canonical"

    if cdr_name == "CDR-L3" and chain_type in ("VL_kappa", "VL_lambda"):
        if 8 <= length <= 9:
            return "L3-1"
        if 10 <= length <= 11:
            return "L3-2"
        return "L3-?"

    return ""


# ---------------------------------------------------------------------------
# H3 kink detection
# ---------------------------------------------------------------------------


def _detect_h3_kink(sequence: str, cdrs_chothia: List[CDRAnnotation]) -> bool:
    """Determine whether the CDR-H3 loop is likely in a kinked conformation.

    The kinked H3 is the most common H3 conformation in antibodies.  Its
    structural determinants include:

    * A Trp residue at position ~103 (Chothia) — the last FR3 residue adjacent
      to CDR-H3.
    * Asp or Asn near the C-terminal base of CDR-H3 (the WGXXD/WGXXY kink
      motif).

    This function applies a simple sequence-level heuristic.

    Parameters
    ----------
    sequence:
        Full variable domain sequence.
    cdrs_chothia:
        Chothia-scheme CDR annotations already computed.

    Returns
    -------
    True if the CDR-H3 loop shows kink determinants.
    """
    h3 = next(
        (c for c in cdrs_chothia if c.name in ("CDR-H3", "CDR3")),
        None,
    )
    if h3 is None or len(h3.sequence) == 0:
        return False

    # Check for Trp or Tyr at FR3 position 103 (canonical kink determinant).
    # W103 is most common; Y103 also supports kinked conformation in some VH families.
    # The full WGXXD/WGXXY kink motif requires a W-G dipeptide immediately before CDR-H3.
    pre_start = max(0, h3.start - 3)  # 0-indexed window
    pre_seq = sequence[pre_start : h3.start - 1].upper()  # exclusive of CDR-H3
    has_pre_wy = any(aa in pre_seq for aa in "WY")
    # Additional check: canonical W-G dipeptide (positions 103-104 in Chothia VH)
    pre2_start = max(0, h3.start - 2)
    pre2_seq = sequence[pre2_start : h3.start - 1].upper()
    has_wg_motif = len(pre2_seq) >= 2 and pre2_seq[0] == "W" and pre2_seq[1] == "G"

    # Check for D or N in the last 3 residues of CDR-H3 (WGXXD/WGXXY kink stabiliser)
    cdr_seq = h3.sequence.upper()
    tail = cdr_seq[-3:] if len(cdr_seq) >= 3 else cdr_seq
    has_dn_tail = any(aa in tail for aa in "DN")

    # Kinked if: (W/Y at FR3) AND (D/N at H3 tail), with bonus if full W-G motif present
    return has_pre_wy and has_dn_tail or has_wg_motif


# ---------------------------------------------------------------------------
# Framework annotation
# ---------------------------------------------------------------------------


def _annotate_frameworks(sequence: str, chain_type: str) -> Dict[str, str]:
    """Extract FR1–FR4 sequences using Chothia-based positional boundaries.

    Parameters
    ----------
    sequence:
        Variable domain amino-acid sequence.
    chain_type:
        ``'VH'``, ``'VL_kappa'``, ``'VL_lambda'``, or ``'VHH'``.

    Returns
    -------
    Dictionary ``{'FR1': str, 'FR2': str, 'FR3': str, 'FR4': str}``.
    """
    cat = _chain_category(chain_type)
    boundaries = _FR_BOUNDARIES.get(cat, _FR_BOUNDARIES["VH"])
    result: Dict[str, str] = {}
    for fr_name, (raw_start, raw_end) in boundaries.items():
        if raw_start > len(sequence):
            result[fr_name] = ""
            continue
        _, _, fragment = _extract_cdr(sequence, raw_start, raw_end)
        result[fr_name] = fragment
    return result


# ---------------------------------------------------------------------------
# Germline assignment
# ---------------------------------------------------------------------------


def _assign_germline(
    sequence: str,
    chain_type: str,
) -> Optional[GermlineAssignment]:
    """Find the closest human germline V gene by sequence identity.

    Compares the first 90 residues of *sequence* against all germlines of the
    matching chain category.

    Parameters
    ----------
    sequence:
        Variable domain sequence.
    chain_type:
        Detected chain type (e.g. ``'VH'``).

    Returns
    -------
    :class:`GermlineAssignment` or ``None`` if chain_type is ``'Unknown'``.
    """
    if chain_type == "Unknown":
        return None

    query = sequence[:90].upper()

    # Determine allowed germline categories
    if chain_type == "VHH":
        allowed = {"VHH"}
    elif chain_type in ("VH",):
        allowed = {"VH"}
    elif chain_type == "VL_kappa":
        allowed = {"VL_kappa"}
    else:  # VL_lambda
        allowed = {"VL_lambda"}

    best_gene: Optional[str] = None
    best_identity = -1.0
    best_mutations = 0

    for gene, germ_seq in GERMLINE_SEQUENCES.items():
        if _GERMLINE_CHAIN.get(gene) not in allowed:
            continue
        ref = germ_seq[:90].upper()
        identity, mutations = _pct_identity(query, ref)
        if identity > best_identity:
            best_identity = identity
            best_mutations = mutations
            best_gene = gene

    if best_gene is None:
        return None

    family = "-".join(best_gene.split("-")[:1] + [best_gene.split("-")[1][0]]) if "-" in best_gene else best_gene

    return GermlineAssignment(
        gene=best_gene,
        family=family,
        identity=round(best_identity, 2),
        n_mutations=best_mutations,
        chain_type=chain_type,
    )


# ---------------------------------------------------------------------------
# Framework mutations
# ---------------------------------------------------------------------------


def _compute_framework_mutations(
    sequence: str,
    germline: Optional[GermlineAssignment],
) -> List[str]:
    """List positional substitutions between query and closest germline.

    Only the first 90 residues (FR1–FR3 region) are compared.

    Parameters
    ----------
    sequence:
        Variable domain sequence.
    germline:
        Result of :func:`_assign_germline`.

    Returns
    -------
    List of strings like ``['V37I', 'A40G']``.
    """
    if germline is None:
        return []

    germ_seq = GERMLINE_SEQUENCES.get(germline.gene, "")
    if not germ_seq:
        return []

    query = sequence[:90].upper()
    ref = germ_seq[:90].upper()
    overlap = min(len(query), len(ref))

    mutations: List[str] = []
    for i in range(overlap):
        if query[i] != ref[i]:
            mutations.append(f"{ref[i]}{i + 1}{query[i]}")

    return mutations


# ---------------------------------------------------------------------------
# Paratope candidate identification
# ---------------------------------------------------------------------------


def _identify_paratope_candidates(
    cdrs_chothia: List[CDRAnnotation],
    chain_type: str,
) -> List[int]:
    """Identify CDR positions most likely to contact antigen.

    Strategy (based on MacCallum 1996 / Lo Conte 1999 paratope analysis):

    * **VH / VHH CDR-H3 (CDR3):** all residues included (dominant contributor).
    * **VH CDR-H1, CDR-H2:** middle 60% of each loop (trim 1 residue from each
      end) are included.
    * **VL CDR-L3:** all residues included.
    * **VL CDR-L1:** middle 60% included.
    * **VL CDR-L2:** excluded (minimal contacts, ~5%).

    Parameters
    ----------
    cdrs_chothia:
        Chothia-scheme CDR annotations.
    chain_type:
        Chain type string.

    Returns
    -------
    Sorted, deduplicated list of 1-indexed sequence positions.
    """
    candidates: set[int] = set()

    for cdr in cdrs_chothia:
        name = cdr.name
        length = cdr.length
        s = cdr.start  # 1-indexed

        if length == 0:
            continue

        if chain_type in ("VH", "VHH"):
            if name in ("CDR-H3", "CDR3"):
                # All residues
                candidates.update(range(s, s + length))
            elif name in ("CDR-H1", "CDR1", "CDR-H2", "CDR2"):
                # Middle 60% — trim 1 from each end (at minimum include centre)
                trim = max(1, length // 5)  # ~20% each side → middle ~60%
                start_trim = s + trim
                end_trim = s + length - trim - 1
                if start_trim <= end_trim:
                    candidates.update(range(start_trim, end_trim + 1))
                else:
                    # Very short CDR — include all
                    candidates.update(range(s, s + length))

        elif chain_type in ("VL_kappa", "VL_lambda"):
            if name == "CDR-L3":
                candidates.update(range(s, s + length))
            elif name == "CDR-L1":
                trim = max(1, length // 5)
                start_trim = s + trim
                end_trim = s + length - trim - 1
                if start_trim <= end_trim:
                    candidates.update(range(start_trim, end_trim + 1))
                else:
                    candidates.update(range(s, s + length))
            elif name == "CDR-L2":
                # CDR-L2 contacts antigen in ~10-20% of Ab:Ag interfaces
                # (Lo Conte et al. 1999; MacCallum et al. 1996).
                # Include last 3 residues (C-terminal end is most variable/accessible).
                candidates.update(range(s + max(0, length - 3), s + length))

    return sorted(candidates)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def annotate_antibody(sequence: str) -> AntibodyAnnotation:
    """Fully annotate an antibody variable domain sequence.

    Performs chain-type detection, CDR annotation (Chothia, IMGT, Kabat),
    framework extraction, germline assignment, humanness scoring, H3 kink
    detection, and paratope candidate identification — all without external
    dependencies.

    Parameters
    ----------
    sequence:
        Single-letter amino-acid sequence.  Non-standard characters are silently
        kept but may reduce annotation accuracy.  Case-insensitive.

    Returns
    -------
    :class:`AntibodyAnnotation` with all fields populated.  For non-antibody
    sequences the ``chain_type`` will be ``'Unknown'`` and CDR/germline fields
    will be empty / None.
    """
    seq = sequence.upper().strip()

    # ---- Chain type detection -------------------------------------------
    chain_type, confidence = detect_chain_type(seq)

    # ---- CDR annotation (all three schemes) -----------------------------
    if chain_type == "Unknown":
        cdrs_chothia = []
        cdrs_imgt = []
        cdrs_kabat = []
    else:
        cdrs_chothia = _annotate_cdrs_for_scheme(seq, chain_type, "chothia")
        cdrs_imgt = _annotate_cdrs_for_scheme(seq, chain_type, "imgt")
        cdrs_kabat = _annotate_cdrs_for_scheme(seq, chain_type, "kabat")

    # ---- Framework annotation -------------------------------------------
    frameworks = _annotate_frameworks(seq, chain_type) if chain_type != "Unknown" else {}

    # ---- Germline assignment --------------------------------------------
    germline = _assign_germline(seq, chain_type)

    # ---- Humanness score ------------------------------------------------
    humanness_score = germline.identity if germline is not None else 0.0

    # ---- Framework mutations --------------------------------------------
    fw_mutations = _compute_framework_mutations(seq, germline)

    # ---- H3 kink detection ----------------------------------------------
    h3_kink = _detect_h3_kink(seq, cdrs_chothia) if chain_type in ("VH", "VHH") else False

    # ---- Paratope candidates --------------------------------------------
    paratope_candidates = _identify_paratope_candidates(cdrs_chothia, chain_type)

    return AntibodyAnnotation(
        sequence=seq,
        chain_type=chain_type,
        chain_confidence=confidence,
        cdrs=cdrs_chothia,
        cdrs_imgt=cdrs_imgt,
        cdrs_kabat=cdrs_kabat,
        frameworks=frameworks,
        germline=germline,
        humanness_score=humanness_score,
        framework_mutations=fw_mutations,
        h3_kink_motif=h3_kink,
        paratope_candidates=paratope_candidates,
    )


def get_cdr_sequences(sequence: str, scheme: str = "chothia") -> Dict[str, str]:
    """Quick helper: return a ``{cdr_name: sequence}`` mapping.

    Parameters
    ----------
    sequence:
        Variable domain amino-acid sequence.
    scheme:
        Numbering scheme: ``'chothia'`` (default), ``'imgt'``, or ``'kabat'``.

    Returns
    -------
    Dictionary mapping CDR names (e.g. ``'CDR-H1'``) to their sequences.
    Positions that exceed the sequence length are returned as empty strings.
    """
    ann = annotate_antibody(sequence)
    scheme = scheme.lower()
    if scheme == "chothia":
        source = ann.cdrs
    elif scheme == "imgt":
        source = ann.cdrs_imgt
    elif scheme == "kabat":
        source = ann.cdrs_kabat
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Use 'chothia', 'imgt', or 'kabat'.")

    return {cdr.name: cdr.sequence for cdr in source}


def summarize_annotation(ann: AntibodyAnnotation) -> str:
    """Return a human-readable multi-line summary of an :class:`AntibodyAnnotation`.

    Parameters
    ----------
    ann:
        Annotation object returned by :func:`annotate_antibody`.

    Returns
    -------
    Multi-line string suitable for printing to a terminal or Streamlit ``st.text``.
    """
    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("Antibody Variable Domain Annotation Summary")
    lines.append("=" * 60)
    lines.append(f"Chain type   : {ann.chain_type}  (confidence {ann.chain_confidence:.2f})")
    lines.append(f"Sequence len : {len(ann.sequence)} aa")
    lines.append(f"Humanness    : {ann.humanness_score:.1f}%")
    lines.append(f"H3 kink      : {'Yes' if ann.h3_kink_motif else 'No'}")
    lines.append("")

    if ann.germline:
        g = ann.germline
        lines.append(f"Germline V-gene : {g.gene} (family {g.family})")
        lines.append(f"  Identity        : {g.identity:.1f}%")
        lines.append(f"  FR mutations    : {g.n_mutations}")
    else:
        lines.append("Germline V-gene : Not assigned")
    lines.append("")

    lines.append("CDRs (Chothia):")
    for cdr in ann.cdrs:
        cls_str = f"  [{cdr.canonical_class}]" if cdr.canonical_class else ""
        lines.append(
            f"  {cdr.name:<10}  pos {cdr.start:>3}-{cdr.end:<3}  "
            f"len {cdr.length:<3}  seq {cdr.sequence}{cls_str}"
        )
    lines.append("")

    lines.append("CDRs (IMGT):")
    for cdr in ann.cdrs_imgt:
        lines.append(
            f"  {cdr.name:<10}  pos {cdr.start:>3}-{cdr.end:<3}  "
            f"len {cdr.length:<3}  seq {cdr.sequence}"
        )
    lines.append("")

    lines.append("CDRs (Kabat):")
    for cdr in ann.cdrs_kabat:
        lines.append(
            f"  {cdr.name:<10}  pos {cdr.start:>3}-{cdr.end:<3}  "
            f"len {cdr.length:<3}  seq {cdr.sequence}"
        )
    lines.append("")

    lines.append("Framework regions (Chothia):")
    for fr_name, fr_seq in ann.frameworks.items():
        lines.append(f"  {fr_name}: {fr_seq}")
    lines.append("")

    if ann.framework_mutations:
        lines.append(f"Framework mutations vs. germline: {', '.join(ann.framework_mutations)}")
    else:
        lines.append("Framework mutations vs. germline: none")
    lines.append("")

    if ann.paratope_candidates:
        cands_str = ", ".join(str(p) for p in ann.paratope_candidates)
        lines.append(f"Paratope candidates ({len(ann.paratope_candidates)} residues):")
        lines.append(f"  Positions: {cands_str}")
    else:
        lines.append("Paratope candidates: none identified")

    lines.append("=" * 60)
    return "\n".join(lines)
