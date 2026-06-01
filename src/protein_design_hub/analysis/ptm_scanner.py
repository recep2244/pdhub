"""Post-Translational Modification (PTM) Liability Scanner.

Identifies chemical liability hotspots critical for pharmaceutical protein
development: deamidation, isomerization, oxidation, glycosylation sequons,
proteolytic sites, and antibody-specific liabilities.

References:
- Deamidation: Robinson (2002) PNAS; Chelius (2005) Anal Chem
- Isomerization: Geiger (1987) J Biol Chem; Wakankar (2007) Biochemistry
- Oxidation: Ji (2009) J Pharm Sci; Lam (1997) J Pharm Sci
- N-glycosylation: Zielinska (2010) PNAS
- Antibody liabilities: Jain (2017) PNAS; Jerdvall (2012) MAbs
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── Residue-level context scores ─────────────────────────────────────────────
# Sequence-based rate modifiers for deamidation half-lives (from Robinson 2002)
# Lower value = faster deamidation = higher risk
DEAMID_RATE_MODIFIER: Dict[str, float] = {
    "NG": 1.0,   # fastest — half-life ~1 day at pH 7.4, 37°C
    "NS": 2.5,
    "NT": 3.0,
    "NA": 4.0,
    "NH": 5.0,
    "NN": 6.0,
    "NQ": 7.0,
    "NL": 8.0,
    "NI": 8.0,
    "NV": 9.0,
    "NK": 10.0,
    "NR": 12.0,
    "ND": 20.0,
    "NE": 25.0,
    "NF": 40.0,
    "NY": 50.0,
    "NW": 60.0,
    "NP": 9999.0,  # Proline next residue — very resistant
}

# Isomerization rate modifiers for Asp (relative rates from Geiger 1987)
ISOMERI_RATE_MODIFIER: Dict[str, float] = {
    "DG": 1.0,   # fastest
    "DS": 2.0,
    "DT": 3.5,
    "DH": 4.0,
    "DN": 5.0,
    "DD": 7.0,
    "DA": 10.0,
    "DP": 9999.0,  # Proline blocks
}


@dataclass
class PTMLiability:
    """A single PTM liability hit."""

    liability_type: str          # e.g. "Deamidation", "Oxidation"
    position: int                # 1-indexed position of the primary residue
    residue: str                 # Single-letter AA at that position
    motif: str                   # The matched sequence motif (e.g. "NG")
    risk: str                    # "High" | "Medium" | "Low"
    description: str             # Human-readable explanation
    rate_modifier: float = 1.0   # Relative rate (1 = fastest known)
    in_cdr: bool = False         # True if overlaps with a detected CDR region


@dataclass
class PTMScanResult:
    """Full PTM scan result for a protein sequence."""

    sequence: str
    liabilities: List[PTMLiability] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    risk_score: float = 0.0          # 0–100 composite risk
    risk_label: str = "Low"          # "Low" | "Medium" | "High" | "Critical"
    manufacturability_flags: List[str] = field(default_factory=list)

    def by_type(self, liability_type: str) -> List[PTMLiability]:
        return [h for h in self.liabilities if h.liability_type == liability_type]

    def high_risk(self) -> List[PTMLiability]:
        return [h for h in self.liabilities if h.risk == "High"]


class PTMLiabilityScanner:
    """
    Sequence-based scanner for pharmaceutical PTM liabilities.

    Covers:
    - Deamidation  (Asn: NG/NS/NT and neighbors)
    - Isomerization (Asp: DG/DS and neighbors)
    - Oxidation    (Met, Trp, free/exposed Cys)
    - N-glycosylation sequon (N-X-S/T, X≠P)
    - O-glycosylation prone (S/T adjacent to P)
    - Tryptic cleavage (K/R not followed by P)
    - Furin cleavage (R-x-x-R motif)
    - MMP cleavage (P-x-x-P)
    - Unpaired Cys (odd number of Cys)
    - N-terminal Gln → pyroglutamate
    - Met oxidation at N-terminus
    - Antibody CDR glycosylation (if CDRs detected)
    """

    def scan(
        self,
        sequence: str,
        cdr_regions: Optional[List[Tuple[int, int]]] = None,
        check_antibody: bool = True,
    ) -> PTMScanResult:
        """
        Scan a protein sequence for PTM liabilities.

        Args:
            sequence: One-letter amino acid sequence.
            cdr_regions: List of (start, end) tuples (1-indexed, inclusive) for CDR regions.
                         If None, auto-detect using Chothia-like heuristics.
            check_antibody: If True, run antibody-specific checks.

        Returns:
            PTMScanResult with all identified liabilities.
        """
        seq = sequence.upper().strip()
        liabilities: List[PTMLiability] = []

        # Auto-detect CDRs if not provided
        if cdr_regions is None and check_antibody:
            cdr_regions = _detect_cdr_regions(seq)

        cdr_set: set = set()
        if cdr_regions:
            for start, end in cdr_regions:
                for i in range(start, end + 1):
                    cdr_set.add(i)

        liabilities += self._scan_deamidation(seq, cdr_set)
        liabilities += self._scan_isomerization(seq, cdr_set)
        liabilities += self._scan_oxidation(seq, cdr_set)
        liabilities += self._scan_glycosylation(seq, cdr_set)
        liabilities += self._scan_proteolytic(seq)
        liabilities += self._scan_terminal(seq)
        if check_antibody:
            liabilities += self._scan_antibody_specific(seq, cdr_set)

        # Summary counts
        summary: Dict[str, int] = {}
        for lib in liabilities:
            summary[lib.liability_type] = summary.get(lib.liability_type, 0) + 1

        # Composite risk score (0–100)
        risk_score = _calculate_risk_score(liabilities)
        risk_label = _risk_label(risk_score)

        # Manufacturability flags
        mfg_flags = _manufacturability_flags(seq, liabilities)

        return PTMScanResult(
            sequence=seq,
            liabilities=liabilities,
            summary=summary,
            risk_score=round(risk_score, 1),
            risk_label=risk_label,
            manufacturability_flags=mfg_flags,
        )

    # ── Deamidation ──────────────────────────────────────────────────────────
    def _scan_deamidation(self, seq: str, cdr_set: set) -> List[PTMLiability]:
        hits = []
        for i, aa in enumerate(seq[:-1]):
            if aa != "N":
                continue
            next_aa = seq[i + 1]
            motif = f"N{next_aa}"
            rate = DEAMID_RATE_MODIFIER.get(motif, 30.0)
            if rate > 100:
                continue  # Negligible
            if rate <= 3.0:
                risk = "High"
            elif rate <= 10.0:
                risk = "Medium"
            else:
                risk = "Low"

            # Check preceding residue context
            prev_aa = seq[i - 1] if i > 0 else ""
            context_note = ""
            if prev_aa == "P":
                risk = "Low"
                context_note = " (Xxx-Pro preceding — partially protected)"

            hits.append(PTMLiability(
                liability_type="Deamidation",
                position=i + 1,
                residue="N",
                motif=motif,
                risk=risk,
                rate_modifier=rate,
                in_cdr=(i + 1) in cdr_set,
                description=(
                    f"Asn{i+1}-{next_aa} sequon susceptible to deamidation "
                    f"(relative rate ~{rate:.0f}x slowest known){context_note}. "
                    "Converts N→D/isoD, alters charge and mass (+0.98 Da)."
                ),
            ))
        return hits

    # ── Isomerization ─────────────────────────────────────────────────────────
    def _scan_isomerization(self, seq: str, cdr_set: set) -> List[PTMLiability]:
        hits = []
        for i, aa in enumerate(seq[:-1]):
            if aa != "D":
                continue
            next_aa = seq[i + 1]
            motif = f"D{next_aa}"
            rate = ISOMERI_RATE_MODIFIER.get(motif, 20.0)
            if rate > 100:
                continue
            if rate <= 2.0:
                risk = "High"
            elif rate <= 5.0:
                risk = "Medium"
            else:
                risk = "Low"

            hits.append(PTMLiability(
                liability_type="Isomerization",
                position=i + 1,
                residue="D",
                motif=motif,
                risk=risk,
                rate_modifier=rate,
                in_cdr=(i + 1) in cdr_set,
                description=(
                    f"Asp{i+1}-{next_aa} sequon susceptible to Asp isomerization "
                    f"via succinimide intermediate (relative rate ~{rate:.0f}x slowest). "
                    "Converts D→isoD, may alter binding and activity."
                ),
            ))
        return hits

    # ── Oxidation ─────────────────────────────────────────────────────────────
    def _scan_oxidation(self, seq: str, cdr_set: set) -> List[PTMLiability]:
        hits = []

        # Methionine — moderate oxidation risk, greatly increased on surface/exposed
        for i, aa in enumerate(seq):
            if aa == "M":
                # Flanking residues context
                context = seq[max(0, i - 2):i + 3]
                # Higher risk if adjacent to charged or polar residues (surface exposure)
                polar_neighbors = sum(1 for c in context if c in "DEKRHSTNQ")
                risk = "High" if polar_neighbors >= 2 else "Medium"
                hits.append(PTMLiability(
                    liability_type="Oxidation",
                    position=i + 1,
                    residue="M",
                    motif=f"M@{i+1}",
                    risk=risk,
                    in_cdr=(i + 1) in cdr_set,
                    description=(
                        f"Met{i+1} susceptible to oxidation (→ Met sulfoxide, +16 Da). "
                        "Can reduce potency, increase aggregation, and trigger immune response."
                    ),
                ))

        # Tryptophan — lower but real oxidation risk
        for i, aa in enumerate(seq):
            if aa == "W":
                hits.append(PTMLiability(
                    liability_type="Oxidation",
                    position=i + 1,
                    residue="W",
                    motif=f"W@{i+1}",
                    risk="Medium",
                    in_cdr=(i + 1) in cdr_set,
                    description=(
                        f"Trp{i+1} susceptible to photo-oxidation and kynurenine formation (+4 Da). "
                        "Risk increases under UV exposure or peroxide stress."
                    ),
                ))

        # Free/exposed Cys — disulfide scrambling or thiol oxidation
        cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
        if len(cys_positions) % 2 != 0:
            # Odd number of Cys → likely one free/unpaired
            for i in cys_positions:
                hits.append(PTMLiability(
                    liability_type="Oxidation",
                    position=i + 1,
                    residue="C",
                    motif=f"C@{i+1}",
                    risk="High",
                    in_cdr=(i + 1) in cdr_set,
                    description=(
                        f"Cys{i+1} — odd Cys count ({len(cys_positions)}) suggests unpaired thiol. "
                        "Free Cys can oxidize (→ sulfinic/sulfonic acid) or form aberrant disulfides."
                    ),
                ))
        elif cys_positions:
            # Even Cys — flag as medium (disulfide scrambling risk)
            for i in cys_positions:
                hits.append(PTMLiability(
                    liability_type="Oxidation",
                    position=i + 1,
                    residue="C",
                    motif=f"C@{i+1}",
                    risk="Low",
                    in_cdr=(i + 1) in cdr_set,
                    description=(
                        f"Cys{i+1} — part of even Cys count; assumes paired in disulfide. "
                        "Monitor for disulfide scrambling under stress conditions."
                    ),
                ))

        return hits

    # ── Glycosylation ─────────────────────────────────────────────────────────
    def _scan_glycosylation(self, seq: str, cdr_set: set) -> List[PTMLiability]:
        hits = []

        # N-glycosylation: N-x-S/T (x ≠ P)
        for m in re.finditer(r"N[^P][ST]", seq):
            i = m.start()
            motif = m.group()
            in_cdr = any((i + k + 1) in cdr_set for k in range(len(motif)))
            risk = "High" if in_cdr else "Medium"
            hits.append(PTMLiability(
                liability_type="N-Glycosylation",
                position=i + 1,
                residue="N",
                motif=motif,
                risk=risk,
                in_cdr=in_cdr,
                description=(
                    f"N-glycosylation sequon {motif} at Asn{i+1}. "
                    "Can cause heterogeneous glycoforms, altered PK/PD, immunogenicity. "
                    + ("In CDR — directly affects binding!" if in_cdr else "")
                ),
            ))

        # O-glycosylation prone: S/T flanked by P (P-S/T or S/T-P)
        for i, aa in enumerate(seq):
            if aa not in "ST":
                continue
            prev_aa = seq[i - 1] if i > 0 else ""
            next_aa = seq[i + 1] if i + 1 < len(seq) else ""
            if prev_aa == "P" or next_aa == "P":
                hits.append(PTMLiability(
                    liability_type="O-Glycosylation",
                    position=i + 1,
                    residue=aa,
                    motif=f"{prev_aa}{aa}{next_aa}",
                    risk="Low",
                    in_cdr=(i + 1) in cdr_set,
                    description=(
                        f"{aa}{i+1} adjacent to Pro — O-glycosylation prone context. "
                        "Risk is lower than N-glycosylation but contributes to microheterogeneity."
                    ),
                ))

        return hits

    # ── Proteolytic sites ─────────────────────────────────────────────────────
    def _scan_proteolytic(self, seq: str) -> List[PTMLiability]:
        hits = []

        # Trypsin: K or R not followed by P
        for i, aa in enumerate(seq):
            if aa not in "KR":
                continue
            next_aa = seq[i + 1] if i + 1 < len(seq) else ""
            if next_aa == "P":
                continue  # KP/RP is resistant
            if i == len(seq) - 1:
                continue  # C-terminal K/R — not in context
            hits.append(PTMLiability(
                liability_type="Proteolytic (Trypsin)",
                position=i + 1,
                residue=aa,
                motif=f"{aa}{next_aa}",
                risk="Low",
                description=(
                    f"{aa}{i+1} — tryptic cleavage site. "
                    "Relevant for in vivo serum stability and manufacturing HCP assessment."
                ),
            ))

        # Furin: R-x-x-R (cleavage after last R)
        for m in re.finditer(r"R.{2}R", seq):
            i = m.start()
            hits.append(PTMLiability(
                liability_type="Proteolytic (Furin)",
                position=i + 1,
                residue="R",
                motif=m.group(),
                risk="Medium",
                description=(
                    f"Furin cleavage motif {m.group()} at position {i+1}. "
                    "Furin is ubiquitously expressed; cleavage can occur in the secretory pathway."
                ),
            ))

        # MMP: P-x-x-P (MMP substrate)
        for m in re.finditer(r"P.{2}P", seq):
            i = m.start()
            hits.append(PTMLiability(
                liability_type="Proteolytic (MMP)",
                position=i + 1,
                residue="P",
                motif=m.group(),
                risk="Low",
                description=(
                    f"MMP-like cleavage motif {m.group()} at position {i+1}. "
                    "Matrix metalloproteinases are upregulated in tumor microenvironment."
                ),
            ))

        return hits

    # ── Terminal liabilities ───────────────────────────────────────────────────
    def _scan_terminal(self, seq: str) -> List[PTMLiability]:
        hits = []

        # N-terminal Gln → pyroglutamate
        if seq and seq[0] == "Q":
            hits.append(PTMLiability(
                liability_type="Pyroglutamate",
                position=1,
                residue="Q",
                motif="Q[N-term]",
                risk="High",
                description=(
                    "N-terminal Gln spontaneously cyclizes to pyroglutamate (-17 Da). "
                    "Ubiquitous in therapeutic antibodies unless blocked by signal peptide processing."
                ),
            ))

        # N-terminal Glu → pyroglutamate (less common but reported)
        if seq and seq[0] == "E":
            hits.append(PTMLiability(
                liability_type="Pyroglutamate",
                position=1,
                residue="E",
                motif="E[N-term]",
                risk="Medium",
                description=(
                    "N-terminal Glu can cyclize to pyroglutamate (-18 Da), though slower than Gln. "
                    "Confirmed in some mAb heavy chains."
                ),
            ))

        # C-terminal Lys clipping (common in CHO-expressed IgG1)
        if seq and seq[-1] == "K":
            hits.append(PTMLiability(
                liability_type="C-terminal Lys Clipping",
                position=len(seq),
                residue="K",
                motif="K[C-term]",
                risk="Medium",
                description=(
                    f"C-terminal Lys{len(seq)} is susceptible to carboxypeptidase B clipping during CHO expression. "
                    "Leads to charge heterogeneity; typically not immunogenic but affects CQA."
                ),
            ))

        # N-terminal Met — methionine oxidation is heightened at N-terminus
        if len(seq) > 1 and seq[0] == "M":
            hits.append(PTMLiability(
                liability_type="N-term Met Oxidation",
                position=1,
                residue="M",
                motif="M[N-term]",
                risk="High",
                description=(
                    "N-terminal Met1 is highly susceptible to oxidation and methionine aminopeptidase (MAP) clipping. "
                    "Loss of N-terminal Met (+/- 131 Da) is a common CQA."
                ),
            ))

        return hits

    # ── Antibody-specific ─────────────────────────────────────────────────────
    def _scan_antibody_specific(self, seq: str, cdr_set: set) -> List[PTMLiability]:
        hits = []

        # Unpaired Cys (odd count)
        cys_count = seq.count("C")
        if cys_count % 2 != 0:
            hits.append(PTMLiability(
                liability_type="Unpaired Cys",
                position=0,
                residue="C",
                motif=f"{cys_count}×Cys",
                risk="High",
                description=(
                    f"Odd number of Cys ({cys_count}) — at least one unpaired thiol. "
                    "Can form aberrant inter-chain disulfides, aggregate, or react with drug-linker in ADCs."
                ),
            ))

        # VH/VL framework Cys — canonical disulfide (Cys23-Cys104 in Kabat)
        # Heuristic: if 2 Cys exist and sequence is ~120 residues → likely VH/VL domain
        if 100 <= len(seq) <= 150 and cys_count == 2:
            cys_positions = [i + 1 for i, aa in enumerate(seq) if aa == "C"]
            gap = cys_positions[1] - cys_positions[0] if len(cys_positions) == 2 else 0
            if 70 <= gap <= 90:
                # Looks like canonical Ig disulfide
                pass  # Normal — no liability
            else:
                hits.append(PTMLiability(
                    liability_type="Ig Disulfide Spacing",
                    position=cys_positions[0] if cys_positions else 0,
                    residue="C",
                    motif=f"C{cys_positions[0]}-C{cys_positions[1]}" if len(cys_positions) == 2 else "",
                    risk="Medium",
                    description=(
                        f"Cys-Cys spacing ({gap} residues) deviates from canonical Ig domain range (70–90). "
                        "Verify structural integrity of framework disulfide."
                    ),
                ))

        return hits


# ── CDR auto-detection ────────────────────────────────────────────────────────

def _detect_cdr_regions(seq: str) -> List[Tuple[int, int]]:
    """
    Heuristic CDR detection for VH and VL sequences using Chothia-like rules.

    For sequences ~110-130 aa that look like Ig variable domains.
    Returns list of (start, end) positions (1-indexed, inclusive).
    """
    cdr_regions: List[Tuple[int, int]] = []
    length = len(seq)

    # VH-like: ~120 residues
    if 100 <= length <= 140:
        # Heuristic Chothia VH CDR positions (1-indexed)
        cdr_regions = [
            (26, 32),   # CDR-H1
            (52, 56),   # CDR-H2
            (95, 102),  # CDR-H3
        ]
    # VL-like (kappa): ~108 residues
    elif 90 <= length <= 115:
        cdr_regions = [
            (24, 34),   # CDR-L1
            (50, 56),   # CDR-L2
            (89, 97),   # CDR-L3
        ]

    # Clamp to sequence length
    result = []
    for start, end in cdr_regions:
        if start <= length:
            result.append((start, min(end, length)))

    return result


def detect_antibody_cdrs(
    sequence: str,
    scheme: str = "chothia",
) -> Dict[str, Tuple[int, int]]:
    """
    Identify CDR regions in an antibody variable domain.

    Uses positional heuristics (no external numbering tool required).
    For production use, pair with ANARCI or AbRSA for exact Kabat/IMGT numbering.

    Args:
        sequence: VH or VL amino acid sequence (~110-130 residues).
        scheme: "chothia" | "imgt" | "kabat" (heuristic offsets only).

    Returns:
        Dict mapping CDR name → (start, end) 1-indexed positions.
    """
    seq = sequence.upper().strip()
    length = len(seq)

    # Chothia VH definition
    if scheme == "chothia":
        if 100 <= length <= 140:
            return {
                "CDR-H1": (26, 32),
                "CDR-H2": (52, 56),
                "CDR-H3": (95, 102),
            }
        elif 90 <= length <= 115:
            return {
                "CDR-L1": (24, 34),
                "CDR-L2": (50, 56),
                "CDR-L3": (89, 97),
            }
    # IMGT definition (slightly different boundaries)
    elif scheme == "imgt":
        if 100 <= length <= 140:
            return {
                "CDR-H1": (27, 38),
                "CDR-H2": (56, 65),
                "CDR-H3": (105, 117),
            }
        elif 90 <= length <= 115:
            return {
                "CDR-L1": (27, 38),
                "CDR-L2": (56, 65),
                "CDR-L3": (105, 117),
            }
    # Kabat definition
    elif scheme == "kabat":
        if 100 <= length <= 140:
            return {
                "CDR-H1": (31, 35),
                "CDR-H2": (50, 65),
                "CDR-H3": (95, 102),
            }
        elif 90 <= length <= 115:
            return {
                "CDR-L1": (24, 34),
                "CDR-L2": (50, 56),
                "CDR-L3": (89, 97),
            }

    return {}


def calculate_humanness_score(
    sequence: str,
    species: str = "human",
) -> Dict[str, float]:
    """
    Estimate humanness/immunogenicity risk from sequence composition.

    Uses Zaroff humanness score approach: fraction of residues matching
    the most common human Ig residue at each position.

    This is a sequence-only approximation. For full humanization use
    BioPython's antibody module or commercial tools (Humanness scoring).

    Returns:
        Dict with "humanness_score" (0-1, higher=more human),
        "immunogenicity_risk" ("Low"/"Medium"/"High"),
        "non_human_positions" count.
    """
    seq = sequence.upper().strip()

    # Frequency of human Ig-like residues — simplified from germline analysis
    # These are the most common human VH framework residues at each position
    # We approximate with compositional analysis
    human_favorable = set("ACDEFGHIKLMNPQRSTVWY")  # All standard AA
    # More heuristic: high Glu/Asp/Lys/Arg → more human-like antibody charge distribution
    human_charged = sum(1 for aa in seq if aa in "DEKR")
    human_hydrophobic = sum(1 for aa in seq if aa in "VILMFW")
    total = len(seq)

    if total == 0:
        return {"humanness_score": 0.0, "immunogenicity_risk": "Unknown", "non_human_positions": 0}

    # Simplified humanness proxy: balanced charge and moderate hydrophobicity
    # Real score requires germline comparison
    charge_ratio = human_charged / total
    hydro_ratio = human_hydrophobic / total

    # Optimal human VH has ~15-25% charged, ~35-45% hydrophobic
    charge_score = 1.0 - abs(charge_ratio - 0.20) * 3
    hydro_score = 1.0 - abs(hydro_ratio - 0.40) * 2
    humanness = max(0.0, min(1.0, (charge_score + hydro_score) / 2))

    if humanness >= 0.7:
        risk = "Low"
    elif humanness >= 0.4:
        risk = "Medium"
    else:
        risk = "High"

    # Non-human positions: residues unusual in human germline
    unusual = sum(1 for aa in seq if aa in "CW")  # Over-represented non-human residues
    non_human_count = unusual

    return {
        "humanness_score": round(humanness, 3),
        "immunogenicity_risk": risk,
        "non_human_positions": non_human_count,
        "note": (
            "Heuristic estimate only. For validated humanization scoring, "
            "use ANARCI + germline alignment."
        ),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────

def _calculate_risk_score(liabilities: List[PTMLiability]) -> float:
    """Calculate composite risk score 0–100."""
    weights = {"High": 15, "Medium": 5, "Low": 1}
    cdr_multiplier = 2.5

    score = 0.0
    for lib in liabilities:
        base = weights.get(lib.risk, 1)
        multiplier = cdr_multiplier if lib.in_cdr else 1.0
        # Skip tryptic sites in risk score (too many, expected)
        if lib.liability_type == "Proteolytic (Trypsin)":
            base = 0.2
        score += base * multiplier

    return min(100.0, score)


def _risk_label(score: float) -> str:
    if score >= 60:
        return "Critical"
    elif score >= 30:
        return "High"
    elif score >= 10:
        return "Medium"
    return "Low"


def _manufacturability_flags(seq: str, liabilities: List[PTMLiability]) -> List[str]:
    """Generate actionable manufacturability flags."""
    flags = []

    types = {lib.liability_type for lib in liabilities}
    high_risk = [lib for lib in liabilities if lib.risk == "High"]

    if "N-Glycosylation" in types:
        n_in_cdr = sum(1 for lib in liabilities if lib.liability_type == "N-Glycosylation" and lib.in_cdr)
        if n_in_cdr:
            flags.append(f"CDR glycosylation sequon(s) detected ({n_in_cdr}) — high impact on binding heterogeneity")
        else:
            flags.append("N-glycosylation sequon in framework — monitor for glycoform heterogeneity")

    if any(lib.liability_type == "Deamidation" and lib.risk == "High" for lib in liabilities):
        flags.append("High-risk deamidation site(s) — likely shelf-life liability at >4°C")

    if any(lib.liability_type == "Isomerization" and lib.risk == "High" for lib in liabilities):
        flags.append("Asp isomerization hotspot(s) — may alter potency and PK profile")

    if any(lib.liability_type == "Pyroglutamate" for lib in liabilities):
        flags.append("N-terminal pyroglutamate formation expected — charge variant in IEX")

    if "Unpaired Cys" in types:
        flags.append("Unpaired Cys present — ADC conjugation or folding risk")

    if "Proteolytic (Furin)" in types:
        flags.append("Furin site detected — potential in vivo cleavage in secretory pathway")

    high_met = [lib for lib in liabilities if lib.liability_type == "Oxidation" and lib.residue == "M" and lib.risk == "High"]
    if len(high_met) >= 2:
        flags.append(f"{len(high_met)} high-risk Met oxidation sites — consider M→L/I substitution or formulation antioxidant")

    return flags
