"""Solubility and aggregation propensity predictions.

This module provides:
- Solubility score prediction
- Aggregation propensity (APR - Aggregation Prone Regions)
- CamSol-like intrinsic solubility
- Sequence-based solubility features
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

from protein_design_hub.biophysics.properties import (
    calculate_gravy,
    calculate_charge,
    calculate_pi,
    HYDROPATHY,
)


# Solubility propensity scale (based on experimental data)
# Higher values = more soluble
SOLUBILITY_SCALE: Dict[str, float] = {
    "A": 0.1,
    "R": 1.5,
    "N": 0.8,
    "D": 1.2,
    "C": -0.3,
    "E": 1.3,
    "Q": 0.6,
    "G": 0.2,
    "H": 0.4,
    "I": -1.5,
    "L": -1.2,
    "K": 1.4,
    "M": -0.5,
    "F": -1.8,
    "P": 0.3,
    "S": 0.5,
    "T": 0.2,
    "W": -1.5,
    "Y": -0.9,
    "V": -1.0,
}

# Aggregation propensity scale (higher = more aggregation prone)
AGGREGATION_SCALE: Dict[str, float] = {
    "A": 0.2,
    "R": -1.5,
    "N": -0.5,
    "D": -1.0,
    "C": 0.5,
    "E": -1.2,
    "Q": -0.3,
    "G": 0.1,
    "H": -0.2,
    "I": 1.5,
    "L": 1.3,
    "K": -1.3,
    "M": 0.8,
    "F": 1.8,
    "P": -0.8,
    "S": -0.2,
    "T": 0.0,
    "W": 1.5,
    "Y": 1.0,
    "V": 1.2,
}

# Beta-sheet propensity (Chou-Fasman)
BETA_PROPENSITY: Dict[str, float] = {
    "A": 0.83,
    "R": 0.93,
    "N": 0.89,
    "D": 0.54,
    "C": 1.19,
    "E": 0.37,
    "Q": 1.10,
    "G": 0.75,
    "H": 0.87,
    "I": 1.60,
    "L": 1.30,
    "K": 0.74,
    "M": 1.05,
    "F": 1.38,
    "P": 0.55,
    "S": 0.75,
    "T": 1.19,
    "W": 1.37,
    "Y": 1.47,
    "V": 1.70,
}


def calculate_solubility_score(sequence: str) -> float:
    """
    Calculate overall solubility score.

    Based on a combination of:
    - Amino acid solubility propensities
    - Charge distribution
    - Hydrophobicity balance

    Args:
        sequence: Amino acid sequence.

    Returns:
        Solubility score (higher = more soluble). Range roughly -2 to +2.
    """
    sequence = sequence.upper()
    if not sequence:
        return 0.0

    # Component 1: Average solubility propensity
    sol_score = sum(SOLUBILITY_SCALE.get(aa, 0.0) for aa in sequence) / len(sequence)

    # Component 2: Charge contribution (charged residues help solubility)
    charge = abs(calculate_charge(sequence, 7.0))
    charge_contrib = min(charge / len(sequence) * 10, 0.5)  # Cap contribution

    # Component 3: Hydrophobicity penalty
    gravy = calculate_gravy(sequence)
    hydro_penalty = -gravy * 0.3  # Hydrophobic sequences get penalized

    # Component 4: Gatekeeper residues (R, K, D, E, P) help prevent aggregation
    gatekeepers = sum(1 for aa in sequence if aa in "RKDEP")
    gatekeeper_bonus = min(gatekeepers / len(sequence) * 2, 0.3)

    total_score = sol_score + charge_contrib + hydro_penalty + gatekeeper_bonus

    return total_score


def predict_aggregation_propensity(
    sequence: str, window_size: int = 7
) -> Tuple[float, List[float], List[Tuple[int, int]]]:
    """
    Predict aggregation propensity and identify aggregation-prone regions (APRs).

    Args:
        sequence: Amino acid sequence.
        window_size: Window size for scanning.

    Returns:
        Tuple of:
        - Overall aggregation score (0-1, higher = more prone)
        - Per-residue scores
        - List of APR regions as (start, end) tuples
    """
    sequence = sequence.upper()
    if len(sequence) < window_size:
        return 0.0, [], []

    per_residue_scores: List[float] = []

    # Calculate per-residue aggregation propensity using sliding window
    half_window = window_size // 2

    for i in range(len(sequence)):
        start = max(0, i - half_window)
        end = min(len(sequence), i + half_window + 1)
        window = sequence[start:end]

        # Calculate window score
        agg_score = sum(AGGREGATION_SCALE.get(aa, 0.0) for aa in window) / len(window)
        beta_score = sum(BETA_PROPENSITY.get(aa, 1.0) for aa in window) / len(window)

        # Combine aggregation propensity with beta-sheet propensity
        combined = (agg_score + (beta_score - 1.0)) / 2
        per_residue_scores.append(combined)

    # Normalize scores to 0-1 range
    min_score = min(per_residue_scores)
    max_score = max(per_residue_scores)
    score_range = max_score - min_score if max_score != min_score else 1.0

    normalized_scores = [
        (s - min_score) / score_range for s in per_residue_scores
    ]

    # Identify APRs (regions with score > 0.6)
    threshold = 0.6
    aprs: List[Tuple[int, int]] = []
    in_apr = False
    apr_start = 0

    for i, score in enumerate(normalized_scores):
        if score > threshold and not in_apr:
            in_apr = True
            apr_start = i
        elif score <= threshold and in_apr:
            in_apr = False
            if i - apr_start >= 5:  # Minimum APR length
                aprs.append((apr_start, i - 1))

    if in_apr and len(sequence) - apr_start >= 5:
        aprs.append((apr_start, len(sequence) - 1))

    # Overall score is average of top 10% worst regions
    sorted_scores = sorted(normalized_scores, reverse=True)
    top_10_pct = max(1, len(sorted_scores) // 10)
    overall_score = sum(sorted_scores[:top_10_pct]) / top_10_pct

    return overall_score, normalized_scores, aprs


def calculate_camsol_like_score(sequence: str) -> Dict[str, float]:
    """
    Calculate CamSol-like intrinsic solubility features.

    Based on the CamSol algorithm principles.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Dictionary with solubility features.
    """
    sequence = sequence.upper()
    if not sequence:
        return {}

    length = len(sequence)

    # Intrinsic solubility from amino acid composition
    intrinsic_sol = calculate_solubility_score(sequence)

    # Charge distribution
    total_charge = abs(calculate_charge(sequence, 7.0))
    charge_density = total_charge / length

    # Hydrophobic patches (runs of hydrophobic residues)
    hydrophobic = "AVILMFYW"
    max_hydro_patch = 0
    current_patch = 0
    for aa in sequence:
        if aa in hydrophobic:
            current_patch += 1
            max_hydro_patch = max(max_hydro_patch, current_patch)
        else:
            current_patch = 0

    # Proline content (proline breaks aggregation)
    proline_content = sequence.count("P") / length

    # Aromatic content (can contribute to aggregation)
    aromatic_content = sum(1 for aa in sequence if aa in "FWY") / length

    # Calculate overall CamSol-like score
    camsol_score = (
        intrinsic_sol
        + charge_density * 2.0
        - max_hydro_patch * 0.1
        + proline_content * 3.0
        - aromatic_content * 0.5
    )

    return {
        "camsol_score": camsol_score,
        "intrinsic_solubility": intrinsic_sol,
        "charge_density": charge_density,
        "max_hydrophobic_patch": max_hydro_patch,
        "proline_content": proline_content,
        "aromatic_content": aromatic_content,
        "solubility_class": "High" if camsol_score > 0.5 else "Medium" if camsol_score > -0.5 else "Low",
    }


def identify_problematic_regions(
    sequence: str, window_size: int = 11
) -> List[Dict]:
    """
    Identify potentially problematic regions for expression/solubility.

    Args:
        sequence: Amino acid sequence.
        window_size: Window size for analysis.

    Returns:
        List of problematic regions with details.
    """
    sequence = sequence.upper()
    problems: List[Dict] = []

    half_window = window_size // 2

    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        center = i + half_window

        issues = []

        # Check for hydrophobic stretch
        gravy = calculate_gravy(window)
        if gravy > 1.5:
            issues.append(f"Highly hydrophobic (GRAVY={gravy:.2f})")

        # Check for aggregation-prone beta strand
        beta_score = sum(BETA_PROPENSITY.get(aa, 1.0) for aa in window) / len(window)
        if beta_score > 1.3:
            issues.append(f"Beta-aggregation prone (score={beta_score:.2f})")

        # Check for low charge density
        local_charge = abs(calculate_charge(window, 7.0))
        if local_charge < 0.5 and gravy > 0.5:
            issues.append("Low charge + hydrophobic")

        # Check for consecutive hydrophobic residues
        max_consecutive = 0
        current = 0
        for aa in window:
            if aa in "VILMFW":
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        if max_consecutive >= 6:
            issues.append(f"Long hydrophobic run ({max_consecutive} residues)")

        if issues:
            problems.append({
                "start": i,
                "end": i + window_size - 1,
                "center": center,
                "sequence": window,
                "issues": issues,
            })

    # Merge overlapping regions
    merged: List[Dict] = []
    for region in problems:
        if merged and region["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = max(merged[-1]["end"], region["end"])
            merged[-1]["issues"] = list(set(merged[-1]["issues"] + region["issues"]))
        else:
            merged.append(region)

    return merged


@dataclass
class SolubilityPredictor:
    """Comprehensive solubility prediction for proteins."""

    sequence: str

    def __post_init__(self):
        self.sequence = self.sequence.upper()

    def predict(self) -> Dict:
        """Run all solubility predictions."""
        sol_score = calculate_solubility_score(self.sequence)
        agg_score, agg_per_residue, aprs = predict_aggregation_propensity(self.sequence)
        camsol = calculate_camsol_like_score(self.sequence)
        problems = identify_problematic_regions(self.sequence)

        return {
            "solubility_score": sol_score,
            "aggregation_propensity": agg_score,
            "aggregation_per_residue": agg_per_residue,
            "aggregation_prone_regions": [
                {"start": s, "end": e, "sequence": self.sequence[s:e+1]}
                for s, e in aprs
            ],
            "camsol_features": camsol,
            "problematic_regions": problems,
            "overall_assessment": self._assess_overall(sol_score, agg_score),
            "recommendations": self._get_recommendations(sol_score, agg_score, aprs),
        }

    def _assess_overall(self, sol_score: float, agg_score: float) -> str:
        """Generate overall assessment."""
        if sol_score > 0.5 and agg_score < 0.4:
            return "GOOD - High predicted solubility, low aggregation risk"
        elif sol_score > 0 and agg_score < 0.6:
            return "MODERATE - Acceptable solubility, some aggregation risk"
        elif sol_score > -0.5 or agg_score < 0.7:
            return "POOR - Low solubility or high aggregation risk"
        else:
            return "VERY POOR - High risk of expression/solubility issues"

    def _get_recommendations(
        self, sol_score: float, agg_score: float, aprs: List[Tuple[int, int]]
    ) -> List[str]:
        """Generate recommendations for improving solubility."""
        recs = []

        if sol_score < 0:
            recs.append("Consider adding charged residues (K, R, E, D) to improve solubility")

        if agg_score > 0.5:
            recs.append("Introduce gatekeeper residues (P, K, R, E, D) near aggregation-prone regions")

        if aprs:
            recs.append(f"Identified {len(aprs)} aggregation-prone regions - consider mutations")
            for i, (start, end) in enumerate(aprs[:3]):  # Show top 3
                recs.append(f"  APR {i+1}: positions {start+1}-{end+1}")

        if calculate_gravy(self.sequence) > 0.5:
            recs.append("High overall hydrophobicity - consider surface mutations to polar residues")

        if not recs:
            recs.append("Sequence appears suitable for recombinant expression")

        return recs
