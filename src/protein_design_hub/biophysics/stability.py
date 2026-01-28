"""Protein stability predictions and disorder propensity.

This module provides:
- ΔΔG mutation effect estimation
- Disorder propensity prediction (IUPred-like)
- Folding free energy estimation
- Thermal stability indicators
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


# ΔΔG scales for amino acid substitutions (simplified FoldX-like)
# Values represent typical ΔΔG contributions in kcal/mol
# Positive = destabilizing, Negative = stabilizing
DDG_SCALES: Dict[str, Dict[str, float]] = {
    # From -> To: ΔΔG
    # Based on average effects from experimental data
    "hydrophobic_to_polar": 2.5,
    "polar_to_hydrophobic": -0.5,  # Can be stabilizing in core
    "charged_to_neutral": 1.5,
    "neutral_to_charged": 0.5,
    "small_to_large": 0.8,
    "large_to_small": 1.2,  # Creates cavity
    "proline_introduction": 2.0,
    "proline_removal": 0.5,
    "glycine_introduction": 1.5,
    "glycine_removal": -0.5,
    "cysteine_change": 1.0,
}

# Amino acid properties for classification
AA_PROPERTIES: Dict[str, Dict[str, bool]] = {
    "A": {"hydrophobic": True, "polar": False, "charged": False, "small": True, "proline": False, "glycine": False},
    "C": {"hydrophobic": False, "polar": True, "charged": False, "small": True, "proline": False, "glycine": False},
    "D": {"hydrophobic": False, "polar": True, "charged": True, "small": True, "proline": False, "glycine": False},
    "E": {"hydrophobic": False, "polar": True, "charged": True, "small": False, "proline": False, "glycine": False},
    "F": {"hydrophobic": True, "polar": False, "charged": False, "small": False, "proline": False, "glycine": False},
    "G": {"hydrophobic": False, "polar": False, "charged": False, "small": True, "proline": False, "glycine": True},
    "H": {"hydrophobic": False, "polar": True, "charged": True, "small": False, "proline": False, "glycine": False},
    "I": {"hydrophobic": True, "polar": False, "charged": False, "small": False, "proline": False, "glycine": False},
    "K": {"hydrophobic": False, "polar": True, "charged": True, "small": False, "proline": False, "glycine": False},
    "L": {"hydrophobic": True, "polar": False, "charged": False, "small": False, "proline": False, "glycine": False},
    "M": {"hydrophobic": True, "polar": False, "charged": False, "small": False, "proline": False, "glycine": False},
    "N": {"hydrophobic": False, "polar": True, "charged": False, "small": True, "proline": False, "glycine": False},
    "P": {"hydrophobic": False, "polar": False, "charged": False, "small": True, "proline": True, "glycine": False},
    "Q": {"hydrophobic": False, "polar": True, "charged": False, "small": False, "proline": False, "glycine": False},
    "R": {"hydrophobic": False, "polar": True, "charged": True, "small": False, "proline": False, "glycine": False},
    "S": {"hydrophobic": False, "polar": True, "charged": False, "small": True, "proline": False, "glycine": False},
    "T": {"hydrophobic": False, "polar": True, "charged": False, "small": True, "proline": False, "glycine": False},
    "V": {"hydrophobic": True, "polar": False, "charged": False, "small": True, "proline": False, "glycine": False},
    "W": {"hydrophobic": True, "polar": False, "charged": False, "small": False, "proline": False, "glycine": False},
    "Y": {"hydrophobic": True, "polar": True, "charged": False, "small": False, "proline": False, "glycine": False},
}

# IUPred-like disorder propensity (based on energy estimation)
DISORDER_PROPENSITY: Dict[str, float] = {
    "A": -0.15,
    "R": 0.10,
    "N": 0.20,
    "D": 0.25,
    "C": -0.25,
    "E": 0.15,
    "Q": 0.10,
    "G": 0.30,
    "H": -0.05,
    "I": -0.45,
    "L": -0.35,
    "K": 0.20,
    "M": -0.20,
    "F": -0.40,
    "P": 0.35,
    "S": 0.15,
    "T": 0.05,
    "W": -0.35,
    "Y": -0.25,
    "V": -0.40,
}


def estimate_ddg_mutation(
    from_aa: str,
    to_aa: str,
    position_type: str = "unknown"
) -> Tuple[float, str]:
    """
    Estimate ΔΔG for a single amino acid mutation.

    This is a simplified estimate without structure information.
    For accurate predictions, use structure-based methods like FoldX or Rosetta.

    Args:
        from_aa: Original amino acid (one-letter code).
        to_aa: Mutant amino acid (one-letter code).
        position_type: "core", "surface", or "unknown".

    Returns:
        Tuple of (estimated ΔΔG in kcal/mol, classification).
    """
    from_aa = from_aa.upper()
    to_aa = to_aa.upper()

    if from_aa == to_aa:
        return 0.0, "No change"

    if from_aa not in AA_PROPERTIES or to_aa not in AA_PROPERTIES:
        return 0.0, "Unknown amino acid"

    from_props = AA_PROPERTIES[from_aa]
    to_props = AA_PROPERTIES[to_aa]

    ddg = 0.0
    effects = []

    # Special cases for Pro and Gly
    if to_props["proline"] and not from_props["proline"]:
        ddg += DDG_SCALES["proline_introduction"]
        effects.append("Pro introduction")
    elif from_props["proline"] and not to_props["proline"]:
        ddg += DDG_SCALES["proline_removal"]
        effects.append("Pro removal")

    if to_props["glycine"] and not from_props["glycine"]:
        ddg += DDG_SCALES["glycine_introduction"]
        effects.append("Gly introduction")
    elif from_props["glycine"] and not to_props["glycine"]:
        ddg += DDG_SCALES["glycine_removal"]
        effects.append("Gly removal")

    # Cysteine changes (potential disulfide disruption)
    if from_aa == "C" or to_aa == "C":
        ddg += DDG_SCALES["cysteine_change"]
        effects.append("Cys change")

    # Hydrophobicity changes
    if from_props["hydrophobic"] and to_props["polar"]:
        if position_type == "core":
            ddg += DDG_SCALES["hydrophobic_to_polar"] * 1.5
        else:
            ddg += DDG_SCALES["hydrophobic_to_polar"]
        effects.append("Hydrophobic→Polar")
    elif from_props["polar"] and to_props["hydrophobic"]:
        if position_type == "surface":
            ddg += abs(DDG_SCALES["polar_to_hydrophobic"]) * 1.5  # Bad on surface
        else:
            ddg += DDG_SCALES["polar_to_hydrophobic"]
        effects.append("Polar→Hydrophobic")

    # Charge changes
    if from_props["charged"] and not to_props["charged"]:
        ddg += DDG_SCALES["charged_to_neutral"]
        effects.append("Charged→Neutral")
    elif not from_props["charged"] and to_props["charged"]:
        ddg += DDG_SCALES["neutral_to_charged"]
        effects.append("Neutral→Charged")

    # Size changes
    if from_props["small"] and not to_props["small"]:
        ddg += DDG_SCALES["small_to_large"]
        effects.append("Small→Large")
    elif not from_props["small"] and to_props["small"]:
        ddg += DDG_SCALES["large_to_small"]
        effects.append("Large→Small (cavity)")

    # Classify overall effect
    if ddg > 2.0:
        classification = "Highly destabilizing"
    elif ddg > 1.0:
        classification = "Destabilizing"
    elif ddg > 0.5:
        classification = "Mildly destabilizing"
    elif ddg > -0.5:
        classification = "Neutral"
    elif ddg > -1.0:
        classification = "Mildly stabilizing"
    else:
        classification = "Stabilizing"

    effect_str = f"{classification} ({', '.join(effects) if effects else 'similar properties'})"

    return round(ddg, 2), effect_str


def calculate_disorder_propensity(
    sequence: str, window_size: int = 21
) -> Tuple[float, List[float], List[Tuple[int, int]]]:
    """
    Calculate disorder propensity (IUPred-like prediction).

    Args:
        sequence: Amino acid sequence.
        window_size: Window size for smoothing.

    Returns:
        Tuple of:
        - Overall disorder fraction
        - Per-residue disorder scores (0-1)
        - Disordered regions as (start, end) tuples
    """
    sequence = sequence.upper()
    if len(sequence) < window_size:
        return 0.0, [], []

    half_window = window_size // 2
    per_residue_scores: List[float] = []

    for i in range(len(sequence)):
        start = max(0, i - half_window)
        end = min(len(sequence), i + half_window + 1)
        window = sequence[start:end]

        # Calculate average disorder propensity
        score = sum(DISORDER_PROPENSITY.get(aa, 0.0) for aa in window) / len(window)

        # Add composition-based features
        # High charge = more disorder
        charged = sum(1 for aa in window if aa in "DEKR") / len(window)
        # Low hydrophobicity = more disorder
        hydrophobic = sum(1 for aa in window if aa in "AVILMFYW") / len(window)
        # Pro/Gly = more disorder
        disorder_promoters = sum(1 for aa in window if aa in "PG") / len(window)

        # Combined score
        combined = score + charged * 0.3 - hydrophobic * 0.3 + disorder_promoters * 0.2
        per_residue_scores.append(combined)

    # Normalize to 0-1 range using sigmoid
    normalized_scores = [1 / (1 + math.exp(-s * 5)) for s in per_residue_scores]

    # Identify disordered regions (score > 0.5)
    threshold = 0.5
    regions: List[Tuple[int, int]] = []
    in_region = False
    region_start = 0

    for i, score in enumerate(normalized_scores):
        if score > threshold and not in_region:
            in_region = True
            region_start = i
        elif score <= threshold and in_region:
            in_region = False
            if i - region_start >= 5:  # Minimum length
                regions.append((region_start, i - 1))

    if in_region and len(sequence) - region_start >= 5:
        regions.append((region_start, len(sequence) - 1))

    # Calculate disorder fraction
    disordered_residues = sum(1 for s in normalized_scores if s > threshold)
    disorder_fraction = disordered_residues / len(sequence)

    return disorder_fraction, normalized_scores, regions


def estimate_folding_free_energy(sequence: str) -> Dict[str, float]:
    """
    Estimate folding free energy indicators.

    This provides rough estimates based on sequence composition.
    Not a substitute for physics-based calculations.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Dictionary with stability indicators.
    """
    sequence = sequence.upper()
    length = len(sequence)

    if length == 0:
        return {}

    # Hydrophobic burial potential
    hydrophobic_aa = "AVILMFW"
    hydrophobic_fraction = sum(1 for aa in sequence if aa in hydrophobic_aa) / length

    # Charge balance
    positive = sum(1 for aa in sequence if aa in "KR")
    negative = sum(1 for aa in sequence if aa in "DE")
    charge_balance = abs(positive - negative) / length

    # Proline/glycine content (can destabilize helices)
    helix_breakers = sum(1 for aa in sequence if aa in "PG") / length

    # Aromatic content (can contribute to core stability)
    aromatic = sum(1 for aa in sequence if aa in "FWY") / length

    # Cysteine pairs potential (disulfide bonds)
    cysteine_count = sequence.count("C")
    potential_disulfides = cysteine_count // 2

    # Rough stability score
    stability_score = (
        hydrophobic_fraction * 2.0  # Hydrophobic core
        - charge_balance * 0.5      # Charge imbalance penalty
        - helix_breakers * 1.0      # Helix breakers penalty
        + aromatic * 0.5            # Aromatic contribution
        + potential_disulfides * 0.3  # Disulfide bonus
    )

    return {
        "stability_score": stability_score,
        "hydrophobic_fraction": hydrophobic_fraction,
        "charge_balance": charge_balance,
        "helix_breaker_fraction": helix_breakers,
        "aromatic_fraction": aromatic,
        "potential_disulfides": potential_disulfides,
        "predicted_stability": "High" if stability_score > 0.6 else "Medium" if stability_score > 0.3 else "Low",
    }


def identify_stabilizing_mutations(
    sequence: str,
    target_positions: Optional[List[int]] = None
) -> List[Dict]:
    """
    Suggest potentially stabilizing mutations.

    Args:
        sequence: Amino acid sequence.
        target_positions: Specific positions to analyze (1-indexed). If None, analyzes all.

    Returns:
        List of suggested mutations with predicted effects.
    """
    sequence = sequence.upper()
    suggestions: List[Dict] = []

    positions = target_positions or list(range(1, len(sequence) + 1))

    for pos in positions:
        if pos < 1 or pos > len(sequence):
            continue

        idx = pos - 1
        current_aa = sequence[idx]

        # Skip if already optimal
        if current_aa in "IVLF":  # Already hydrophobic/stable
            continue

        best_mutations = []

        # Test all possible mutations
        for new_aa in "ACDEFGHIKLMNPQRSTVWY":
            if new_aa == current_aa:
                continue

            ddg, effect = estimate_ddg_mutation(current_aa, new_aa)

            if ddg < -0.3:  # Potentially stabilizing
                best_mutations.append({
                    "from": current_aa,
                    "to": new_aa,
                    "position": pos,
                    "mutation": f"{current_aa}{pos}{new_aa}",
                    "predicted_ddg": ddg,
                    "effect": effect,
                })

        # Sort by ΔΔG and keep top suggestions
        best_mutations.sort(key=lambda x: x["predicted_ddg"])
        suggestions.extend(best_mutations[:2])  # Top 2 per position

    # Sort all suggestions by ΔΔG
    suggestions.sort(key=lambda x: x["predicted_ddg"])

    return suggestions[:20]  # Return top 20 overall


@dataclass
class StabilityPredictor:
    """Comprehensive stability prediction for proteins."""

    sequence: str

    def __post_init__(self):
        self.sequence = self.sequence.upper()

    def predict(self) -> Dict:
        """Run all stability predictions."""
        disorder_frac, disorder_scores, disorder_regions = calculate_disorder_propensity(
            self.sequence
        )
        folding_energy = estimate_folding_free_energy(self.sequence)
        stabilizing = identify_stabilizing_mutations(self.sequence)

        return {
            "disorder_fraction": disorder_frac,
            "disorder_per_residue": disorder_scores,
            "disordered_regions": [
                {"start": s + 1, "end": e + 1, "sequence": self.sequence[s:e+1]}
                for s, e in disorder_regions
            ],
            "folding_indicators": folding_energy,
            "stabilizing_mutations": stabilizing[:10],  # Top 10
            "overall_assessment": self._assess_overall(disorder_frac, folding_energy),
        }

    def _assess_overall(self, disorder_frac: float, folding: Dict) -> str:
        """Generate overall stability assessment."""
        stability = folding.get("predicted_stability", "Unknown")
        disorder_pct = disorder_frac * 100

        if stability == "High" and disorder_frac < 0.2:
            return f"STABLE - Good folding indicators, {disorder_pct:.0f}% predicted disorder"
        elif stability == "Medium" or disorder_frac < 0.4:
            return f"MODERATE - Average stability, {disorder_pct:.0f}% predicted disorder"
        else:
            return f"UNSTABLE - Poor folding indicators, {disorder_pct:.0f}% predicted disorder"

    def predict_mutation_effects(
        self, mutations: List[str]
    ) -> List[Dict]:
        """
        Predict effects of specific mutations.

        Args:
            mutations: List of mutations in format "A123G" (from_aa + position + to_aa).

        Returns:
            List of mutation effect predictions.
        """
        results = []

        for mut in mutations:
            if len(mut) < 3:
                continue

            from_aa = mut[0].upper()
            to_aa = mut[-1].upper()
            try:
                position = int(mut[1:-1])
            except ValueError:
                continue

            # Verify the mutation matches the sequence
            if position < 1 or position > len(self.sequence):
                results.append({
                    "mutation": mut,
                    "error": f"Position {position} out of range",
                })
                continue

            actual_aa = self.sequence[position - 1]
            if actual_aa != from_aa:
                results.append({
                    "mutation": mut,
                    "error": f"Position {position} is {actual_aa}, not {from_aa}",
                })
                continue

            ddg, effect = estimate_ddg_mutation(from_aa, to_aa)
            results.append({
                "mutation": mut,
                "from_aa": from_aa,
                "to_aa": to_aa,
                "position": position,
                "predicted_ddg": ddg,
                "effect": effect,
            })

        return results
