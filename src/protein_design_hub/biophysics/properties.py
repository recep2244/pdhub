"""Core biophysical property calculations for proteins.

This module provides calculations for:
- Molecular weight
- Isoelectric point (pI)
- Net charge at given pH
- Extinction coefficient
- GRAVY (hydropathicity)
- Instability index
- Aliphatic index
- Aromaticity
- Amino acid composition
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# Amino acid molecular weights (Da) - monoisotopic masses
AA_MW: Dict[str, float] = {
    "A": 89.09,
    "R": 174.20,
    "N": 132.12,
    "D": 133.10,
    "C": 121.15,
    "E": 147.13,
    "Q": 146.15,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "L": 131.17,
    "K": 146.19,
    "M": 149.21,
    "F": 165.19,
    "P": 115.13,
    "S": 105.09,
    "T": 119.12,
    "W": 204.23,
    "Y": 181.19,
    "V": 117.15,
    # Non-standard
    "U": 168.06,  # Selenocysteine
    "O": 255.31,  # Pyrrolysine
    "B": 132.61,  # Asx (N or D average)
    "Z": 146.64,  # Glx (Q or E average)
    "X": 110.0,   # Unknown (average)
}

# Water molecular weight (lost during peptide bond formation)
WATER_MW = 18.015

# pKa values for amino acids (N-terminus, C-terminus, and side chains)
PKA_VALUES: Dict[str, Dict[str, float]] = {
    # Amino acid: {group: pKa}
    "N_TERM": {"pKa": 9.69},  # Average N-terminus
    "C_TERM": {"pKa": 2.34},  # Average C-terminus
    "D": {"pKa": 3.86},  # Aspartate side chain
    "E": {"pKa": 4.25},  # Glutamate side chain
    "C": {"pKa": 8.33},  # Cysteine side chain
    "Y": {"pKa": 10.07},  # Tyrosine side chain
    "H": {"pKa": 6.00},  # Histidine side chain
    "K": {"pKa": 10.53},  # Lysine side chain
    "R": {"pKa": 12.48},  # Arginine side chain
}

# Kyte-Doolittle hydropathy scale
HYDROPATHY: Dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Instability index weights (DIWV values)
INSTABILITY_WEIGHTS: Dict[str, Dict[str, float]] = {
    "A": {"A": 1.0, "C": 44.94, "E": 1.0, "D": -7.49, "G": 1.0, "F": 1.0, "I": 1.0, "H": -7.49, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 1.0},
    "C": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 20.26, "G": 1.0, "F": 1.0, "I": 1.0, "H": 33.6, "K": 1.0, "M": 33.6, "L": 20.26, "N": 1.0, "Q": -6.54, "P": 20.26, "S": 1.0, "R": 1.0, "T": 33.6, "W": 24.68, "V": -6.54, "Y": 1.0},
    "E": {"A": 1.0, "C": 44.94, "E": 33.6, "D": 20.26, "G": 1.0, "F": 1.0, "I": 20.26, "H": -6.54, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": 1.0, "T": 1.0, "W": -14.03, "V": 1.0, "Y": 1.0},
    "D": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 1.0, "S": 20.26, "R": -6.54, "T": -14.03, "W": 1.0, "V": 1.0, "Y": 1.0},
    "G": {"A": -7.49, "C": 1.0, "E": -6.54, "D": 1.0, "G": 13.34, "F": 1.0, "I": -7.49, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": -7.49, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -7.49, "W": 13.34, "V": 1.0, "Y": -7.49},
    "F": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 13.34, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -14.03, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 33.6},
    "I": {"A": 1.0, "C": 1.0, "E": 44.94, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 13.34, "K": -7.49, "M": 1.0, "L": 20.26, "N": 1.0, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": 1.0, "W": 1.0, "V": -7.49, "Y": 1.0},
    "H": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": -9.37, "I": 44.94, "H": 1.0, "K": 24.68, "M": 1.0, "L": 1.0, "N": 24.68, "Q": 1.0, "P": -1.88, "S": 1.0, "R": 1.0, "T": -6.54, "W": -1.88, "V": 1.0, "Y": 44.94},
    "K": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": -7.49, "H": 1.0, "K": 1.0, "M": 33.6, "L": -7.49, "N": 1.0, "Q": 24.68, "P": -6.54, "S": 1.0, "R": 33.6, "T": 1.0, "W": 1.0, "V": -7.49, "Y": 1.0},
    "M": {"A": 13.34, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 58.28, "K": 1.0, "M": -1.88, "L": 1.0, "N": 1.0, "Q": -6.54, "P": 44.94, "S": 44.94, "R": -6.54, "T": -1.88, "W": 1.0, "V": 1.0, "Y": 24.68},
    "L": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": -7.49, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 33.6, "P": 20.26, "S": 1.0, "R": 20.26, "T": 1.0, "W": 24.68, "V": 1.0, "Y": 1.0},
    "N": {"A": 1.0, "C": -1.88, "E": 1.0, "D": 1.0, "G": -14.03, "F": -14.03, "I": 44.94, "H": 1.0, "K": 24.68, "M": 1.0, "L": 1.0, "N": 1.0, "Q": -6.54, "P": -1.88, "S": 1.0, "R": 1.0, "T": -7.49, "W": -9.37, "V": 1.0, "Y": 1.0},
    "Q": {"A": 1.0, "C": -6.54, "E": 20.26, "D": 20.26, "G": 1.0, "F": -6.54, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 1.0, "T": 1.0, "W": 1.0, "V": -6.54, "Y": -6.54},
    "P": {"A": 20.26, "C": -6.54, "E": 18.38, "D": -6.54, "G": 1.0, "F": 20.26, "I": 1.0, "H": 1.0, "K": 1.0, "M": -6.54, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 20.26, "S": 20.26, "R": -6.54, "T": 1.0, "W": -1.88, "V": 20.26, "Y": 1.0},
    "S": {"A": 1.0, "C": 33.6, "E": 20.26, "D": 1.0, "G": 1.0, "F": 1.0, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 20.26, "P": 44.94, "S": 20.26, "R": 20.26, "T": 1.0, "W": 1.0, "V": 1.0, "Y": 1.0},
    "R": {"A": 1.0, "C": 1.0, "E": 1.0, "D": 1.0, "G": -7.49, "F": 1.0, "I": 1.0, "H": 20.26, "K": 1.0, "M": 1.0, "L": 1.0, "N": 13.34, "Q": 20.26, "P": 20.26, "S": 44.94, "R": 58.28, "T": 1.0, "W": 58.28, "V": 1.0, "Y": -6.54},
    "T": {"A": 1.0, "C": 1.0, "E": 20.26, "D": 1.0, "G": -7.49, "F": 13.34, "I": 1.0, "H": 1.0, "K": 1.0, "M": 1.0, "L": 1.0, "N": -14.03, "Q": -6.54, "P": 1.0, "S": 1.0, "R": 1.0, "T": 1.0, "W": -14.03, "V": 1.0, "Y": 1.0},
    "W": {"A": -14.03, "C": 1.0, "E": 1.0, "D": 1.0, "G": -9.37, "F": 1.0, "I": 1.0, "H": 24.68, "K": 1.0, "M": 24.68, "L": 13.34, "N": 13.34, "Q": 1.0, "P": 1.0, "S": 1.0, "R": 1.0, "T": -14.03, "W": 1.0, "V": -7.49, "Y": 1.0},
    "V": {"A": 1.0, "C": 1.0, "E": 1.0, "D": -14.03, "G": -7.49, "F": 1.0, "I": 1.0, "H": 1.0, "K": -1.88, "M": 1.0, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 20.26, "S": 1.0, "R": 1.0, "T": -7.49, "W": 1.0, "V": 1.0, "Y": -6.54},
    "Y": {"A": 24.68, "C": 1.0, "E": -6.54, "D": 24.68, "G": -7.49, "F": 1.0, "I": 1.0, "H": 13.34, "K": 1.0, "M": 44.94, "L": 1.0, "N": 1.0, "Q": 1.0, "P": 13.34, "S": 1.0, "R": -15.91, "T": -7.49, "W": -9.37, "V": 1.0, "Y": 13.34},
}

# Aliphatic residues
ALIPHATIC_RESIDUES = {"A", "V", "I", "L"}

# Aromatic residues
AROMATIC_RESIDUES = {"F", "W", "Y"}


def calculate_mw(sequence: str, include_water: bool = True) -> float:
    """
    Calculate molecular weight of a protein sequence.

    Args:
        sequence: Amino acid sequence (one-letter codes).
        include_water: Include water in calculation (for full protein).

    Returns:
        Molecular weight in Daltons (Da).
    """
    sequence = sequence.upper()
    mw = sum(AA_MW.get(aa, AA_MW["X"]) for aa in sequence)

    # Subtract water lost during peptide bond formation
    if include_water and len(sequence) > 1:
        mw -= (len(sequence) - 1) * WATER_MW

    return mw


def calculate_charge(sequence: str, pH: float = 7.0) -> float:
    """
    Calculate net charge of protein at given pH.

    Uses Henderson-Hasselbalch equation.

    Args:
        sequence: Amino acid sequence.
        pH: pH value for calculation.

    Returns:
        Net charge at given pH.
    """
    sequence = sequence.upper()

    # Count charged residues
    counts = {
        "D": sequence.count("D"),
        "E": sequence.count("E"),
        "C": sequence.count("C"),
        "Y": sequence.count("Y"),
        "H": sequence.count("H"),
        "K": sequence.count("K"),
        "R": sequence.count("R"),
    }

    charge = 0.0

    # N-terminus (positive at low pH)
    pKa_n = PKA_VALUES["N_TERM"]["pKa"]
    charge += 1.0 / (1.0 + 10 ** (pH - pKa_n))

    # C-terminus (negative at high pH)
    pKa_c = PKA_VALUES["C_TERM"]["pKa"]
    charge -= 1.0 / (1.0 + 10 ** (pKa_c - pH))

    # Acidic side chains (D, E) - negative at high pH
    for aa in ["D", "E"]:
        if counts[aa] > 0:
            pKa = PKA_VALUES[aa]["pKa"]
            charge -= counts[aa] / (1.0 + 10 ** (pKa - pH))

    # C, Y - negative at high pH
    for aa in ["C", "Y"]:
        if counts[aa] > 0:
            pKa = PKA_VALUES[aa]["pKa"]
            charge -= counts[aa] / (1.0 + 10 ** (pKa - pH))

    # Basic side chains (K, R, H) - positive at low pH
    for aa in ["K", "R", "H"]:
        if counts[aa] > 0:
            pKa = PKA_VALUES[aa]["pKa"]
            charge += counts[aa] / (1.0 + 10 ** (pH - pKa))

    return charge


def calculate_pi(sequence: str, precision: float = 0.01) -> float:
    """
    Calculate isoelectric point (pI) of protein.

    Uses bisection method to find pH where net charge is zero.

    Args:
        sequence: Amino acid sequence.
        precision: Precision of pI calculation.

    Returns:
        Isoelectric point (pH where net charge is 0).
    """
    pH_low = 0.0
    pH_high = 14.0

    while (pH_high - pH_low) > precision:
        pH_mid = (pH_low + pH_high) / 2.0
        charge = calculate_charge(sequence, pH_mid)

        if charge > 0:
            pH_low = pH_mid
        else:
            pH_high = pH_mid

    return (pH_low + pH_high) / 2.0


def calculate_extinction_coefficient(
    sequence: str, reduced: bool = True
) -> Tuple[float, float]:
    """
    Calculate molar extinction coefficient at 280 nm.

    Based on Pace et al. (1995) method.

    Args:
        sequence: Amino acid sequence.
        reduced: If True, assumes all Cys are reduced (no disulfide bonds).

    Returns:
        Tuple of (extinction_coefficient, Abs_0.1%_at_280nm).
    """
    sequence = sequence.upper()

    # Extinction coefficients at 280 nm (M^-1 cm^-1)
    ext_trp = 5500  # Tryptophan
    ext_tyr = 1490  # Tyrosine
    ext_cys = 125   # Cystine (disulfide bond)

    n_trp = sequence.count("W")
    n_tyr = sequence.count("Y")
    n_cys = sequence.count("C")

    # Calculate extinction coefficient
    ext_coeff = n_trp * ext_trp + n_tyr * ext_tyr

    # Add cystine contribution (assumes all Cys form disulfide bonds if not reduced)
    if not reduced:
        n_cystine = n_cys // 2
        ext_coeff += n_cystine * ext_cys

    # Calculate Abs 0.1% (1 mg/ml)
    mw = calculate_mw(sequence)
    abs_01_percent = ext_coeff / mw if mw > 0 else 0

    return ext_coeff, abs_01_percent


def calculate_gravy(sequence: str) -> float:
    """
    Calculate GRAVY (Grand Average of Hydropathicity).

    Uses Kyte-Doolittle scale.

    Args:
        sequence: Amino acid sequence.

    Returns:
        GRAVY score (positive = hydrophobic, negative = hydrophilic).
    """
    sequence = sequence.upper()
    if not sequence:
        return 0.0

    total = sum(HYDROPATHY.get(aa, 0.0) for aa in sequence)
    return total / len(sequence)


def calculate_instability_index(sequence: str) -> float:
    """
    Calculate instability index (Guruprasad et al., 1990).

    Proteins with instability index > 40 are considered unstable.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Instability index value.
    """
    sequence = sequence.upper()
    if len(sequence) < 2:
        return 0.0

    score = 0.0
    for i in range(len(sequence) - 1):
        aa1 = sequence[i]
        aa2 = sequence[i + 1]

        if aa1 in INSTABILITY_WEIGHTS and aa2 in INSTABILITY_WEIGHTS.get(aa1, {}):
            score += INSTABILITY_WEIGHTS[aa1][aa2]

    return (10.0 / len(sequence)) * score


def calculate_aliphatic_index(sequence: str) -> float:
    """
    Calculate aliphatic index (Ikai, 1980).

    Measure of thermostability; higher values indicate more thermostable proteins.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Aliphatic index value.
    """
    sequence = sequence.upper()
    if not sequence:
        return 0.0

    length = len(sequence)
    n_ala = sequence.count("A")
    n_val = sequence.count("V")
    n_ile = sequence.count("I")
    n_leu = sequence.count("L")

    # Aliphatic index = X(Ala) + a*X(Val) + b*[X(Ile) + X(Leu)]
    # where a = 2.9, b = 3.9
    aliphatic_index = (
        (n_ala / length * 100)
        + 2.9 * (n_val / length * 100)
        + 3.9 * ((n_ile + n_leu) / length * 100)
    )

    return aliphatic_index


def calculate_aromaticity(sequence: str) -> float:
    """
    Calculate aromaticity (frequency of aromatic residues).

    Args:
        sequence: Amino acid sequence.

    Returns:
        Aromaticity (fraction of F, W, Y).
    """
    sequence = sequence.upper()
    if not sequence:
        return 0.0

    n_aromatic = sum(1 for aa in sequence if aa in AROMATIC_RESIDUES)
    return n_aromatic / len(sequence)


def calculate_amino_acid_composition(sequence: str) -> Dict[str, float]:
    """
    Calculate amino acid composition.

    Args:
        sequence: Amino acid sequence.

    Returns:
        Dictionary of amino acid frequencies.
    """
    sequence = sequence.upper()
    if not sequence:
        return {}

    length = len(sequence)
    composition = {}

    for aa in "ACDEFGHIKLMNPQRSTVWY":
        count = sequence.count(aa)
        composition[aa] = count / length

    return composition


def calculate_secondary_structure_fraction(sequence: str) -> Dict[str, float]:
    """
    Estimate secondary structure propensity (Chou-Fasman).

    Args:
        sequence: Amino acid sequence.

    Returns:
        Dictionary with helix, sheet, turn propensities.
    """
    # Chou-Fasman propensities
    helix_formers = {"A", "E", "L", "M"}
    helix_breakers = {"P", "G"}
    sheet_formers = {"V", "I", "Y", "F", "W", "T"}
    turn_formers = {"G", "P", "S", "D", "N"}

    sequence = sequence.upper()
    if not sequence:
        return {"helix": 0.0, "sheet": 0.0, "turn": 0.0}

    length = len(sequence)

    helix_count = sum(1 for aa in sequence if aa in helix_formers)
    sheet_count = sum(1 for aa in sequence if aa in sheet_formers)
    turn_count = sum(1 for aa in sequence if aa in turn_formers)

    return {
        "helix_propensity": helix_count / length,
        "sheet_propensity": sheet_count / length,
        "turn_propensity": turn_count / length,
    }


@dataclass
class ProteinProperties:
    """Container for all biophysical properties of a protein."""

    sequence: str
    length: int = field(init=False)
    molecular_weight: float = field(init=False)
    isoelectric_point: float = field(init=False)
    charge_at_ph7: float = field(init=False)
    extinction_coefficient: float = field(init=False)
    abs_01_percent: float = field(init=False)
    gravy: float = field(init=False)
    instability_index: float = field(init=False)
    aliphatic_index: float = field(init=False)
    aromaticity: float = field(init=False)
    amino_acid_composition: Dict[str, float] = field(init=False)
    secondary_structure: Dict[str, float] = field(init=False)

    def __post_init__(self):
        """Calculate all properties after initialization."""
        self.sequence = self.sequence.upper()
        self.length = len(self.sequence)
        self.molecular_weight = calculate_mw(self.sequence)
        self.isoelectric_point = calculate_pi(self.sequence)
        self.charge_at_ph7 = calculate_charge(self.sequence, 7.0)

        ext_coeff, abs_01 = calculate_extinction_coefficient(self.sequence)
        self.extinction_coefficient = ext_coeff
        self.abs_01_percent = abs_01

        self.gravy = calculate_gravy(self.sequence)
        self.instability_index = calculate_instability_index(self.sequence)
        self.aliphatic_index = calculate_aliphatic_index(self.sequence)
        self.aromaticity = calculate_aromaticity(self.sequence)
        self.amino_acid_composition = calculate_amino_acid_composition(self.sequence)
        self.secondary_structure = calculate_secondary_structure_fraction(self.sequence)

    @property
    def is_stable(self) -> bool:
        """Check if protein is predicted to be stable (instability index < 40)."""
        return self.instability_index < 40

    @property
    def is_hydrophobic(self) -> bool:
        """Check if protein is hydrophobic (GRAVY > 0)."""
        return self.gravy > 0

    def charge_at_ph(self, pH: float) -> float:
        """Calculate charge at specific pH."""
        return calculate_charge(self.sequence, pH)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "length": self.length,
            "molecular_weight_da": self.molecular_weight,
            "molecular_weight_kda": self.molecular_weight / 1000,
            "isoelectric_point": self.isoelectric_point,
            "charge_at_ph7": self.charge_at_ph7,
            "extinction_coefficient_m1_cm1": self.extinction_coefficient,
            "abs_01_percent": self.abs_01_percent,
            "gravy": self.gravy,
            "instability_index": self.instability_index,
            "is_stable": self.is_stable,
            "aliphatic_index": self.aliphatic_index,
            "aromaticity": self.aromaticity,
            "is_hydrophobic": self.is_hydrophobic,
            "amino_acid_composition": self.amino_acid_composition,
            "secondary_structure_propensity": self.secondary_structure,
        }

    def summary(self) -> str:
        """Generate text summary."""
        lines = [
            "=" * 50,
            "PROTEIN PROPERTIES",
            "=" * 50,
            f"Length: {self.length} residues",
            f"Molecular Weight: {self.molecular_weight:.2f} Da ({self.molecular_weight/1000:.2f} kDa)",
            f"Isoelectric Point (pI): {self.isoelectric_point:.2f}",
            f"Net Charge at pH 7.0: {self.charge_at_ph7:+.2f}",
            f"Extinction Coefficient (280 nm): {self.extinction_coefficient:.0f} M^-1 cm^-1",
            f"Abs 0.1% (1 mg/ml): {self.abs_01_percent:.3f}",
            "",
            f"GRAVY (Hydropathicity): {self.gravy:.3f} ({'Hydrophobic' if self.is_hydrophobic else 'Hydrophilic'})",
            f"Instability Index: {self.instability_index:.2f} ({'Unstable' if not self.is_stable else 'Stable'})",
            f"Aliphatic Index: {self.aliphatic_index:.2f}",
            f"Aromaticity: {self.aromaticity:.3f} ({self.aromaticity*100:.1f}%)",
            "",
            "Secondary Structure Propensity:",
            f"  Helix: {self.secondary_structure['helix_propensity']*100:.1f}%",
            f"  Sheet: {self.secondary_structure['sheet_propensity']*100:.1f}%",
            f"  Turn:  {self.secondary_structure['turn_propensity']*100:.1f}%",
        ]
        return "\n".join(lines)
