"""MSA and evolutionary analysis module.

This module provides tools for analyzing multiple sequence alignments (MSAs)
and extracting evolutionary information for protein engineering applications.

Features:
- Conservation scoring (Shannon entropy, Kullback-Leibler divergence)
- Coevolution analysis (Mutual Information, Direct Coupling Analysis)
- Position-specific scoring matrices (PSSM)
- Ancestral sequence reconstruction
"""

from protein_design_hub.msa.conservation import (
    ConservationCalculator,
    calculate_conservation,
    calculate_shannon_entropy,
    calculate_js_divergence,
)
from protein_design_hub.msa.coevolution import (
    CoevolutionAnalyzer,
    calculate_mutual_information,
    calculate_apc_corrected_mi,
)
from protein_design_hub.msa.pssm import (
    PSSMCalculator,
    calculate_pssm,
    score_sequence_with_pssm,
)
from protein_design_hub.msa.ancestral import (
    AncestralReconstructor,
    reconstruct_ancestral_sequence,
)

__all__ = [
    "ConservationCalculator",
    "calculate_conservation",
    "calculate_shannon_entropy",
    "calculate_js_divergence",
    "CoevolutionAnalyzer",
    "calculate_mutual_information",
    "calculate_apc_corrected_mi",
    "PSSMCalculator",
    "calculate_pssm",
    "score_sequence_with_pssm",
    "AncestralReconstructor",
    "reconstruct_ancestral_sequence",
]
