"""Analysis module for protein sequence and structure analysis.

This module provides:
- Mutation scanning and saturation mutagenesis
- Sequence analysis and metrics
- Variant recommendation
"""

from protein_design_hub.analysis.mutation_scanner import (
    MutationResult,
    MutationScanner,
    SaturationMutagenesisResult,
    MultiMutationResult,
    MultiMutationVariant,
    create_mutation_heatmap,
    AMINO_ACIDS,
    AA_PROPERTIES,
)

__all__ = [
    "MutationResult",
    "MutationScanner",
    "SaturationMutagenesisResult",
    "MultiMutationResult",
    "MultiMutationVariant",
    "create_mutation_heatmap",
    "AMINO_ACIDS",
    "AA_PROPERTIES",
]
