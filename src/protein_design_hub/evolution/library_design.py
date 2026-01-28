"""Combinatorial library design for directed evolution.

This module provides:
- Mutation library generation
- Combinatorial library design
- Degenerate codon optimization
- Library complexity estimation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from itertools import product
import math


# Degenerate codon table
DEGENERATE_CODONS: Dict[str, str] = {
    # Code: Amino acids encoded
    "NNN": "ACDEFGHIKLMNPQRSTVWY*",
    "NNK": "ACDEFGHIKLMNPQRSTVWY",  # No stop codons
    "NNS": "ACDEFGHIKLMNPQRSTVWY",  # No stop codons
    "NDT": "CDFGHILNRSVY",  # 12 AA, no stop
    "DBK": "ACDEGHKLMNPQRSTVW",
    "NRT": "CDHNRSTY",
    "VHG": "ADEGKLMNQRSTV",
    "VWC": "ADGHILMNPRSTV",
}

# Amino acid to best degenerate codon
AA_TO_CODON: Dict[str, str] = {
    "A": "GCN",
    "C": "TGY",
    "D": "GAY",
    "E": "GAR",
    "F": "TTY",
    "G": "GGN",
    "H": "CAY",
    "I": "ATH",
    "K": "AAR",
    "L": "YTN",
    "M": "ATG",
    "N": "AAY",
    "P": "CCN",
    "Q": "CAR",
    "R": "MGN",
    "S": "WSN",
    "T": "ACN",
    "V": "GTN",
    "W": "TGG",
    "Y": "TAY",
}


@dataclass
class MutationSite:
    """A site for mutation in a library."""
    position: int  # 1-indexed
    wild_type: str
    allowed_aa: List[str]
    degenerate_codon: Optional[str] = None

    @property
    def diversity(self) -> int:
        """Number of amino acids at this site."""
        return len(self.allowed_aa)

    def __str__(self) -> str:
        return f"Site({self.position}: {self.wild_type}->{','.join(self.allowed_aa)})"


@dataclass
class MutationLibrary:
    """A mutation library with defined sites."""
    name: str
    parent_sequence: str
    sites: List[MutationSite] = field(default_factory=list)

    @property
    def theoretical_size(self) -> int:
        """Calculate theoretical library size."""
        size = 1
        for site in self.sites:
            size *= site.diversity
        return size

    @property
    def positions(self) -> List[int]:
        """Get all mutation positions."""
        return [site.position for site in self.sites]

    def add_site(
        self,
        position: int,
        allowed_aa: List[str],
        degenerate_codon: Optional[str] = None,
    ) -> None:
        """Add a mutation site."""
        idx = position - 1
        if 0 <= idx < len(self.parent_sequence):
            wt = self.parent_sequence[idx]
            self.sites.append(MutationSite(
                position=position,
                wild_type=wt,
                allowed_aa=allowed_aa,
                degenerate_codon=degenerate_codon,
            ))

    def add_saturation_site(self, position: int) -> None:
        """Add site for full NNK saturation."""
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        self.add_site(position, aa_list, "NNK")

    def generate_variants(self, max_variants: int = 10000) -> List[str]:
        """Generate all variant sequences up to limit."""
        variants = []
        parent = list(self.parent_sequence.upper())

        # Get all position/aa combinations
        positions = [site.position - 1 for site in self.sites]
        aa_options = [site.allowed_aa for site in self.sites]

        count = 0
        for combo in product(*aa_options):
            if count >= max_variants:
                break

            variant = parent.copy()
            for idx, aa in zip(positions, combo):
                variant[idx] = aa

            variants.append("".join(variant))
            count += 1

        return variants

    def get_codon_scheme(self) -> Dict[int, str]:
        """Get degenerate codon scheme for each site."""
        return {
            site.position: site.degenerate_codon or "NNK"
            for site in self.sites
        }

    def estimate_screening_requirements(
        self,
        coverage: float = 0.95,
    ) -> Dict[str, float]:
        """
        Estimate screening requirements for library coverage.

        Uses the formula: n = -ln(1-p) * L
        where n = number to screen, p = coverage, L = library size

        Args:
            coverage: Desired coverage fraction (0-1).

        Returns:
            Dictionary with screening estimates.
        """
        lib_size = self.theoretical_size

        # Number to screen for given coverage
        n_screen = -math.log(1 - coverage) * lib_size

        # For 3x coverage
        n_3x = 3 * lib_size

        return {
            "library_size": lib_size,
            "coverage_target": coverage,
            "variants_to_screen": math.ceil(n_screen),
            "variants_for_3x_coverage": n_3x,
            "log10_library_size": math.log10(lib_size) if lib_size > 0 else 0,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "parent_sequence": self.parent_sequence,
            "sites": [
                {
                    "position": s.position,
                    "wild_type": s.wild_type,
                    "allowed_aa": s.allowed_aa,
                    "degenerate_codon": s.degenerate_codon,
                }
                for s in self.sites
            ],
            "theoretical_size": self.theoretical_size,
        }


class CombinatorialLibrary:
    """Design combinatorial libraries with defined mutations."""

    def __init__(
        self,
        parent_sequence: str,
        name: str = "library",
    ):
        """
        Initialize library designer.

        Args:
            parent_sequence: Wild-type sequence.
            name: Library name.
        """
        self.parent_sequence = parent_sequence.upper()
        self.name = name
        self.mutation_sites: Dict[int, Set[str]] = {}

    def add_mutation(self, position: int, amino_acids: List[str]) -> None:
        """
        Add allowed mutations at a position.

        Args:
            position: Position (1-indexed).
            amino_acids: List of allowed amino acids.
        """
        if position not in self.mutation_sites:
            self.mutation_sites[position] = set()
        self.mutation_sites[position].update(aa.upper() for aa in amino_acids)

    def add_mutations_from_hits(
        self,
        hits: List[Dict],
        min_fitness: float = 0.5,
    ) -> None:
        """
        Add mutations from experimental hits.

        Args:
            hits: List of dicts with 'mutations' key containing mutation strings.
            min_fitness: Minimum fitness to include.
        """
        for hit in hits:
            if hit.get("fitness", 0) < min_fitness:
                continue

            mutations = hit.get("mutations", [])
            for mut in mutations:
                # Parse mutation string (e.g., "A123G")
                if len(mut) >= 3:
                    try:
                        pos = int(mut[1:-1])
                        new_aa = mut[-1].upper()
                        self.add_mutation(pos, [new_aa])
                    except ValueError:
                        continue

    def set_saturation(self, positions: List[int]) -> None:
        """Set positions for full NNK saturation."""
        all_aa = list("ACDEFGHIKLMNPQRSTVWY")
        for pos in positions:
            self.add_mutation(pos, all_aa)

    def get_library(self) -> MutationLibrary:
        """Get the designed library."""
        library = MutationLibrary(
            name=self.name,
            parent_sequence=self.parent_sequence,
        )

        for pos in sorted(self.mutation_sites.keys()):
            aa_list = sorted(self.mutation_sites[pos])
            library.add_site(pos, aa_list)

        return library

    @property
    def theoretical_size(self) -> int:
        """Calculate theoretical library size."""
        size = 1
        for aa_set in self.mutation_sites.values():
            size *= len(aa_set)
        return size


class LibraryDesigner:
    """Intelligent library design with optimization."""

    def __init__(self, parent_sequence: str):
        """
        Initialize library designer.

        Args:
            parent_sequence: Wild-type sequence.
        """
        self.parent_sequence = parent_sequence.upper()

    def design_focused_library(
        self,
        beneficial_mutations: List[str],
        target_size: int = 1000,
    ) -> MutationLibrary:
        """
        Design a focused library from beneficial mutations.

        Args:
            beneficial_mutations: List of mutation strings (e.g., ["A10G", "V25L"]).
            target_size: Target library size.

        Returns:
            Designed MutationLibrary.
        """
        # Parse mutations
        parsed: Dict[int, Set[str]] = {}

        for mut in beneficial_mutations:
            if len(mut) >= 3:
                try:
                    pos = int(mut[1:-1])
                    new_aa = mut[-1].upper()
                    wt = mut[0].upper()

                    if pos not in parsed:
                        parsed[pos] = {wt}  # Include wild-type
                    parsed[pos].add(new_aa)
                except ValueError:
                    continue

        # Create library
        library = MutationLibrary(
            name="focused",
            parent_sequence=self.parent_sequence,
        )

        for pos in sorted(parsed.keys()):
            aa_list = sorted(parsed[pos])
            library.add_site(pos, aa_list)

        # Check if library is too large
        if library.theoretical_size > target_size:
            # Reduce by removing least diverse sites
            sites_by_diversity = sorted(library.sites, key=lambda s: s.diversity)
            while library.theoretical_size > target_size and len(library.sites) > 1:
                library.sites = [s for s in library.sites if s != sites_by_diversity[0]]
                sites_by_diversity = sorted(library.sites, key=lambda s: s.diversity)

        return library

    def design_alanine_scanning(
        self,
        region: Optional[Tuple[int, int]] = None,
    ) -> MutationLibrary:
        """
        Design alanine scanning library.

        Args:
            region: Optional (start, end) region to scan (1-indexed).

        Returns:
            MutationLibrary for alanine scanning.
        """
        library = MutationLibrary(
            name="alanine_scan",
            parent_sequence=self.parent_sequence,
        )

        start = region[0] if region else 1
        end = region[1] if region else len(self.parent_sequence)

        for pos in range(start, end + 1):
            idx = pos - 1
            if idx < 0 or idx >= len(self.parent_sequence):
                continue

            wt = self.parent_sequence[idx]
            if wt != "A":  # Don't mutate existing alanines
                library.add_site(pos, [wt, "A"])

        return library

    def design_from_conservation(
        self,
        conservation_scores: List[float],
        variability_threshold: float = 0.5,
        top_n_positions: int = 10,
    ) -> MutationLibrary:
        """
        Design library based on conservation scores.

        Targets positions with low conservation (high variability).

        Args:
            conservation_scores: Per-residue conservation (0=variable, 1=conserved).
            variability_threshold: Threshold for considering position variable.
            top_n_positions: Maximum positions to include.

        Returns:
            MutationLibrary targeting variable positions.
        """
        library = MutationLibrary(
            name="conservation_based",
            parent_sequence=self.parent_sequence,
        )

        # Find variable positions
        variable_positions = [
            (i + 1, score)
            for i, score in enumerate(conservation_scores)
            if score < variability_threshold
        ]

        # Sort by variability (lowest conservation first)
        variable_positions.sort(key=lambda x: x[1])

        # Add top positions
        for pos, _ in variable_positions[:top_n_positions]:
            library.add_saturation_site(pos)

        return library

    def estimate_dna_synthesis(
        self,
        library: MutationLibrary,
    ) -> Dict:
        """
        Estimate DNA synthesis requirements.

        Args:
            library: The mutation library.

        Returns:
            Synthesis requirements and costs.
        """
        # Calculate number of oligos needed
        n_sites = len(library.sites)
        oligo_length = 60  # Typical oligo length

        # Estimate primer numbers
        n_primers = n_sites * 2  # Forward and reverse per site

        # For Gibson assembly
        n_fragments = n_sites + 1

        return {
            "mutation_sites": n_sites,
            "estimated_primers": n_primers,
            "fragments_for_gibson": n_fragments,
            "estimated_oligo_length": oligo_length,
            "library_theoretical_size": library.theoretical_size,
            "suggested_method": self._suggest_synthesis_method(library),
        }

    def _suggest_synthesis_method(self, library: MutationLibrary) -> str:
        """Suggest synthesis method based on library design."""
        n_sites = len(library.sites)
        lib_size = library.theoretical_size

        if n_sites == 1:
            return "Site-directed mutagenesis or degenerate primers"
        elif n_sites <= 3 and lib_size < 1000:
            return "Overlap extension PCR with degenerate primers"
        elif n_sites <= 6 and lib_size < 10000:
            return "Gibson assembly with degenerate oligos"
        elif lib_size < 100000:
            return "Gene synthesis with designed variants"
        else:
            return "Error-prone PCR or DNA shuffling recommended"

    def optimize_codon_scheme(
        self,
        library: MutationLibrary,
    ) -> Dict[int, str]:
        """
        Optimize degenerate codon scheme to minimize stop codons and bias.

        Args:
            library: The mutation library.

        Returns:
            Optimized codon scheme for each position.
        """
        optimized = {}

        for site in library.sites:
            # Find best degenerate codon that covers all desired AA
            best_codon = "NNK"  # Default
            best_coverage = 0
            best_stops = 1

            for codon, aa_encoded in DEGENERATE_CODONS.items():
                coverage = sum(1 for aa in site.allowed_aa if aa in aa_encoded)
                stops = 1 if "*" in aa_encoded else 0

                # Prefer codons with better coverage and fewer stops
                if coverage > best_coverage or (
                    coverage == best_coverage and stops < best_stops
                ):
                    best_codon = codon
                    best_coverage = coverage
                    best_stops = stops

            optimized[site.position] = best_codon

        return optimized
