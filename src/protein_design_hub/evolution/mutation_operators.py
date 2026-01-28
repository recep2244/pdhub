"""Mutation operators for directed evolution.

This module provides various mutation strategies:
- Point mutations (single amino acid changes)
- Recombination (crossover between sequences)
- Insertion/Deletion mutations
- Saturation mutagenesis
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random
from enum import Enum


# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Amino acid groups for conservative mutations
AA_GROUPS: Dict[str, Set[str]] = {
    "hydrophobic": {"A", "V", "I", "L", "M", "F", "W", "P"},
    "aromatic": {"F", "W", "Y", "H"},
    "polar": {"S", "T", "N", "Q", "C", "Y"},
    "positive": {"K", "R", "H"},
    "negative": {"D", "E"},
    "small": {"A", "G", "S", "T", "C"},
    "aliphatic": {"A", "V", "I", "L"},
}

# BLOSUM62-based similar amino acids
SIMILAR_AA: Dict[str, str] = {
    "A": "GVLS",
    "C": "SA",
    "D": "EN",
    "E": "DQK",
    "F": "YWL",
    "G": "AS",
    "H": "NQYR",
    "I": "VLM",
    "K": "RQE",
    "L": "IVMF",
    "M": "ILV",
    "N": "DQHKS",
    "P": "A",
    "Q": "EKNR",
    "R": "KQH",
    "S": "TNGA",
    "T": "SNA",
    "V": "ILA",
    "W": "FY",
    "Y": "FWH",
}


class MutationType(Enum):
    """Types of mutations."""
    POINT = "point"
    CONSERVATIVE = "conservative"
    RADICAL = "radical"
    SATURATION = "saturation"
    INSERTION = "insertion"
    DELETION = "deletion"
    RECOMBINATION = "recombination"


@dataclass
class Mutation:
    """Represents a single mutation."""
    position: int  # 1-indexed
    original: str
    mutant: str
    mutation_type: MutationType = MutationType.POINT

    @property
    def notation(self) -> str:
        """Standard mutation notation (e.g., A123G)."""
        return f"{self.original}{self.position}{self.mutant}"

    def __str__(self) -> str:
        return self.notation


@dataclass
class MutantSequence:
    """A mutant sequence with its mutations."""
    sequence: str
    mutations: List[Mutation] = field(default_factory=list)
    parent_id: Optional[str] = None
    fitness: Optional[float] = None

    @property
    def num_mutations(self) -> int:
        return len(self.mutations)

    @property
    def mutation_string(self) -> str:
        return "/".join(m.notation for m in self.mutations)

    def __str__(self) -> str:
        return f"Mutant({self.mutation_string}, fitness={self.fitness})"


class MutationOperator(ABC):
    """Base class for mutation operators."""

    @abstractmethod
    def mutate(
        self,
        sequence: str,
        **kwargs
    ) -> List[MutantSequence]:
        """
        Generate mutant sequences.

        Args:
            sequence: Parent sequence.
            **kwargs: Operator-specific parameters.

        Returns:
            List of MutantSequence objects.
        """
        pass

    @abstractmethod
    def name(self) -> str:
        """Operator name."""
        pass


class PointMutation(MutationOperator):
    """Single amino acid point mutations."""

    def __init__(
        self,
        num_mutations: int = 1,
        positions: Optional[List[int]] = None,
        exclude_positions: Optional[List[int]] = None,
        allowed_aa: str = AMINO_ACIDS,
        conservative_only: bool = False,
    ):
        """
        Initialize point mutation operator.

        Args:
            num_mutations: Number of mutations per variant.
            positions: Specific positions to mutate (1-indexed). If None, random.
            exclude_positions: Positions to never mutate.
            allowed_aa: Allowed amino acids for mutations.
            conservative_only: Only make conservative substitutions.
        """
        self.num_mutations = num_mutations
        self.positions = positions
        self.exclude_positions = set(exclude_positions or [])
        self.allowed_aa = allowed_aa
        self.conservative_only = conservative_only

    def name(self) -> str:
        return f"PointMutation(n={self.num_mutations})"

    def mutate(
        self,
        sequence: str,
        num_variants: int = 10,
        **kwargs
    ) -> List[MutantSequence]:
        """Generate point mutants."""
        sequence = sequence.upper()
        variants: List[MutantSequence] = []

        # Determine mutable positions
        if self.positions:
            available_positions = [
                p for p in self.positions
                if p not in self.exclude_positions and 1 <= p <= len(sequence)
            ]
        else:
            available_positions = [
                i + 1 for i in range(len(sequence))
                if (i + 1) not in self.exclude_positions
            ]

        if len(available_positions) < self.num_mutations:
            return []

        for _ in range(num_variants):
            # Select positions to mutate
            mut_positions = random.sample(available_positions, self.num_mutations)

            seq_list = list(sequence)
            mutations: List[Mutation] = []

            for pos in mut_positions:
                idx = pos - 1
                original = seq_list[idx]

                # Get allowed substitutions
                if self.conservative_only:
                    allowed = [aa for aa in SIMILAR_AA.get(original, "") if aa in self.allowed_aa]
                else:
                    allowed = [aa for aa in self.allowed_aa if aa != original]

                if not allowed:
                    continue

                new_aa = random.choice(allowed)
                seq_list[idx] = new_aa

                mutations.append(Mutation(
                    position=pos,
                    original=original,
                    mutant=new_aa,
                    mutation_type=MutationType.CONSERVATIVE if self.conservative_only else MutationType.POINT,
                ))

            if mutations:
                variants.append(MutantSequence(
                    sequence="".join(seq_list),
                    mutations=mutations,
                ))

        return variants

    def generate_all_single_mutants(
        self,
        sequence: str,
        positions: Optional[List[int]] = None,
    ) -> List[MutantSequence]:
        """Generate all possible single mutants at specified positions."""
        sequence = sequence.upper()
        variants: List[MutantSequence] = []

        pos_list = positions or list(range(1, len(sequence) + 1))
        pos_list = [p for p in pos_list if p not in self.exclude_positions]

        for pos in pos_list:
            idx = pos - 1
            original = sequence[idx]

            for new_aa in self.allowed_aa:
                if new_aa == original:
                    continue

                if self.conservative_only and new_aa not in SIMILAR_AA.get(original, ""):
                    continue

                seq_list = list(sequence)
                seq_list[idx] = new_aa

                variants.append(MutantSequence(
                    sequence="".join(seq_list),
                    mutations=[Mutation(
                        position=pos,
                        original=original,
                        mutant=new_aa,
                    )],
                ))

        return variants


class SaturationMutagenesis(MutationOperator):
    """Saturation mutagenesis at specific positions."""

    def __init__(
        self,
        positions: List[int],
        allowed_aa: str = AMINO_ACIDS,
    ):
        """
        Initialize saturation mutagenesis.

        Args:
            positions: Positions to saturate (1-indexed).
            allowed_aa: Allowed amino acids.
        """
        self.positions = positions
        self.allowed_aa = allowed_aa

    def name(self) -> str:
        return f"Saturation(pos={self.positions})"

    def mutate(
        self,
        sequence: str,
        **kwargs
    ) -> List[MutantSequence]:
        """Generate all variants with saturation at specified positions."""
        sequence = sequence.upper()
        variants: List[MutantSequence] = []

        for pos in self.positions:
            if pos < 1 or pos > len(sequence):
                continue

            idx = pos - 1
            original = sequence[idx]

            for new_aa in self.allowed_aa:
                if new_aa == original:
                    continue

                seq_list = list(sequence)
                seq_list[idx] = new_aa

                variants.append(MutantSequence(
                    sequence="".join(seq_list),
                    mutations=[Mutation(
                        position=pos,
                        original=original,
                        mutant=new_aa,
                        mutation_type=MutationType.SATURATION,
                    )],
                ))

        return variants


class Recombination(MutationOperator):
    """Recombination (crossover) between parent sequences."""

    def __init__(
        self,
        num_crossover_points: int = 1,
        minimum_segment_length: int = 10,
    ):
        """
        Initialize recombination operator.

        Args:
            num_crossover_points: Number of crossover points.
            minimum_segment_length: Minimum length of each segment.
        """
        self.num_crossover_points = num_crossover_points
        self.min_segment = minimum_segment_length

    def name(self) -> str:
        return f"Recombination(n={self.num_crossover_points})"

    def mutate(
        self,
        sequence: str,
        parents: Optional[List[str]] = None,
        num_variants: int = 10,
        **kwargs
    ) -> List[MutantSequence]:
        """
        Generate recombinant sequences.

        Args:
            sequence: First parent (used if parents not provided).
            parents: List of parent sequences for recombination.
            num_variants: Number of recombinants to generate.
        """
        if parents is None or len(parents) < 2:
            return []

        # Ensure all parents are same length
        length = len(parents[0])
        if not all(len(p) == length for p in parents):
            return []

        variants: List[MutantSequence] = []
        parents = [p.upper() for p in parents]

        for _ in range(num_variants):
            # Select two random parents
            p1, p2 = random.sample(parents, 2)

            # Generate crossover points
            available_points = list(range(self.min_segment, length - self.min_segment))
            if len(available_points) < self.num_crossover_points:
                continue

            crossover_points = sorted(random.sample(available_points, self.num_crossover_points))

            # Build recombinant
            recombinant = []
            current_parent = p1
            prev_point = 0

            for point in crossover_points:
                recombinant.append(current_parent[prev_point:point])
                current_parent = p2 if current_parent == p1 else p1
                prev_point = point

            recombinant.append(current_parent[prev_point:])
            recombinant_seq = "".join(recombinant)

            # Identify mutations relative to first parent
            mutations = []
            for i, (orig, mut) in enumerate(zip(p1, recombinant_seq)):
                if orig != mut:
                    mutations.append(Mutation(
                        position=i + 1,
                        original=orig,
                        mutant=mut,
                        mutation_type=MutationType.RECOMBINATION,
                    ))

            variants.append(MutantSequence(
                sequence=recombinant_seq,
                mutations=mutations,
            ))

        return variants


class InsertionDeletion(MutationOperator):
    """Insertion and deletion mutations."""

    def __init__(
        self,
        max_length: int = 3,
        insertion_probability: float = 0.5,
    ):
        """
        Initialize InDel operator.

        Args:
            max_length: Maximum insertion/deletion length.
            insertion_probability: Probability of insertion vs deletion.
        """
        self.max_length = max_length
        self.ins_prob = insertion_probability

    def name(self) -> str:
        return f"InDel(max={self.max_length})"

    def mutate(
        self,
        sequence: str,
        num_variants: int = 10,
        **kwargs
    ) -> List[MutantSequence]:
        """Generate insertion/deletion variants."""
        sequence = sequence.upper()
        variants: List[MutantSequence] = []

        for _ in range(num_variants):
            seq_list = list(sequence)

            if random.random() < self.ins_prob:
                # Insertion
                position = random.randint(0, len(seq_list))
                length = random.randint(1, self.max_length)
                insert_seq = "".join(random.choices(AMINO_ACIDS, k=length))

                seq_list = seq_list[:position] + list(insert_seq) + seq_list[position:]

                variants.append(MutantSequence(
                    sequence="".join(seq_list),
                    mutations=[Mutation(
                        position=position + 1,
                        original="-",
                        mutant=insert_seq,
                        mutation_type=MutationType.INSERTION,
                    )],
                ))
            else:
                # Deletion
                if len(seq_list) <= self.max_length:
                    continue

                length = random.randint(1, min(self.max_length, len(seq_list) - 1))
                position = random.randint(0, len(seq_list) - length)

                deleted = "".join(seq_list[position:position + length])
                seq_list = seq_list[:position] + seq_list[position + length:]

                variants.append(MutantSequence(
                    sequence="".join(seq_list),
                    mutations=[Mutation(
                        position=position + 1,
                        original=deleted,
                        mutant="-",
                        mutation_type=MutationType.DELETION,
                    )],
                ))

        return variants


class CombinatorialMutation(MutationOperator):
    """Generate combinatorial libraries from defined mutations."""

    def __init__(
        self,
        mutation_sites: Dict[int, List[str]],
    ):
        """
        Initialize combinatorial mutation operator.

        Args:
            mutation_sites: Dictionary mapping positions to allowed amino acids.
                           e.g., {5: ["A", "V", "L"], 10: ["K", "R"]}
        """
        self.mutation_sites = mutation_sites

    def name(self) -> str:
        return f"Combinatorial(sites={len(self.mutation_sites)})"

    def mutate(
        self,
        sequence: str,
        max_variants: int = 1000,
        **kwargs
    ) -> List[MutantSequence]:
        """Generate combinatorial variants."""
        sequence = sequence.upper()
        variants: List[MutantSequence] = []

        # Calculate total library size
        from itertools import product

        positions = sorted(self.mutation_sites.keys())
        aa_options = [self.mutation_sites[p] for p in positions]

        # Generate all combinations (with limit)
        count = 0
        for combo in product(*aa_options):
            if count >= max_variants:
                break

            seq_list = list(sequence)
            mutations = []

            for pos, new_aa in zip(positions, combo):
                idx = pos - 1
                if 0 <= idx < len(seq_list):
                    original = seq_list[idx]
                    if new_aa != original:
                        seq_list[idx] = new_aa
                        mutations.append(Mutation(
                            position=pos,
                            original=original,
                            mutant=new_aa,
                        ))

            if mutations:  # Only add if there are actual mutations
                variants.append(MutantSequence(
                    sequence="".join(seq_list),
                    mutations=mutations,
                ))
                count += 1

        return variants

    def get_library_size(self) -> int:
        """Calculate total library size."""
        size = 1
        for aa_list in self.mutation_sites.values():
            size *= len(aa_list)
        return size
