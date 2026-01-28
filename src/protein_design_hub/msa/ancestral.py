"""Ancestral sequence reconstruction from MSA.

Provides methods for inferring ancestral protein sequences using:
- Maximum parsimony
- Maximum likelihood (simplified)
- Consensus-based reconstruction
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Simplified amino acid substitution matrix (based on BLOSUM62 similarities)
# Groups amino acids by physicochemical properties
AA_GROUPS = {
    "hydrophobic": {"A", "V", "I", "L", "M", "F", "W", "P"},
    "aromatic": {"F", "Y", "W", "H"},
    "polar": {"S", "T", "N", "Q", "C"},
    "positive": {"K", "R", "H"},
    "negative": {"D", "E"},
    "small": {"G", "A", "S"},
    "aliphatic": {"A", "V", "I", "L"},
}


@dataclass
class AncestralResult:
    """Container for ancestral reconstruction results."""

    sequence: str
    confidence: List[float]  # Per-position confidence
    alternative_residues: List[Dict[str, float]]  # Position -> {AA: probability}
    method: str
    num_sequences: int
    alignment_length: int


@dataclass
class PhylogeneticNode:
    """Node in a phylogenetic tree."""

    name: str
    sequence: Optional[str] = None
    children: List["PhylogeneticNode"] = field(default_factory=list)
    branch_length: float = 0.1
    ancestral_state: Optional[str] = None
    state_probabilities: Optional[List[Dict[str, float]]] = None


class AncestralReconstructor:
    """Reconstruct ancestral sequences from MSA."""

    def __init__(
        self,
        method: str = "likelihood",
        pseudocount: float = 0.01,
        prior_weight: float = 0.1,
    ):
        """
        Initialize ancestral reconstructor.

        Args:
            method: Reconstruction method ("likelihood", "parsimony", "consensus").
            pseudocount: Pseudocount for probability calculations.
            prior_weight: Weight for prior (background) probabilities.
        """
        self.method = method
        self.pseudocount = pseudocount
        self.prior_weight = prior_weight

        # Background frequencies
        self.background = {aa: 1.0 / len(AMINO_ACIDS) for aa in AMINO_ACIDS}

    def reconstruct(
        self,
        alignment: List[str],
        tree: Optional[PhylogeneticNode] = None,
        names: Optional[List[str]] = None,
    ) -> AncestralResult:
        """
        Reconstruct ancestral sequence from alignment.

        Args:
            alignment: List of aligned sequences.
            tree: Optional phylogenetic tree (uses UPGMA if not provided).
            names: Optional sequence names.

        Returns:
            AncestralResult with reconstructed sequence.
        """
        if not alignment:
            raise ValueError("Empty alignment provided")

        n_seqs = len(alignment)
        length = len(alignment[0])

        if names is None:
            names = [f"seq_{i}" for i in range(n_seqs)]

        # Build tree if not provided
        if tree is None:
            tree = self._build_upgma_tree(alignment, names)

        # Reconstruct based on method
        if self.method == "likelihood":
            sequence, confidence, alternatives = self._reconstruct_ml(
                alignment, tree, length
            )
        elif self.method == "parsimony":
            sequence, confidence, alternatives = self._reconstruct_parsimony(
                alignment, tree, length
            )
        else:  # consensus
            sequence, confidence, alternatives = self._reconstruct_consensus(
                alignment, length
            )

        return AncestralResult(
            sequence=sequence,
            confidence=confidence,
            alternative_residues=alternatives,
            method=self.method,
            num_sequences=n_seqs,
            alignment_length=length,
        )

    def _reconstruct_ml(
        self,
        alignment: List[str],
        tree: PhylogeneticNode,
        length: int,
    ) -> Tuple[str, List[float], List[Dict[str, float]]]:
        """Maximum likelihood reconstruction."""
        sequence = []
        confidence = []
        alternatives = []

        for pos in range(length):
            # Calculate likelihood for each amino acid at root
            aa_likelihoods = self._calculate_position_likelihoods(
                alignment, tree, pos
            )

            # Normalize to probabilities
            total = sum(aa_likelihoods.values())
            if total > 0:
                probs = {aa: l / total for aa, l in aa_likelihoods.items()}
            else:
                probs = {aa: 1.0 / len(AMINO_ACIDS) for aa in AMINO_ACIDS}

            # Find most likely
            best_aa = max(probs.keys(), key=lambda x: probs[x])
            sequence.append(best_aa)
            confidence.append(probs[best_aa])
            alternatives.append(probs)

        return "".join(sequence), confidence, alternatives

    def _calculate_position_likelihoods(
        self,
        alignment: List[str],
        tree: PhylogeneticNode,
        pos: int,
    ) -> Dict[str, float]:
        """Calculate likelihood of each amino acid at a position."""
        # Simplified ML: weighted by sequence similarity
        likelihoods = {aa: self.pseudocount for aa in AMINO_ACIDS}

        for seq in alignment:
            aa = seq[pos].upper()
            if aa in AMINO_ACIDS:
                likelihoods[aa] += 1.0
            elif aa != "-":
                # Distribute probability for ambiguous
                for candidate in AMINO_ACIDS:
                    likelihoods[candidate] += 1.0 / len(AMINO_ACIDS)

        # Apply prior
        for aa in AMINO_ACIDS:
            likelihoods[aa] = (
                (1 - self.prior_weight) * likelihoods[aa]
                + self.prior_weight * self.background[aa] * len(alignment)
            )

        return likelihoods

    def _reconstruct_parsimony(
        self,
        alignment: List[str],
        tree: PhylogeneticNode,
        length: int,
    ) -> Tuple[str, List[float], List[Dict[str, float]]]:
        """Maximum parsimony reconstruction (Fitch algorithm simplified)."""
        sequence = []
        confidence = []
        alternatives = []

        for pos in range(length):
            # Count amino acids
            aa_counts = {}
            for seq in alignment:
                aa = seq[pos].upper()
                if aa in AMINO_ACIDS:
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1

            if not aa_counts:
                sequence.append("X")
                confidence.append(0.0)
                alternatives.append({})
                continue

            # Most parsimonious = most common
            total = sum(aa_counts.values())
            probs = {aa: c / total for aa, c in aa_counts.items()}

            best_aa = max(aa_counts.keys(), key=lambda x: aa_counts[x])
            sequence.append(best_aa)
            confidence.append(probs[best_aa])

            # Add zeros for missing AAs
            full_probs = {aa: probs.get(aa, 0.0) for aa in AMINO_ACIDS}
            alternatives.append(full_probs)

        return "".join(sequence), confidence, alternatives

    def _reconstruct_consensus(
        self,
        alignment: List[str],
        length: int,
    ) -> Tuple[str, List[float], List[Dict[str, float]]]:
        """Simple consensus-based reconstruction."""
        sequence = []
        confidence = []
        alternatives = []

        for pos in range(length):
            aa_counts = {}
            total = 0

            for seq in alignment:
                aa = seq[pos].upper()
                if aa in AMINO_ACIDS:
                    aa_counts[aa] = aa_counts.get(aa, 0) + 1
                    total += 1

            if total == 0:
                sequence.append("X")
                confidence.append(0.0)
                alternatives.append({aa: 0.0 for aa in AMINO_ACIDS})
                continue

            probs = {aa: (aa_counts.get(aa, 0) + self.pseudocount) / (total + self.pseudocount * len(AMINO_ACIDS))
                     for aa in AMINO_ACIDS}

            best_aa = max(aa_counts.keys(), key=lambda x: aa_counts[x])
            sequence.append(best_aa)
            confidence.append(aa_counts[best_aa] / total)
            alternatives.append(probs)

        return "".join(sequence), confidence, alternatives

    def _build_upgma_tree(
        self,
        alignment: List[str],
        names: List[str],
    ) -> PhylogeneticNode:
        """
        Build a simple UPGMA tree from alignment.

        Simplified implementation for ancestral reconstruction.
        """
        n = len(alignment)

        # Create leaf nodes
        nodes = [
            PhylogeneticNode(name=names[i], sequence=alignment[i])
            for i in range(n)
        ]

        if n == 1:
            return nodes[0]

        # Calculate distance matrix
        distances = {}
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._calculate_distance(alignment[i], alignment[j])
                distances[(i, j)] = dist
                distances[(j, i)] = dist

        # UPGMA clustering
        active = list(range(n))
        cluster_sizes = {i: 1 for i in range(n)}

        while len(active) > 1:
            # Find minimum distance pair
            min_dist = float("inf")
            min_pair = (0, 1)

            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    idx_i, idx_j = active[i], active[j]
                    d = distances.get((idx_i, idx_j), float("inf"))
                    if d < min_dist:
                        min_dist = d
                        min_pair = (i, j)

            i, j = min_pair
            idx_i, idx_j = active[i], active[j]

            # Create new node
            new_idx = len(nodes)
            new_node = PhylogeneticNode(
                name=f"node_{new_idx}",
                children=[nodes[idx_i], nodes[idx_j]],
                branch_length=min_dist / 2,
            )
            nodes.append(new_node)

            # Update distances (UPGMA formula)
            size_i = cluster_sizes[idx_i]
            size_j = cluster_sizes[idx_j]

            for k in active:
                if k not in (idx_i, idx_j):
                    d_ik = distances.get((idx_i, k), distances.get((k, idx_i), 0))
                    d_jk = distances.get((idx_j, k), distances.get((k, idx_j), 0))
                    new_dist = (d_ik * size_i + d_jk * size_j) / (size_i + size_j)
                    distances[(new_idx, k)] = new_dist
                    distances[(k, new_idx)] = new_dist

            cluster_sizes[new_idx] = size_i + size_j

            # Update active list
            active = [x for x in active if x not in (idx_i, idx_j)]
            active.append(new_idx)

        return nodes[-1]

    def _calculate_distance(self, seq1: str, seq2: str) -> float:
        """Calculate sequence distance (1 - identity)."""
        if len(seq1) != len(seq2):
            return 1.0

        matches = 0
        total = 0

        for a, b in zip(seq1.upper(), seq2.upper()):
            if a != "-" and b != "-":
                total += 1
                if a == b:
                    matches += 1

        if total == 0:
            return 1.0

        identity = matches / total
        # Jukes-Cantor correction (simplified)
        if identity < 0.05:
            return 3.0  # Max distance
        return -0.75 * math.log(1 - 4 * (1 - identity) / 3)

    def get_ambiguous_positions(
        self,
        result: AncestralResult,
        confidence_threshold: float = 0.5,
    ) -> List[int]:
        """
        Get positions with low reconstruction confidence.

        Args:
            result: AncestralResult from reconstruction.
            confidence_threshold: Minimum confidence threshold.

        Returns:
            List of ambiguous position indices.
        """
        return [
            i for i, conf in enumerate(result.confidence)
            if conf < confidence_threshold
        ]

    def generate_variants(
        self,
        result: AncestralResult,
        num_variants: int = 10,
        positions: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Generate variant ancestral sequences by sampling alternatives.

        Useful for exploring uncertainty in reconstruction.

        Args:
            result: AncestralResult from reconstruction.
            num_variants: Number of variants to generate.
            positions: Optional specific positions to vary.

        Returns:
            List of variant sequences.
        """
        import random

        if positions is None:
            # Use ambiguous positions
            positions = self.get_ambiguous_positions(result)

        variants = [result.sequence]  # Include ML/consensus

        for _ in range(num_variants - 1):
            seq = list(result.sequence)

            for pos in positions:
                if pos >= len(result.alternative_residues):
                    continue

                probs = result.alternative_residues[pos]
                if not probs:
                    continue

                # Sample from distribution
                r = random.random()
                cumulative = 0.0
                for aa, p in sorted(probs.items(), key=lambda x: -x[1]):
                    cumulative += p
                    if r <= cumulative:
                        seq[pos] = aa
                        break

            variant = "".join(seq)
            if variant not in variants:
                variants.append(variant)

        return variants


def reconstruct_ancestral_sequence(
    alignment: Union[List[str], Path],
    method: str = "likelihood",
) -> AncestralResult:
    """
    Reconstruct ancestral sequence from alignment.

    Convenience function for quick reconstruction.

    Args:
        alignment: List of sequences or path to alignment file.
        method: Reconstruction method.

    Returns:
        AncestralResult with reconstructed sequence.
    """
    if isinstance(alignment, Path) or (
        isinstance(alignment, str) and len(alignment) < 1000 and Path(alignment).exists()
    ):
        alignment = _load_alignment(Path(alignment))

    reconstructor = AncestralReconstructor(method=method)
    return reconstructor.reconstruct(alignment)


def _load_alignment(path: Path) -> List[str]:
    """Load alignment from FASTA/A3M file."""
    sequences = []
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)

        if current_seq:
            sequences.append("".join(current_seq))

    return sequences
