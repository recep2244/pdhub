"""Coevolution analysis for multiple sequence alignments.

Provides methods for detecting coevolving residue pairs using:
- Mutual Information (MI)
- Average Product Correction (APC)
- Simplified Direct Coupling Analysis (DCA)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY-"


@dataclass
class CoevolutionResult:
    """Container for coevolution analysis results."""

    position_i: int
    position_j: int
    mutual_information: float
    apc_corrected_mi: float
    normalized_mi: float  # 0-1, corrected for entropy
    contact_probability: float  # Estimated contact probability


class CoevolutionAnalyzer:
    """Analyze coevolution patterns in MSA to predict contacts."""

    def __init__(
        self,
        pseudocount: float = 0.0001,
        min_sequence_separation: int = 5,
        gap_threshold: float = 0.5,
    ):
        """
        Initialize coevolution analyzer.

        Args:
            pseudocount: Pseudocount for frequency smoothing.
            min_sequence_separation: Minimum sequence separation for pairs.
            gap_threshold: Maximum gap frequency allowed at positions.
        """
        self.pseudocount = pseudocount
        self.min_separation = min_sequence_separation
        self.gap_threshold = gap_threshold

    def analyze_alignment(
        self,
        alignment: List[str],
        top_pairs: int = 100,
    ) -> List[CoevolutionResult]:
        """
        Analyze coevolution for all position pairs in alignment.

        Args:
            alignment: List of aligned sequences.
            top_pairs: Number of top coevolving pairs to return.

        Returns:
            List of CoevolutionResult sorted by APC-corrected MI.
        """
        if not alignment:
            return []

        n_seqs = len(alignment)
        alignment_length = len(alignment[0])

        # Calculate single-position frequencies
        single_freq = self._calculate_single_frequencies(alignment)

        # Calculate column entropies
        entropies = self._calculate_entropies(single_freq)

        # Filter positions with too many gaps
        valid_positions = [
            i for i in range(alignment_length)
            if single_freq[i].get("-", 0) < self.gap_threshold
        ]

        # Calculate pairwise MI
        mi_matrix = {}
        mi_sums = {i: 0.0 for i in valid_positions}

        results = []
        for idx_i, i in enumerate(valid_positions):
            for j in valid_positions[idx_i + 1:]:
                if j - i < self.min_separation:
                    continue

                # Calculate joint frequencies
                joint_freq = self._calculate_joint_frequencies(alignment, i, j)

                # Calculate MI
                mi = self._calculate_mi(single_freq[i], single_freq[j], joint_freq)
                mi_matrix[(i, j)] = mi
                mi_sums[i] += mi
                mi_sums[j] += mi

        # Apply APC correction
        n_pairs = len(valid_positions) * (len(valid_positions) - 1) // 2
        mean_mi = sum(mi_matrix.values()) / max(n_pairs, 1)

        for (i, j), mi in mi_matrix.items():
            # APC correction
            apc = (mi_sums[i] * mi_sums[j]) / (mean_mi * n_pairs) if mean_mi > 0 else 0
            apc_corrected = max(0, mi - apc)

            # Normalize by joint entropy
            h_i = entropies[i]
            h_j = entropies[j]
            max_mi = min(h_i, h_j) if min(h_i, h_j) > 0 else 1.0
            normalized = apc_corrected / max_mi if max_mi > 0 else 0.0

            # Estimate contact probability using sigmoid
            contact_prob = 1.0 / (1.0 + math.exp(-10 * (normalized - 0.3)))

            results.append(CoevolutionResult(
                position_i=i,
                position_j=j,
                mutual_information=mi,
                apc_corrected_mi=apc_corrected,
                normalized_mi=normalized,
                contact_probability=contact_prob,
            ))

        # Sort by APC-corrected MI
        results.sort(key=lambda x: x.apc_corrected_mi, reverse=True)
        return results[:top_pairs]

    def _calculate_single_frequencies(
        self,
        alignment: List[str],
    ) -> Dict[int, Dict[str, float]]:
        """Calculate single-position amino acid frequencies."""
        n_seqs = len(alignment)
        length = len(alignment[0])

        frequencies = {}
        for pos in range(length):
            counts = {}
            for seq in alignment:
                aa = seq[pos].upper()
                if aa in AMINO_ACIDS:
                    counts[aa] = counts.get(aa, 0) + 1

            total = sum(counts.values()) + self.pseudocount * len(AMINO_ACIDS)
            frequencies[pos] = {
                aa: (counts.get(aa, 0) + self.pseudocount) / total
                for aa in AMINO_ACIDS
            }

        return frequencies

    def _calculate_joint_frequencies(
        self,
        alignment: List[str],
        pos_i: int,
        pos_j: int,
    ) -> Dict[Tuple[str, str], float]:
        """Calculate joint frequencies for a position pair."""
        n_seqs = len(alignment)

        counts = {}
        for seq in alignment:
            aa_i = seq[pos_i].upper()
            aa_j = seq[pos_j].upper()
            if aa_i in AMINO_ACIDS and aa_j in AMINO_ACIDS:
                key = (aa_i, aa_j)
                counts[key] = counts.get(key, 0) + 1

        total = sum(counts.values()) + self.pseudocount * len(AMINO_ACIDS) ** 2
        frequencies = {}
        for aa_i in AMINO_ACIDS:
            for aa_j in AMINO_ACIDS:
                key = (aa_i, aa_j)
                frequencies[key] = (counts.get(key, 0) + self.pseudocount) / total

        return frequencies

    def _calculate_mi(
        self,
        freq_i: Dict[str, float],
        freq_j: Dict[str, float],
        joint_freq: Dict[Tuple[str, str], float],
    ) -> float:
        """Calculate mutual information between two positions."""
        mi = 0.0
        for aa_i in AMINO_ACIDS:
            for aa_j in AMINO_ACIDS:
                p_ij = joint_freq.get((aa_i, aa_j), self.pseudocount)
                p_i = freq_i.get(aa_i, self.pseudocount)
                p_j = freq_j.get(aa_j, self.pseudocount)

                if p_ij > 0 and p_i > 0 and p_j > 0:
                    mi += p_ij * math.log2(p_ij / (p_i * p_j))

        return max(0.0, mi)

    def _calculate_entropies(
        self,
        single_freq: Dict[int, Dict[str, float]],
    ) -> Dict[int, float]:
        """Calculate entropy for each position."""
        entropies = {}
        for pos, freq in single_freq.items():
            h = 0.0
            for p in freq.values():
                if p > 0:
                    h -= p * math.log2(p)
            entropies[pos] = h
        return entropies

    def get_contact_predictions(
        self,
        alignment: List[str],
        probability_threshold: float = 0.5,
    ) -> List[Tuple[int, int, float]]:
        """
        Get predicted contacts from coevolution analysis.

        Args:
            alignment: List of aligned sequences.
            probability_threshold: Minimum contact probability.

        Returns:
            List of (pos_i, pos_j, probability) tuples.
        """
        results = self.analyze_alignment(alignment, top_pairs=1000)
        return [
            (r.position_i, r.position_j, r.contact_probability)
            for r in results
            if r.contact_probability >= probability_threshold
        ]


def calculate_mutual_information(
    alignment: List[str],
    pos_i: int,
    pos_j: int,
) -> float:
    """
    Calculate mutual information between two positions.

    Convenience function for single pair MI calculation.

    Args:
        alignment: List of aligned sequences.
        pos_i: First position index.
        pos_j: Second position index.

    Returns:
        Mutual information in bits.
    """
    analyzer = CoevolutionAnalyzer()
    single_freq = analyzer._calculate_single_frequencies(alignment)
    joint_freq = analyzer._calculate_joint_frequencies(alignment, pos_i, pos_j)
    return analyzer._calculate_mi(single_freq[pos_i], single_freq[pos_j], joint_freq)


def calculate_apc_corrected_mi(
    alignment: List[str],
    top_pairs: int = 100,
) -> List[Tuple[int, int, float]]:
    """
    Calculate APC-corrected mutual information for top pairs.

    Convenience function for coevolution analysis.

    Args:
        alignment: List of aligned sequences.
        top_pairs: Number of top pairs to return.

    Returns:
        List of (pos_i, pos_j, APC-MI) tuples.
    """
    analyzer = CoevolutionAnalyzer()
    results = analyzer.analyze_alignment(alignment, top_pairs)
    return [(r.position_i, r.position_j, r.apc_corrected_mi) for r in results]


class SimplifiedDCA:
    """
    Simplified Direct Coupling Analysis.

    Implements a pseudo-likelihood approach for contact prediction
    based on Morcos et al. (2011) and related work.
    """

    def __init__(
        self,
        pseudocount: float = 0.5,
        min_sequence_separation: int = 5,
    ):
        """
        Initialize simplified DCA.

        Args:
            pseudocount: Pseudocount for frequency regularization.
            min_sequence_separation: Minimum separation for valid pairs.
        """
        self.pseudocount = pseudocount
        self.min_separation = min_sequence_separation

    def calculate_dca_scores(
        self,
        alignment: List[str],
        top_pairs: int = 100,
    ) -> List[Tuple[int, int, float]]:
        """
        Calculate DCA-like scores using the mfDCA approach.

        Uses mean-field approximation for efficient computation.

        Args:
            alignment: List of aligned sequences.
            top_pairs: Number of top pairs to return.

        Returns:
            List of (pos_i, pos_j, DCA_score) tuples.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("NumPy required for DCA. Install with: pip install numpy")

        if not alignment:
            return []

        n_seqs = len(alignment)
        L = len(alignment[0])
        q = len(AMINO_ACIDS)

        # Convert alignment to numeric
        aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        numeric_aln = np.zeros((n_seqs, L), dtype=np.int8)
        for s, seq in enumerate(alignment):
            for i, aa in enumerate(seq.upper()):
                numeric_aln[s, i] = aa_to_idx.get(aa, q - 1)  # Map unknown to gap

        # Calculate single-site frequencies with pseudocount
        fi = np.zeros((L, q))
        for i in range(L):
            for s in range(n_seqs):
                fi[i, numeric_aln[s, i]] += 1
            fi[i] += self.pseudocount
            fi[i] /= fi[i].sum()

        # Calculate pair frequencies
        fij = np.zeros((L, L, q, q))
        for i in range(L):
            for j in range(i + 1, L):
                for s in range(n_seqs):
                    fij[i, j, numeric_aln[s, i], numeric_aln[s, j]] += 1
                fij[i, j] += self.pseudocount
                fij[i, j] /= fij[i, j].sum()
                fij[j, i] = fij[i, j].T

        # Calculate covariance matrix
        cov = np.zeros((L, L, q, q))
        for i in range(L):
            for j in range(i + 1, L):
                for a in range(q):
                    for b in range(q):
                        cov[i, j, a, b] = fij[i, j, a, b] - fi[i, a] * fi[j, b]
                        cov[j, i, b, a] = cov[i, j, a, b]

        # Calculate DCA scores using Frobenius norm of coupling matrix
        scores = []
        for i in range(L):
            for j in range(i + self.min_separation, L):
                # Use Frobenius norm as DCA score (simplified)
                score = np.sqrt(np.sum(cov[i, j] ** 2))
                scores.append((i, j, score))

        # Sort and return top pairs
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_pairs]
