"""Position-Specific Scoring Matrix (PSSM) calculations.

Provides methods for generating and using PSSMs for sequence analysis
and design scoring.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP_CHAR = "-"

# BLOSUM62 background frequencies
BLOSUM62_BACKGROUND = {
    "A": 0.0740, "C": 0.0250, "D": 0.0540, "E": 0.0540,
    "F": 0.0470, "G": 0.0740, "H": 0.0260, "I": 0.0680,
    "K": 0.0580, "L": 0.0990, "M": 0.0250, "N": 0.0450,
    "P": 0.0390, "Q": 0.0340, "R": 0.0520, "S": 0.0570,
    "T": 0.0510, "V": 0.0730, "W": 0.0130, "Y": 0.0320,
}


@dataclass
class PSSMResult:
    """Container for PSSM analysis results."""

    matrix: List[Dict[str, float]]  # Position -> {AA: score}
    consensus_sequence: str
    information_content: List[float]  # Per-position IC
    total_information: float
    sequence_weights: Optional[List[float]] = None


class PSSMCalculator:
    """Generate and analyze Position-Specific Scoring Matrices."""

    def __init__(
        self,
        pseudocount: float = 0.0,
        beta: float = 10.0,
        background_freq: Optional[Dict[str, float]] = None,
        use_sequence_weighting: bool = True,
    ):
        """
        Initialize PSSM calculator.

        Args:
            pseudocount: Direct pseudocount to add to counts.
            beta: Beta parameter for position-based pseudocounts.
            background_freq: Background amino acid frequencies.
            use_sequence_weighting: Apply Henikoff sequence weighting.
        """
        self.pseudocount = pseudocount
        self.beta = beta
        self.background_freq = background_freq or BLOSUM62_BACKGROUND
        self.use_sequence_weighting = use_sequence_weighting

    def calculate_pssm(
        self,
        alignment: List[str],
        query_sequence: Optional[str] = None,
    ) -> PSSMResult:
        """
        Calculate PSSM from multiple sequence alignment.

        Args:
            alignment: List of aligned sequences.
            query_sequence: Optional query for position mapping.

        Returns:
            PSSMResult with scoring matrix and metadata.
        """
        if not alignment:
            raise ValueError("Empty alignment provided")

        n_seqs = len(alignment)
        length = len(alignment[0])

        # Calculate sequence weights
        if self.use_sequence_weighting and n_seqs > 1:
            weights = self._calculate_henikoff_weights(alignment)
        else:
            weights = [1.0] * n_seqs

        # Calculate weighted frequencies
        freq_matrix = self._calculate_frequencies(alignment, weights)

        # Apply pseudocounts
        freq_matrix = self._apply_pseudocounts(freq_matrix, n_seqs)

        # Convert to log-odds scores
        pssm = []
        consensus = []
        ic_values = []

        for pos in range(length):
            scores = {}
            max_score = -float("inf")
            max_aa = "X"

            # Calculate log-odds and find consensus
            for aa in AMINO_ACIDS:
                freq = freq_matrix[pos].get(aa, 0.0)
                bg = self.background_freq.get(aa, 0.05)

                if freq > 0 and bg > 0:
                    score = math.log2(freq / bg)
                else:
                    score = -10.0  # Penalty for impossible residues

                scores[aa] = score

                if freq > max_score:
                    max_score = freq
                    max_aa = aa

            pssm.append(scores)
            consensus.append(max_aa)

            # Calculate information content
            ic = self._calculate_ic(freq_matrix[pos])
            ic_values.append(ic)

        return PSSMResult(
            matrix=pssm,
            consensus_sequence="".join(consensus),
            information_content=ic_values,
            total_information=sum(ic_values),
            sequence_weights=weights,
        )

    def _calculate_frequencies(
        self,
        alignment: List[str],
        weights: List[float],
    ) -> List[Dict[str, float]]:
        """Calculate weighted amino acid frequencies per position."""
        length = len(alignment[0])
        freq_matrix = []

        for pos in range(length):
            counts = {aa: 0.0 for aa in AMINO_ACIDS}

            for seq_idx, seq in enumerate(alignment):
                aa = seq[pos].upper()
                if aa in AMINO_ACIDS:
                    counts[aa] += weights[seq_idx]

            total = sum(counts.values())
            if total > 0:
                freq = {aa: c / total for aa, c in counts.items()}
            else:
                freq = {aa: 0.0 for aa in AMINO_ACIDS}

            freq_matrix.append(freq)

        return freq_matrix

    def _apply_pseudocounts(
        self,
        freq_matrix: List[Dict[str, float]],
        n_seqs: int,
    ) -> List[Dict[str, float]]:
        """Apply pseudocounts to frequency matrix."""
        alpha = n_seqs - 1
        adjusted = []

        for pos_freq in freq_matrix:
            new_freq = {}
            for aa in AMINO_ACIDS:
                # Position-based pseudocount formula
                observed = pos_freq.get(aa, 0.0)
                bg = self.background_freq.get(aa, 0.05)

                if self.beta > 0:
                    # Weighted pseudocount approach
                    new_freq[aa] = (alpha * observed + self.beta * bg) / (alpha + self.beta)
                else:
                    # Simple pseudocount
                    new_freq[aa] = (observed * n_seqs + self.pseudocount) / (
                        n_seqs + self.pseudocount * len(AMINO_ACIDS)
                    )

            adjusted.append(new_freq)

        return adjusted

    def _calculate_henikoff_weights(
        self,
        alignment: List[str],
    ) -> List[float]:
        """
        Calculate Henikoff sequence weights.

        Based on Henikoff & Henikoff (1994) position-based weights.
        """
        n_seqs = len(alignment)
        length = len(alignment[0])
        weights = [0.0] * n_seqs

        for pos in range(length):
            # Count unique amino acids at position
            aa_counts = {}
            for seq_idx, seq in enumerate(alignment):
                aa = seq[pos].upper()
                if aa in AMINO_ACIDS:
                    if aa not in aa_counts:
                        aa_counts[aa] = []
                    aa_counts[aa].append(seq_idx)

            r = len(aa_counts)  # Number of unique residues
            if r == 0:
                continue

            # Add weight contribution
            for aa, seq_indices in aa_counts.items():
                s = len(seq_indices)  # Number of sequences with this AA
                for seq_idx in seq_indices:
                    weights[seq_idx] += 1.0 / (r * s)

        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total * n_seqs for w in weights]

        return weights

    def _calculate_ic(self, freq: Dict[str, float]) -> float:
        """Calculate information content for a position."""
        max_ic = math.log2(len(AMINO_ACIDS))
        entropy = 0.0

        for aa in AMINO_ACIDS:
            p = freq.get(aa, 0.0)
            if p > 0:
                entropy -= p * math.log2(p)

        return max_ic - entropy

    def score_sequence(
        self,
        sequence: str,
        pssm: PSSMResult,
        start_pos: int = 0,
    ) -> Tuple[float, List[float]]:
        """
        Score a sequence using a PSSM.

        Args:
            sequence: Sequence to score.
            pssm: PSSM result from calculate_pssm.
            start_pos: Starting position in PSSM (for partial scoring).

        Returns:
            Tuple of (total_score, per_position_scores).
        """
        scores = []
        for i, aa in enumerate(sequence.upper()):
            pos = start_pos + i
            if pos >= len(pssm.matrix):
                break

            if aa in AMINO_ACIDS:
                score = pssm.matrix[pos].get(aa, -10.0)
            else:
                score = 0.0  # Neutral for unknown/gap

            scores.append(score)

        return sum(scores), scores

    def suggest_mutations(
        self,
        sequence: str,
        pssm: PSSMResult,
        threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Suggest beneficial mutations based on PSSM.

        Args:
            sequence: Query sequence.
            pssm: PSSM result.
            threshold: Minimum score improvement to report.

        Returns:
            List of mutation suggestions with scores.
        """
        suggestions = []

        for pos, aa in enumerate(sequence.upper()):
            if pos >= len(pssm.matrix):
                break

            if aa not in AMINO_ACIDS:
                continue

            current_score = pssm.matrix[pos].get(aa, -10.0)

            # Find better alternatives
            for alt_aa in AMINO_ACIDS:
                if alt_aa == aa:
                    continue

                alt_score = pssm.matrix[pos].get(alt_aa, -10.0)
                improvement = alt_score - current_score

                if improvement > threshold:
                    suggestions.append({
                        "position": pos + 1,  # 1-indexed
                        "original": aa,
                        "mutant": alt_aa,
                        "original_score": current_score,
                        "mutant_score": alt_score,
                        "improvement": improvement,
                        "information_content": pssm.information_content[pos],
                    })

        # Sort by improvement
        suggestions.sort(key=lambda x: x["improvement"], reverse=True)
        return suggestions


def calculate_pssm(
    alignment: Union[List[str], Path],
    pseudocount: float = 0.0,
) -> PSSMResult:
    """
    Calculate PSSM from alignment.

    Convenience function for quick PSSM generation.

    Args:
        alignment: List of sequences or path to alignment file.
        pseudocount: Pseudocount for smoothing.

    Returns:
        PSSMResult with scoring matrix.
    """
    if isinstance(alignment, Path) or (
        isinstance(alignment, str) and len(alignment) < 1000 and Path(alignment).exists()
    ):
        alignment = _load_alignment(Path(alignment))

    calculator = PSSMCalculator(pseudocount=pseudocount)
    return calculator.calculate_pssm(alignment)


def score_sequence_with_pssm(
    sequence: str,
    alignment: Union[List[str], PSSMResult],
) -> float:
    """
    Score a sequence against a PSSM or alignment.

    Args:
        sequence: Sequence to score.
        alignment: Alignment to build PSSM from, or pre-computed PSSM.

    Returns:
        Total PSSM score.
    """
    if isinstance(alignment, PSSMResult):
        pssm = alignment
    else:
        pssm = calculate_pssm(alignment)

    calculator = PSSMCalculator()
    total_score, _ = calculator.score_sequence(sequence, pssm)
    return total_score


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
