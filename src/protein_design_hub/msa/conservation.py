"""Conservation scoring for multiple sequence alignments.

Provides methods for calculating position-specific conservation scores
including Shannon entropy, Kullback-Leibler divergence, and Jensen-Shannon divergence.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math

# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
GAP_CHAR = "-"

# Background amino acid frequencies (from UniProtKB/Swiss-Prot)
BACKGROUND_FREQUENCIES = {
    "A": 0.0825, "C": 0.0137, "D": 0.0546, "E": 0.0675,
    "F": 0.0386, "G": 0.0707, "H": 0.0227, "I": 0.0596,
    "K": 0.0584, "L": 0.0966, "M": 0.0242, "N": 0.0406,
    "P": 0.0470, "Q": 0.0393, "R": 0.0553, "S": 0.0656,
    "T": 0.0534, "V": 0.0687, "W": 0.0108, "Y": 0.0292,
}


@dataclass
class ConservationResult:
    """Container for conservation analysis results."""

    position: int
    shannon_entropy: float
    normalized_entropy: float  # 0-1, higher = more conserved
    js_divergence: Optional[float]
    kl_divergence: Optional[float]
    conservation_score: float  # Composite score
    consensus_residue: str
    consensus_frequency: float
    gap_frequency: float
    amino_acid_frequencies: Dict[str, float]


class ConservationCalculator:
    """Calculate conservation scores for MSA positions."""

    def __init__(
        self,
        background_freq: Optional[Dict[str, float]] = None,
        gap_penalty: float = 0.5,
        pseudocount: float = 0.0001,
    ):
        """
        Initialize conservation calculator.

        Args:
            background_freq: Background amino acid frequencies. Uses Swiss-Prot if None.
            gap_penalty: Penalty factor for gaps (0-1).
            pseudocount: Pseudocount for smoothing frequencies.
        """
        self.background_freq = background_freq or BACKGROUND_FREQUENCIES
        self.gap_penalty = gap_penalty
        self.pseudocount = pseudocount

    def analyze_alignment(
        self,
        alignment: List[str],
        query_sequence: Optional[str] = None,
    ) -> List[ConservationResult]:
        """
        Analyze conservation across all positions in alignment.

        Args:
            alignment: List of aligned sequences (strings of same length).
            query_sequence: Optional query sequence for position mapping.

        Returns:
            List of ConservationResult for each position.
        """
        if not alignment:
            return []

        alignment_length = len(alignment[0])

        # Validate alignment
        if not all(len(seq) == alignment_length for seq in alignment):
            raise ValueError("All sequences must have the same length")

        results = []
        for pos in range(alignment_length):
            column = [seq[pos].upper() for seq in alignment]
            result = self._analyze_position(pos, column)
            results.append(result)

        return results

    def _analyze_position(
        self,
        position: int,
        column: List[str],
    ) -> ConservationResult:
        """Analyze conservation at a single position."""
        n_seqs = len(column)

        # Count amino acids and gaps
        aa_counts: Dict[str, int] = {}
        gap_count = 0

        for residue in column:
            if residue in GAP_CHAR or residue == ".":
                gap_count += 1
            elif residue in AMINO_ACIDS:
                aa_counts[residue] = aa_counts.get(residue, 0) + 1
            # Non-standard residues (X, B, Z) are ignored

        # Calculate frequencies with pseudocounts
        total_aa = sum(aa_counts.values())
        gap_frequency = gap_count / n_seqs

        if total_aa == 0:
            # All gaps
            return ConservationResult(
                position=position,
                shannon_entropy=0.0,
                normalized_entropy=0.0,
                js_divergence=None,
                kl_divergence=None,
                conservation_score=0.0,
                consensus_residue=GAP_CHAR,
                consensus_frequency=gap_frequency,
                gap_frequency=gap_frequency,
                amino_acid_frequencies={},
            )

        # Calculate frequencies
        frequencies = {}
        for aa in AMINO_ACIDS:
            count = aa_counts.get(aa, 0) + self.pseudocount
            frequencies[aa] = count / (total_aa + self.pseudocount * len(AMINO_ACIDS))

        # Shannon entropy
        entropy = calculate_shannon_entropy(list(frequencies.values()))
        max_entropy = math.log2(len(AMINO_ACIDS))
        normalized_entropy = 1.0 - (entropy / max_entropy)  # Higher = more conserved

        # Jensen-Shannon divergence from background
        js_div = calculate_js_divergence(frequencies, self.background_freq)

        # Kullback-Leibler divergence from background
        kl_div = self._calculate_kl_divergence(frequencies, self.background_freq)

        # Find consensus
        if aa_counts:
            consensus = max(aa_counts.keys(), key=lambda x: aa_counts[x])
            consensus_freq = aa_counts[consensus] / total_aa
        else:
            consensus = GAP_CHAR
            consensus_freq = 0.0

        # Composite conservation score
        # Combines normalized entropy with gap penalty
        conservation_score = normalized_entropy * (1.0 - self.gap_penalty * gap_frequency)

        return ConservationResult(
            position=position,
            shannon_entropy=entropy,
            normalized_entropy=normalized_entropy,
            js_divergence=js_div,
            kl_divergence=kl_div,
            conservation_score=conservation_score,
            consensus_residue=consensus,
            consensus_frequency=consensus_freq,
            gap_frequency=gap_frequency,
            amino_acid_frequencies=frequencies,
        )

    def _calculate_kl_divergence(
        self,
        observed: Dict[str, float],
        background: Dict[str, float],
    ) -> float:
        """Calculate Kullback-Leibler divergence D(observed || background)."""
        kl = 0.0
        for aa in AMINO_ACIDS:
            p = observed.get(aa, self.pseudocount)
            q = background.get(aa, self.pseudocount)
            if p > 0 and q > 0:
                kl += p * math.log2(p / q)
        return kl

    def get_conserved_positions(
        self,
        alignment: List[str],
        threshold: float = 0.7,
    ) -> List[int]:
        """
        Get positions with conservation score above threshold.

        Args:
            alignment: List of aligned sequences.
            threshold: Conservation score threshold (0-1).

        Returns:
            List of position indices.
        """
        results = self.analyze_alignment(alignment)
        return [r.position for r in results if r.conservation_score >= threshold]

    def get_variable_positions(
        self,
        alignment: List[str],
        threshold: float = 0.3,
    ) -> List[int]:
        """
        Get positions with conservation score below threshold (variable positions).

        Args:
            alignment: List of aligned sequences.
            threshold: Conservation score threshold (0-1).

        Returns:
            List of position indices.
        """
        results = self.analyze_alignment(alignment)
        return [r.position for r in results if r.conservation_score < threshold]


def calculate_shannon_entropy(probabilities: List[float]) -> float:
    """
    Calculate Shannon entropy of a probability distribution.

    Args:
        probabilities: List of probabilities (must sum to ~1).

    Returns:
        Shannon entropy in bits.
    """
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_js_divergence(
    p: Dict[str, float],
    q: Dict[str, float],
) -> float:
    """
    Calculate Jensen-Shannon divergence between two distributions.

    JS divergence is a symmetric measure bounded between 0 and 1.

    Args:
        p: First distribution.
        q: Second distribution.

    Returns:
        Jensen-Shannon divergence.
    """
    # Calculate midpoint distribution
    m = {}
    all_keys = set(p.keys()) | set(q.keys())

    for key in all_keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        m[key] = (p_val + q_val) / 2.0

    # Calculate KL divergences
    kl_pm = 0.0
    kl_qm = 0.0

    for key in all_keys:
        p_val = p.get(key, 0.0)
        q_val = q.get(key, 0.0)
        m_val = m[key]

        if p_val > 0 and m_val > 0:
            kl_pm += p_val * math.log2(p_val / m_val)
        if q_val > 0 and m_val > 0:
            kl_qm += q_val * math.log2(q_val / m_val)

    return (kl_pm + kl_qm) / 2.0


def calculate_conservation(
    alignment: Union[List[str], Path],
    method: str = "shannon",
) -> List[float]:
    """
    Calculate conservation scores for an alignment.

    Convenience function for quick conservation analysis.

    Args:
        alignment: List of aligned sequences or path to alignment file.
        method: Conservation method ("shannon", "js", or "composite").

    Returns:
        List of conservation scores (0-1, higher = more conserved).
    """
    if isinstance(alignment, Path) or (isinstance(alignment, str) and len(alignment) < 1000 and Path(alignment).exists()):
        alignment = _load_alignment(Path(alignment))

    calculator = ConservationCalculator()
    results = calculator.analyze_alignment(alignment)

    if method == "shannon":
        return [r.normalized_entropy for r in results]
    elif method == "js":
        return [r.js_divergence or 0.0 for r in results]
    else:  # composite
        return [r.conservation_score for r in results]


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
