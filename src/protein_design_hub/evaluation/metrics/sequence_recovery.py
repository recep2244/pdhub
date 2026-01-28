"""Sequence recovery metric for design evaluation."""

from pathlib import Path
from typing import Dict, List, Optional, Any

from protein_design_hub.evaluation.base import BaseMetric


class SequenceRecoveryMetric(BaseMetric):
    """
    Calculate sequence recovery rate between designed and native sequences.

    Useful for evaluating inverse folding / sequence design methods.
    """

    @property
    def name(self) -> str:
        return "sequence_recovery"

    @property
    def description(self) -> str:
        return "Sequence identity between designed and native sequences"

    @property
    def requires_reference(self) -> bool:
        return True  # Need native sequence

    def is_available(self) -> bool:
        return True  # No external dependencies

    def get_requirements(self) -> str:
        return "No external requirements"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        designed_sequence: Optional[str] = None,
        native_sequence: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute sequence recovery.

        Args:
            model_path: Path to designed structure (to extract sequence).
            reference_path: Path to native structure (to extract native sequence).
            designed_sequence: Optional direct sequence input.
            native_sequence: Optional direct native sequence.

        Returns:
            Dictionary with recovery metrics.
        """
        # Get sequences from structures if not provided
        if designed_sequence is None:
            designed_sequence = self._extract_sequence(model_path)

        if native_sequence is None and reference_path:
            native_sequence = self._extract_sequence(reference_path)

        if not designed_sequence or not native_sequence:
            return {"error": "Could not extract sequences"}

        # Align sequences if different lengths
        if len(designed_sequence) != len(native_sequence):
            # Simple global alignment
            designed_sequence, native_sequence = self._align_sequences(
                designed_sequence, native_sequence
            )

        # Calculate metrics
        matches = 0
        total = 0
        per_residue_match = []

        for d, n in zip(designed_sequence, native_sequence):
            if d != "-" and n != "-":
                total += 1
                if d == n:
                    matches += 1
                    per_residue_match.append(1.0)
                else:
                    per_residue_match.append(0.0)

        recovery = matches / total if total > 0 else 0.0

        # Calculate per-region recovery
        regions = self._calculate_region_recovery(
            designed_sequence, native_sequence
        )

        return {
            "sequence_recovery": recovery,
            "matches": matches,
            "total_aligned": total,
            "designed_length": len(designed_sequence.replace("-", "")),
            "native_length": len(native_sequence.replace("-", "")),
            "per_residue_match": per_residue_match,
            "region_recovery": regions,
        }

    def _extract_sequence(self, structure_path: Path) -> Optional[str]:
        """Extract sequence from structure file."""
        try:
            from Bio.PDB import PDBParser, MMCIFParser
            from Bio.SeqUtils import seq1

            structure_path = Path(structure_path)

            if structure_path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            structure = parser.get_structure('struct', str(structure_path))

            sequence_chars = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            sequence_chars.append(seq1(residue.resname))
                break

            return "".join(sequence_chars)

        except ImportError:
            return None
        except Exception:
            return None

    def _align_sequences(
        self, seq1: str, seq2: str
    ) -> tuple:
        """Simple Needleman-Wunsch alignment."""
        try:
            from Bio import pairwise2
            alignments = pairwise2.align.globalms(
                seq1, seq2,
                2, -1,  # Match, mismatch
                -2, -1,  # Gap open, extend
            )
            if alignments:
                return alignments[0][0], alignments[0][1]
        except ImportError:
            pass

        # Fallback: pad shorter sequence
        if len(seq1) < len(seq2):
            seq1 = seq1 + "-" * (len(seq2) - len(seq1))
        elif len(seq2) < len(seq1):
            seq2 = seq2 + "-" * (len(seq1) - len(seq2))

        return seq1, seq2

    def _calculate_region_recovery(
        self, designed: str, native: str, window: int = 10
    ) -> List[Dict]:
        """Calculate recovery in sliding windows."""
        regions = []

        for i in range(0, len(designed) - window + 1, window):
            d_region = designed[i:i + window]
            n_region = native[i:i + window]

            matches = sum(1 for d, n in zip(d_region, n_region) if d == n and d != "-")
            total = sum(1 for d, n in zip(d_region, n_region) if d != "-" and n != "-")

            if total > 0:
                regions.append({
                    "start": i + 1,
                    "end": i + window,
                    "recovery": matches / total,
                })

        return regions
