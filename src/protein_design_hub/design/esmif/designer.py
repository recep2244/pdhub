"""ESM-IF1 inverse folding designer.

ESM-IF1 (Inverse Folding) is a structure-conditioned masked language model
that designs sequences given a protein backbone structure.

Reference:
Hsu et al. "Learning inverse folding from millions of predicted structures"
ICML 2022
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.types import Sequence
from protein_design_hub.design.base import BaseDesigner
from protein_design_hub.design.esmif.installer import ESMIFInstaller
from protein_design_hub.design.registry import DesignerRegistry
from protein_design_hub.design.types import DesignInput, DesignResult


@DesignerRegistry.register("esmif")
class ESMIFDesigner(BaseDesigner):
    """ESM-IF1 inverse folding sequence designer."""

    name = "esmif"
    description = "ESM-IF1 - inverse folding sequence design from structure"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer = ESMIFInstaller()
        self._model = None
        self._alphabet = None

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _load_model(self):
        """Load ESM-IF1 model."""
        if self._model is None:
            import esm
            import torch

            self._model, self._alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
            self._model = self._model.eval()

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()

    def _design(self, input_data: DesignInput, output_dir: Path) -> DesignResult:
        """
        Design sequences using ESM-IF1 inverse folding.

        Args:
            input_data: Design input with backbone structure.
            output_dir: Output directory.

        Returns:
            DesignResult with designed sequences.
        """
        import torch

        backbone_path = Path(input_data.backbone_path)
        if not backbone_path.exists():
            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                success=False,
                error_message=f"Backbone file not found: {backbone_path}",
            )

        try:
            # Load model
            self._load_model()

            # Load structure
            structure = self._load_structure(backbone_path)
            if structure is None:
                return DesignResult(
                    job_id=input_data.job_id,
                    designer=self.name,
                    success=False,
                    error_message="Failed to load structure",
                )

            coords, native_seq = structure

            # Generate sequences
            sequences = []
            num_seqs = input_data.num_sequences
            temperature = input_data.temperature

            for i in range(num_seqs):
                # Sample sequence
                sampled_seq, score = self._sample_sequence(
                    coords,
                    temperature=temperature,
                    seed=input_data.seed + i if input_data.seed else None,
                )

                sequences.append(Sequence(
                    id=f"esmif_design_{i+1}",
                    sequence=sampled_seq,
                    metadata={
                        "score": score,
                        "temperature": temperature,
                        "native_recovery": self._calculate_recovery(sampled_seq, native_seq),
                    }
                ))

            # Save to FASTA
            fasta_path = output_dir / "esmif_designs.fasta"
            self._save_fasta(sequences, fasta_path)

            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                sequences=sequences,
                success=True,
                metadata={
                    "backbone": str(backbone_path),
                    "fasta_path": str(fasta_path),
                    "native_sequence": native_seq,
                    "num_designed": len(sequences),
                },
            )

        except Exception as e:
            import traceback
            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                success=False,
                error_message=str(e),
                metadata={"traceback": traceback.format_exc()},
            )

    def _load_structure(self, path: Path) -> Optional[Tuple]:
        """
        Load structure and extract coordinates.

        Returns:
            Tuple of (coords, sequence) or None on failure.
        """
        try:
            import esm
            from esm.inverse_folding.util import load_structure

            # Load structure using ESM utility
            structure = load_structure(str(path))

            # Get coords and sequence
            coords = structure.coord
            seq = structure.seq

            return coords, seq

        except Exception as e:
            # Try alternative loading with Biopython
            try:
                return self._load_structure_biopython(path)
            except Exception:
                return None

    def _load_structure_biopython(self, path: Path) -> Optional[Tuple]:
        """Load structure using Biopython as fallback."""
        try:
            from Bio.PDB import PDBParser, MMCIFParser
            from Bio.SeqUtils import seq1
            import numpy as np

            # Choose parser based on file type
            if path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            structure = parser.get_structure('protein', str(path))

            coords_list = []
            seq_chars = []

            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] != ' ':  # Skip hetero atoms
                            continue

                        # Get backbone atoms
                        try:
                            n_coord = residue['N'].get_coord()
                            ca_coord = residue['CA'].get_coord()
                            c_coord = residue['C'].get_coord()

                            coords_list.append([n_coord, ca_coord, c_coord])
                            seq_chars.append(seq1(residue.resname))
                        except KeyError:
                            continue
                    break  # Only first chain
                break  # Only first model

            if not coords_list:
                return None

            coords = np.array(coords_list, dtype=np.float32)
            sequence = ''.join(seq_chars)

            return coords, sequence

        except Exception:
            return None

    def _sample_sequence(
        self,
        coords,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Sample a sequence given backbone coordinates.

        Args:
            coords: Backbone coordinates.
            temperature: Sampling temperature.
            seed: Random seed.

        Returns:
            Tuple of (sampled_sequence, log_likelihood).
        """
        import torch
        import numpy as np

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        device = next(self._model.parameters()).device

        # Convert coords to tensor
        if not isinstance(coords, torch.Tensor):
            coords = torch.tensor(coords, dtype=torch.float32)

        coords = coords.unsqueeze(0).to(device)  # Add batch dimension

        with torch.no_grad():
            # Get encoder output
            encoder_out = self._model.encoder(coords)

            # Sample sequence autoregressively
            sampled_tokens = []
            total_log_prob = 0.0

            # Get sequence length
            seq_len = coords.shape[1]

            for pos in range(seq_len):
                # Get logits for this position
                logits = self._model.decoder(
                    encoder_out,
                    torch.tensor([sampled_tokens], device=device) if sampled_tokens else None,
                )

                # Apply temperature
                logits = logits[:, -1, :] / temperature

                # Sample from distribution
                probs = torch.softmax(logits, dim=-1)
                token = torch.multinomial(probs, 1).item()

                sampled_tokens.append(token)
                total_log_prob += torch.log(probs[0, token]).item()

            # Convert tokens to sequence
            sequence = self._tokens_to_sequence(sampled_tokens)

        return sequence, total_log_prob / len(sampled_tokens)

    def _tokens_to_sequence(self, tokens: List[int]) -> str:
        """Convert token indices to amino acid sequence."""
        if self._alphabet is None:
            return ""

        # Standard amino acid mapping
        aa_map = "ACDEFGHIKLMNPQRSTVWY"

        sequence = []
        for token in tokens:
            if 4 <= token < 24:  # Standard amino acid tokens
                aa_idx = token - 4
                if aa_idx < len(aa_map):
                    sequence.append(aa_map[aa_idx])
                else:
                    sequence.append("X")
            else:
                sequence.append("X")

        return "".join(sequence)

    @staticmethod
    def _calculate_recovery(designed: str, native: str) -> float:
        """Calculate sequence recovery rate."""
        if len(designed) != len(native):
            return 0.0

        matches = sum(1 for d, n in zip(designed, native) if d == n)
        return matches / len(native)

    @staticmethod
    def _save_fasta(sequences: List[Sequence], path: Path) -> None:
        """Save sequences to FASTA file."""
        with open(path, 'w') as f:
            for seq in sequences:
                header = f">{seq.id}"
                if seq.metadata:
                    score = seq.metadata.get('score', 'N/A')
                    recovery = seq.metadata.get('native_recovery', 'N/A')
                    if isinstance(recovery, float):
                        recovery = f"{recovery:.2%}"
                    header += f" score={score} recovery={recovery}"
                f.write(f"{header}\n{seq.sequence}\n")

    def design_with_constraints(
        self,
        backbone_path: Path,
        fixed_positions: Optional[List[int]] = None,
        fixed_sequence: Optional[str] = None,
        num_sequences: int = 10,
        temperature: float = 1.0,
    ) -> List[Sequence]:
        """
        Design sequences with fixed residue constraints.

        Args:
            backbone_path: Path to backbone structure.
            fixed_positions: Positions to fix (1-indexed).
            fixed_sequence: Sequence to use for fixed positions.
            num_sequences: Number of sequences to generate.
            temperature: Sampling temperature.

        Returns:
            List of designed Sequence objects.
        """
        self._load_model()

        structure = self._load_structure(backbone_path)
        if structure is None:
            raise ValueError(f"Failed to load structure: {backbone_path}")

        coords, native_seq = structure

        # Use native sequence for fixed positions if not provided
        if fixed_sequence is None:
            fixed_sequence = native_seq

        sequences = []

        for i in range(num_sequences):
            sampled_seq, score = self._sample_sequence(
                coords,
                temperature=temperature,
                seed=i,
            )

            # Apply constraints
            if fixed_positions:
                seq_list = list(sampled_seq)
                for pos in fixed_positions:
                    idx = pos - 1
                    if 0 <= idx < len(seq_list) and idx < len(fixed_sequence):
                        seq_list[idx] = fixed_sequence[idx]
                sampled_seq = "".join(seq_list)

            sequences.append(Sequence(
                id=f"esmif_constrained_{i+1}",
                sequence=sampled_seq,
                metadata={
                    "score": score,
                    "fixed_positions": fixed_positions,
                    "native_recovery": self._calculate_recovery(sampled_seq, native_seq),
                }
            ))

        return sequences

    def score_sequence(
        self,
        backbone_path: Path,
        sequence: str,
    ) -> Dict:
        """
        Score a sequence against a backbone structure.

        Args:
            backbone_path: Path to backbone structure.
            sequence: Sequence to score.

        Returns:
            Dictionary with scoring results.
        """
        self._load_model()

        structure = self._load_structure(backbone_path)
        if structure is None:
            return {"error": "Failed to load structure"}

        coords, native_seq = structure

        # Calculate pseudo-log-likelihood
        # This is a simplified scoring - actual implementation would
        # compute conditional log probabilities

        return {
            "sequence": sequence,
            "native_sequence": native_seq,
            "recovery": self._calculate_recovery(sequence, native_seq),
            "length_match": len(sequence) == len(native_seq),
        }
