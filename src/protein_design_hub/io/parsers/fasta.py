"""FASTA file parser."""

from pathlib import Path
from typing import Union

from protein_design_hub.core.types import Sequence, MoleculeType
from protein_design_hub.core.exceptions import InputValidationError


class FastaParser:
    """Parser for FASTA format files."""

    def __init__(self, default_molecule_type: MoleculeType = MoleculeType.PROTEIN):
        self.default_molecule_type = default_molecule_type

    def parse(self, input_data: Union[str, Path]) -> list[Sequence]:
        """
        Parse FASTA input from file path or string content.

        Args:
            input_data: Path to FASTA file or FASTA content as string.

        Returns:
            List of Sequence objects.

        Raises:
            InputValidationError: If parsing fails.
        """
        if isinstance(input_data, Path) or (
            isinstance(input_data, str) and not input_data.startswith(">")
        ):
            # Treat as file path
            path = Path(input_data)
            if not path.exists():
                raise InputValidationError(f"FASTA file not found: {path}", "input_path")
            content = path.read_text()
        else:
            content = input_data

        return self._parse_content(content)

    def _parse_content(self, content: str) -> list[Sequence]:
        """Parse FASTA content string."""
        sequences = []
        current_id = None
        current_description = ""
        current_seq_lines = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence if exists
                if current_id is not None:
                    sequences.append(
                        self._create_sequence(
                            current_id, current_description, "".join(current_seq_lines)
                        )
                    )

                # Parse header
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0] if parts else f"seq_{len(sequences) + 1}"
                current_description = parts[1] if len(parts) > 1 else ""
                current_seq_lines = []
            else:
                # Sequence line
                current_seq_lines.append(line)

        # Save last sequence
        if current_id is not None:
            sequences.append(
                self._create_sequence(
                    current_id, current_description, "".join(current_seq_lines)
                )
            )

        if not sequences:
            raise InputValidationError("No sequences found in FASTA input", "input_data")

        return sequences

    def _create_sequence(self, seq_id: str, description: str, sequence: str) -> Sequence:
        """Create a Sequence object with molecule type detection."""
        # Detect molecule type from sequence content
        molecule_type = self._detect_molecule_type(sequence)

        return Sequence(
            id=seq_id,
            sequence=sequence,
            molecule_type=molecule_type,
            description=description,
        )

    def _detect_molecule_type(self, sequence: str) -> MoleculeType:
        """Detect molecule type from sequence content."""
        seq_upper = sequence.upper()
        unique_chars = set(seq_upper)

        # Check for nucleotide sequences
        nucleotide_chars = set("ACGTU")
        if unique_chars <= nucleotide_chars:
            if "U" in unique_chars:
                return MoleculeType.RNA
            return MoleculeType.DNA

        # Default to protein
        return self.default_molecule_type

    def parse_multimer(self, input_data: Union[str, Path], chain_separator: str = ":") -> list[Sequence]:
        """
        Parse multimer FASTA where chains are separated by a special character.

        Example input:
        >complex
        MKFLILLFNILCLFPVLAAD:MNFLLSFVFVFLLPFVLVAD

        Args:
            input_data: Path to FASTA file or FASTA content.
            chain_separator: Character separating chains in sequence.

        Returns:
            List of Sequence objects, one per chain.
        """
        sequences = self.parse(input_data)

        # Expand sequences with chain separators
        expanded = []
        for seq in sequences:
            if chain_separator in seq.sequence:
                chains = seq.sequence.split(chain_separator)
                for i, chain_seq in enumerate(chains):
                    chain_id = f"{seq.id}_chain{chr(65 + i)}"  # A, B, C, ...
                    expanded.append(
                        Sequence(
                            id=chain_id,
                            sequence=chain_seq,
                            molecule_type=seq.molecule_type,
                            description=f"Chain {chr(65 + i)} of {seq.id}",
                        )
                    )
            else:
                expanded.append(seq)

        return expanded

    def write(self, sequences: list[Sequence], output_path: Path, line_width: int = 80) -> None:
        """
        Write sequences to a FASTA file.

        Args:
            sequences: List of Sequence objects to write.
            output_path: Path to output file.
            line_width: Maximum characters per sequence line.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for seq in sequences:
                # Write header
                header = f">{seq.id}"
                if seq.description:
                    header += f" {seq.description}"
                f.write(header + "\n")

                # Write sequence with line wrapping
                for i in range(0, len(seq.sequence), line_width):
                    f.write(seq.sequence[i : i + line_width] + "\n")
