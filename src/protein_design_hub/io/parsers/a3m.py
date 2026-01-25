"""A3M (MSA) file parser."""

from pathlib import Path
from typing import Union

from protein_design_hub.core.types import MSA
from protein_design_hub.core.exceptions import InputValidationError


class A3MParser:
    """Parser for A3M format multiple sequence alignments."""

    def parse(self, input_data: Union[str, Path]) -> MSA:
        """
        Parse A3M input from file path or string content.

        Args:
            input_data: Path to A3M file or A3M content as string.

        Returns:
            MSA object containing the alignment.

        Raises:
            InputValidationError: If parsing fails.
        """
        if isinstance(input_data, Path) or (
            isinstance(input_data, str) and not input_data.startswith(">")
        ):
            path = Path(input_data)
            if not path.exists():
                raise InputValidationError(f"A3M file not found: {path}", "input_path")
            content = path.read_text()
        else:
            content = input_data

        return self._parse_content(content)

    def _parse_content(self, content: str) -> MSA:
        """Parse A3M content string."""
        sequences = []
        current_header = None
        current_seq_lines = []
        query_id = None

        for line in content.strip().split("\n"):
            line = line.rstrip()

            if line.startswith(">"):
                # Save previous sequence if exists
                if current_header is not None:
                    seq = "".join(current_seq_lines)
                    sequences.append((current_header, seq))

                    # First sequence is the query
                    if query_id is None:
                        query_id = current_header.split()[0]

                current_header = line[1:].strip()
                current_seq_lines = []
            elif line.startswith("#"):
                # Skip comment lines
                continue
            else:
                current_seq_lines.append(line)

        # Save last sequence
        if current_header is not None:
            seq = "".join(current_seq_lines)
            sequences.append((current_header, seq))

            if query_id is None:
                query_id = current_header.split()[0]

        if not sequences:
            raise InputValidationError("No sequences found in A3M input", "input_data")

        return MSA(
            query_id=query_id or "query",
            sequences=sequences,
            format="a3m",
        )

    def write(self, msa: MSA, output_path: Path, line_width: int = 80) -> None:
        """
        Write MSA to an A3M file.

        Args:
            msa: MSA object to write.
            output_path: Path to output file.
            line_width: Maximum characters per sequence line.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for header, sequence in msa.sequences:
                f.write(f">{header}\n")
                for i in range(0, len(sequence), line_width):
                    f.write(sequence[i : i + line_width] + "\n")

    def to_stockholm(self, msa: MSA) -> str:
        """
        Convert MSA to Stockholm format.

        Args:
            msa: MSA object to convert.

        Returns:
            Stockholm format string.
        """
        lines = ["# STOCKHOLM 1.0", ""]

        # Find max header length for alignment
        max_header_len = max(len(h.split()[0]) for h, _ in msa.sequences)

        for header, sequence in msa.sequences:
            # Remove gaps from A3M format (lowercase = insertions)
            aligned_seq = "".join(c for c in sequence if c.isupper() or c == "-")
            name = header.split()[0]
            lines.append(f"{name:<{max_header_len}}  {aligned_seq}")

        lines.append("//")
        return "\n".join(lines)

    def remove_insertions(self, msa: MSA) -> MSA:
        """
        Remove insertion columns (lowercase letters) from A3M format.

        Args:
            msa: MSA object with A3M format sequences.

        Returns:
            New MSA with insertions removed.
        """
        cleaned_sequences = []
        for header, sequence in msa.sequences:
            cleaned_seq = "".join(c for c in sequence if c.isupper() or c == "-")
            cleaned_sequences.append((header, cleaned_seq))

        return MSA(
            query_id=msa.query_id,
            sequences=cleaned_sequences,
            format="a3m",
        )

    def get_depth_per_position(self, msa: MSA) -> list[int]:
        """
        Calculate the number of non-gap residues at each position.

        Args:
            msa: MSA object.

        Returns:
            List of depth counts per position.
        """
        if not msa.sequences:
            return []

        # Remove insertions first
        cleaned = self.remove_insertions(msa)

        # Get alignment length from first sequence
        aln_length = len(cleaned.sequences[0][1])
        depths = [0] * aln_length

        for _, sequence in cleaned.sequences:
            for i, char in enumerate(sequence):
                if char != "-":
                    depths[i] += 1

        return depths
