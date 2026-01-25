"""Constraint file parser."""

import csv
import json
from pathlib import Path
from typing import Union

from protein_design_hub.core.types import Constraint
from protein_design_hub.core.exceptions import InputValidationError


class ConstraintParser:
    """Parser for distance and contact constraint files."""

    def parse(self, input_data: Union[str, Path]) -> list[Constraint]:
        """
        Parse constraints from file.

        Supports formats:
        - CSV/TSV: residue1,residue2,distance_min,distance_max,chain1,chain2,weight
        - JSON: [{"residue1": 1, "residue2": 10, ...}, ...]
        - Simple text: "1-10 8.0" (contact at 8 Angstroms)

        Args:
            input_data: Path to constraint file or constraint string.

        Returns:
            List of Constraint objects.

        Raises:
            InputValidationError: If parsing fails.
        """
        path = Path(input_data)

        if not path.exists():
            raise InputValidationError(f"Constraint file not found: {path}", "path")

        suffix = path.suffix.lower()
        content = path.read_text()

        if suffix == ".json":
            return self._parse_json(content)
        elif suffix in (".csv", ".tsv"):
            delimiter = "\t" if suffix == ".tsv" else ","
            return self._parse_csv(content, delimiter)
        else:
            # Try to auto-detect format
            if content.strip().startswith("["):
                return self._parse_json(content)
            elif "," in content or "\t" in content:
                delimiter = "\t" if "\t" in content else ","
                return self._parse_csv(content, delimiter)
            else:
                return self._parse_simple(content)

    def _parse_json(self, content: str) -> list[Constraint]:
        """Parse JSON format constraints."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise InputValidationError(f"Invalid JSON: {e}", "content")

        constraints = []
        for item in data:
            constraints.append(
                Constraint(
                    residue1=item["residue1"],
                    residue2=item["residue2"],
                    distance_min=item.get("distance_min", 0.0),
                    distance_max=item.get("distance_max", 8.0),
                    chain1=item.get("chain1", "A"),
                    chain2=item.get("chain2", "A"),
                    weight=item.get("weight", 1.0),
                )
            )

        return constraints

    def _parse_csv(self, content: str, delimiter: str) -> list[Constraint]:
        """Parse CSV/TSV format constraints."""
        constraints = []
        reader = csv.DictReader(content.strip().split("\n"), delimiter=delimiter)

        for row in reader:
            # Handle different column name conventions
            residue1 = int(row.get("residue1") or row.get("res1") or row.get("i"))
            residue2 = int(row.get("residue2") or row.get("res2") or row.get("j"))

            constraint = Constraint(
                residue1=residue1,
                residue2=residue2,
                distance_min=float(row.get("distance_min", row.get("d_min", 0.0))),
                distance_max=float(row.get("distance_max", row.get("d_max", 8.0))),
                chain1=row.get("chain1", row.get("chain_i", "A")),
                chain2=row.get("chain2", row.get("chain_j", "A")),
                weight=float(row.get("weight", 1.0)),
            )
            constraints.append(constraint)

        return constraints

    def _parse_simple(self, content: str) -> list[Constraint]:
        """
        Parse simple text format constraints.

        Format: "residue1-residue2 distance" or "residue1 residue2 distance"
        Example: "10-50 8.0" means residues 10 and 50 should be within 8 Angstroms
        """
        constraints = []

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            # Handle "10-50 8.0" format
            if "-" in parts[0]:
                res_parts = parts[0].split("-")
                residue1 = int(res_parts[0])
                residue2 = int(res_parts[1])
                distance = float(parts[1]) if len(parts) > 1 else 8.0
            # Handle "10 50 8.0" format
            else:
                residue1 = int(parts[0])
                residue2 = int(parts[1])
                distance = float(parts[2]) if len(parts) > 2 else 8.0

            constraints.append(
                Constraint(
                    residue1=residue1,
                    residue2=residue2,
                    distance_min=0.0,
                    distance_max=distance,
                )
            )

        return constraints

    def write(self, constraints: list[Constraint], output_path: Path, format: str = "json") -> None:
        """
        Write constraints to file.

        Args:
            constraints: List of Constraint objects.
            output_path: Path to output file.
            format: Output format ("json", "csv", or "tsv").
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [
                {
                    "residue1": c.residue1,
                    "residue2": c.residue2,
                    "distance_min": c.distance_min,
                    "distance_max": c.distance_max,
                    "chain1": c.chain1,
                    "chain2": c.chain2,
                    "weight": c.weight,
                }
                for c in constraints
            ]
            output_path.write_text(json.dumps(data, indent=2))

        elif format in ("csv", "tsv"):
            delimiter = "\t" if format == "tsv" else ","
            lines = ["residue1,residue2,distance_min,distance_max,chain1,chain2,weight".replace(",", delimiter)]
            for c in constraints:
                line = delimiter.join([
                    str(c.residue1),
                    str(c.residue2),
                    str(c.distance_min),
                    str(c.distance_max),
                    c.chain1,
                    c.chain2,
                    str(c.weight),
                ])
                lines.append(line)
            output_path.write_text("\n".join(lines))

    def to_alphafold_format(self, constraints: list[Constraint]) -> str:
        """
        Convert constraints to AlphaFold-compatible format.

        Returns a string representation for use with ColabFold.
        """
        # ColabFold doesn't directly support constraints, but this could be
        # used with custom modifications or other tools
        lines = []
        for c in constraints:
            lines.append(f"{c.chain1}{c.residue1}-{c.chain2}{c.residue2}:{c.distance_max}")
        return "\n".join(lines)
