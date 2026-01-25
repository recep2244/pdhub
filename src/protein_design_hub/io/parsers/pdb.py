"""PDB/mmCIF file parser."""

from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from protein_design_hub.core.types import Template, Sequence, MoleculeType
from protein_design_hub.core.exceptions import InputValidationError


@dataclass
class StructureInfo:
    """Information extracted from a structure file."""

    path: Path
    format: str  # "pdb" or "cif"
    chains: list[str]
    sequences: dict[str, str]  # chain_id -> sequence
    resolution: Optional[float] = None
    method: Optional[str] = None
    title: Optional[str] = None


class PDBParser:
    """Parser for PDB and mmCIF structure files."""

    # Standard amino acid codes
    AA_3TO1 = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        "SEC": "U", "PYL": "O",  # Selenocysteine, Pyrrolysine
    }

    # Nucleotide codes
    NUC_3TO1 = {
        "DA": "A", "DC": "C", "DG": "G", "DT": "T",  # DNA
        "A": "A", "C": "C", "G": "G", "U": "U",      # RNA
    }

    def parse(self, path: Union[str, Path]) -> StructureInfo:
        """
        Parse a structure file and extract information.

        Args:
            path: Path to PDB or mmCIF file.

        Returns:
            StructureInfo object with extracted data.

        Raises:
            InputValidationError: If parsing fails.
        """
        path = Path(path)
        if not path.exists():
            raise InputValidationError(f"Structure file not found: {path}", "path")

        # Determine format from extension
        suffix = path.suffix.lower()
        if suffix in (".pdb", ".ent"):
            return self._parse_pdb(path)
        elif suffix in (".cif", ".mmcif"):
            return self._parse_mmcif(path)
        else:
            # Try to detect from content
            content = path.read_text()[:1000]
            if content.startswith("data_"):
                return self._parse_mmcif(path)
            return self._parse_pdb(path)

    def _parse_pdb(self, path: Path) -> StructureInfo:
        """Parse PDB format file."""
        chains = set()
        sequences = {}
        resolution = None
        method = None
        title = None

        current_chain = None
        residue_list = []

        with open(path) as f:
            for line in f:
                record = line[:6].strip()

                if record == "TITLE":
                    title = (title or "") + line[10:].strip()

                elif record == "EXPDTA":
                    method = line[10:].strip()

                elif record == "REMARK" and line[7:10] == "  2":
                    # Resolution
                    try:
                        res_str = line[23:30].strip()
                        if res_str and res_str != "NOT":
                            resolution = float(res_str)
                    except (ValueError, IndexError):
                        pass

                elif record == "ATOM" or record == "HETATM":
                    chain_id = line[21]
                    chains.add(chain_id)

                    if chain_id != current_chain:
                        if current_chain is not None and residue_list:
                            sequences[current_chain] = self._residues_to_sequence(residue_list)
                        current_chain = chain_id
                        residue_list = []

                    # Extract residue info
                    res_name = line[17:20].strip()
                    res_num = line[22:26].strip()

                    # Only add if it's a new residue
                    if not residue_list or residue_list[-1][1] != res_num:
                        residue_list.append((res_name, res_num))

        # Save last chain
        if current_chain is not None and residue_list:
            sequences[current_chain] = self._residues_to_sequence(residue_list)

        return StructureInfo(
            path=path,
            format="pdb",
            chains=sorted(chains),
            sequences=sequences,
            resolution=resolution,
            method=method,
            title=title,
        )

    def _parse_mmcif(self, path: Path) -> StructureInfo:
        """Parse mmCIF format file."""
        # Simple mmCIF parser - for full support consider using BioPython
        chains = set()
        sequences = {}
        resolution = None
        method = None
        title = None

        chain_residues = {}

        with open(path) as f:
            content = f.read()

        # Extract title
        if "_struct.title" in content:
            for line in content.split("\n"):
                if line.startswith("_struct.title"):
                    title = line.split(None, 1)[1].strip().strip("'\"")
                    break

        # Parse atom sites
        in_atom_site = False
        atom_site_columns = []

        for line in content.split("\n"):
            if line.startswith("_atom_site."):
                in_atom_site = True
                col_name = line.split(".")[1].strip()
                atom_site_columns.append(col_name)
            elif in_atom_site and line.startswith("ATOM") or line.startswith("HETATM"):
                parts = line.split()
                if len(parts) >= len(atom_site_columns):
                    # Find column indices
                    try:
                        chain_idx = atom_site_columns.index("auth_asym_id")
                        res_name_idx = atom_site_columns.index("auth_comp_id")
                        res_num_idx = atom_site_columns.index("auth_seq_id")

                        chain_id = parts[chain_idx]
                        res_name = parts[res_name_idx]
                        res_num = parts[res_num_idx]

                        chains.add(chain_id)

                        if chain_id not in chain_residues:
                            chain_residues[chain_id] = []

                        if not chain_residues[chain_id] or chain_residues[chain_id][-1][1] != res_num:
                            chain_residues[chain_id].append((res_name, res_num))
                    except (ValueError, IndexError):
                        continue
            elif in_atom_site and line.startswith("#"):
                in_atom_site = False
                atom_site_columns = []

        # Convert residues to sequences
        for chain_id, residues in chain_residues.items():
            sequences[chain_id] = self._residues_to_sequence(residues)

        return StructureInfo(
            path=path,
            format="cif",
            chains=sorted(chains),
            sequences=sequences,
            resolution=resolution,
            method=method,
            title=title,
        )

    def _residues_to_sequence(self, residues: list[tuple[str, str]]) -> str:
        """Convert list of residue names to one-letter sequence."""
        sequence = []
        for res_name, _ in residues:
            if res_name in self.AA_3TO1:
                sequence.append(self.AA_3TO1[res_name])
            elif res_name in self.NUC_3TO1:
                sequence.append(self.NUC_3TO1[res_name])
            else:
                sequence.append("X")  # Unknown
        return "".join(sequence)

    def extract_sequences(self, path: Union[str, Path]) -> list[Sequence]:
        """
        Extract sequences from a structure file.

        Args:
            path: Path to structure file.

        Returns:
            List of Sequence objects, one per chain.
        """
        info = self.parse(path)
        sequences = []

        for chain_id in info.chains:
            if chain_id in info.sequences:
                seq_str = info.sequences[chain_id]

                # Detect molecule type
                if all(c in "ACGT" for c in seq_str):
                    mol_type = MoleculeType.DNA
                elif all(c in "ACGU" for c in seq_str):
                    mol_type = MoleculeType.RNA
                else:
                    mol_type = MoleculeType.PROTEIN

                sequences.append(
                    Sequence(
                        id=f"{path.stem}_{chain_id}",
                        sequence=seq_str,
                        molecule_type=mol_type,
                        description=f"Chain {chain_id} from {path.name}",
                    )
                )

        return sequences

    def to_template(
        self,
        path: Union[str, Path],
        chain_id: str = "A",
        sequence_identity: Optional[float] = None,
    ) -> Template:
        """
        Create a Template object from a structure file.

        Args:
            path: Path to structure file.
            chain_id: Chain to use as template.
            sequence_identity: Optional sequence identity to query.

        Returns:
            Template object.
        """
        path = Path(path)
        info = self.parse(path)

        if chain_id not in info.chains:
            raise InputValidationError(
                f"Chain {chain_id} not found in {path}. Available: {info.chains}",
                "chain_id",
            )

        return Template(
            id=f"{path.stem}_{chain_id}",
            path=path,
            chain_id=chain_id,
            sequence_identity=sequence_identity,
        )
