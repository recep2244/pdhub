"""Structure file writer and converter."""

import shutil
from pathlib import Path
from typing import Optional

from protein_design_hub.core.types import PredictionResult
from protein_design_hub.core.exceptions import ProteinDesignHubError


class StructureWriter:
    """Writer for structure files with format conversion support."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize structure writer.

        Args:
            output_dir: Default output directory for structures.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")

    def copy_structure(
        self,
        source: Path,
        dest: Optional[Path] = None,
        rename: Optional[str] = None,
    ) -> Path:
        """
        Copy a structure file to the output directory.

        Args:
            source: Source structure file path.
            dest: Destination path (uses output_dir if not specified).
            rename: New filename (keeps original if not specified).

        Returns:
            Path to the copied file.
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if dest is None:
            dest = self.output_dir / "structures"
        else:
            dest = Path(dest)

        dest.mkdir(parents=True, exist_ok=True)

        if rename:
            dest_file = dest / rename
        else:
            dest_file = dest / source.name

        shutil.copy2(source, dest_file)
        return dest_file

    def organize_prediction_results(
        self,
        result: PredictionResult,
        output_dir: Optional[Path] = None,
    ) -> list[Path]:
        """
        Organize prediction result structures into a standard directory structure.

        Args:
            result: PredictionResult with structure paths.
            output_dir: Output directory (uses default if not specified).

        Returns:
            List of paths to organized structure files.
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        predictor_dir = output_dir / result.predictor.value / "structures"
        predictor_dir.mkdir(parents=True, exist_ok=True)

        organized_paths = []
        for i, struct_path in enumerate(result.structure_paths):
            # Determine output filename
            suffix = struct_path.suffix
            new_name = f"model_{i:02d}{suffix}"

            dest_path = predictor_dir / new_name
            shutil.copy2(struct_path, dest_path)
            organized_paths.append(dest_path)

        return organized_paths

    def convert_pdb_to_cif(self, pdb_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Convert PDB file to mmCIF format using BioPython.

        Args:
            pdb_path: Path to PDB file.
            output_path: Output path for CIF file.

        Returns:
            Path to the converted CIF file.
        """
        try:
            from Bio.PDB import PDBParser, MMCIFIO
        except ImportError:
            raise ProteinDesignHubError(
                "BioPython required for PDB to CIF conversion. "
                "Install with: pip install biopython"
            )

        pdb_path = Path(pdb_path)
        if output_path is None:
            output_path = pdb_path.with_suffix(".cif")
        else:
            output_path = Path(output_path)

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("structure", str(pdb_path))

        io = MMCIFIO()
        io.set_structure(structure)
        io.save(str(output_path))

        return output_path

    def convert_cif_to_pdb(self, cif_path: Path, output_path: Optional[Path] = None) -> Path:
        """
        Convert mmCIF file to PDB format using BioPython.

        Args:
            cif_path: Path to mmCIF file.
            output_path: Output path for PDB file.

        Returns:
            Path to the converted PDB file.
        """
        try:
            from Bio.PDB import MMCIFParser, PDBIO
        except ImportError:
            raise ProteinDesignHubError(
                "BioPython required for CIF to PDB conversion. "
                "Install with: pip install biopython"
            )

        cif_path = Path(cif_path)
        if output_path is None:
            output_path = cif_path.with_suffix(".pdb")
        else:
            output_path = Path(output_path)

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", str(cif_path))

        io = PDBIO()
        io.set_structure(structure)
        io.save(str(output_path))

        return output_path

    def add_bfactors(
        self,
        structure_path: Path,
        bfactors: list[float],
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Add B-factors (e.g., pLDDT scores) to a structure file.

        Args:
            structure_path: Path to structure file.
            bfactors: List of B-factor values per residue.
            output_path: Output path (overwrites input if not specified).

        Returns:
            Path to the modified structure file.
        """
        try:
            from Bio.PDB import PDBParser, PDBIO, MMCIFParser, MMCIFIO
        except ImportError:
            raise ProteinDesignHubError(
                "BioPython required for B-factor modification. "
                "Install with: pip install biopython"
            )

        structure_path = Path(structure_path)
        suffix = structure_path.suffix.lower()

        if suffix in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
            io = MMCIFIO()
        else:
            parser = PDBParser(QUIET=True)
            io = PDBIO()

        structure = parser.get_structure("structure", str(structure_path))

        # Apply B-factors
        residue_idx = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue_idx < len(bfactors):
                        for atom in residue:
                            atom.set_bfactor(bfactors[residue_idx])
                        residue_idx += 1

        # Save
        if output_path is None:
            output_path = structure_path
        else:
            output_path = Path(output_path)

        io.set_structure(structure)
        io.save(str(output_path))

        return output_path
