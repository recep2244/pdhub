"""TM-score metric using TMalign."""

import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import re

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.core.exceptions import EvaluationError


class TMScoreMetric(BaseMetric):
    """
    TM-score computation using TMalign.

    TM-score is a metric for measuring structural similarity between proteins,
    normalized by the length of the target protein.
    """

    name = "tm_score"
    description = "Template Modeling Score via TMalign"
    requires_reference = True

    def __init__(self, tmalign_path: Optional[Path] = None):
        """
        Initialize TM-score metric.

        Args:
            tmalign_path: Path to TMalign executable. Auto-detected if not provided.
        """
        self._tmalign_path = tmalign_path

    @property
    def tmalign_path(self) -> Optional[Path]:
        """Get path to TMalign executable."""
        if self._tmalign_path and self._tmalign_path.exists():
            return self._tmalign_path

        # Try to find TMalign in PATH
        which_result = shutil.which("TMalign")
        if which_result:
            return Path(which_result)

        # Try common locations
        common_paths = [
            Path("/usr/local/bin/TMalign"),
            Path("/usr/bin/TMalign"),
            Path.home() / "bin" / "TMalign",
        ]
        for path in common_paths:
            if path.exists():
                return path

        return None

    def is_available(self) -> bool:
        """Check if TMalign is available."""
        return self.tmalign_path is not None

    def get_requirements(self) -> str:
        return "TMalign (download from https://zhanggroup.org/TM-align/)"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute TM-score using TMalign.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with TM-score and related metrics.
        """
        if reference_path is None:
            raise EvaluationError(self.name, "Reference structure required for TM-score")

        if not self.is_available():
            # Fall back to BioPython-based alignment
            return self._compute_with_biopython(model_path, reference_path)

        model_path = Path(model_path)
        reference_path = Path(reference_path)

        # Convert to PDB if necessary (TMalign prefers PDB)
        model_pdb = self._ensure_pdb(model_path)
        ref_pdb = self._ensure_pdb(reference_path)

        try:
            result = subprocess.run(
                [str(self.tmalign_path), str(model_pdb), str(ref_pdb)],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                raise EvaluationError(self.name, f"TMalign failed: {result.stderr}")

            return self._parse_tmalign_output(result.stdout)

        except subprocess.TimeoutExpired:
            raise EvaluationError(self.name, "TMalign timed out")
        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(self.name, str(e), original_error=e)

    def _ensure_pdb(self, path: Path) -> Path:
        """Ensure structure is in PDB format."""
        if path.suffix.lower() in (".pdb", ".ent"):
            return path

        # Convert CIF to PDB
        try:
            from Bio.PDB import MMCIFParser, PDBIO
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("struct", str(path))

            pdb_path = path.with_suffix(".pdb")
            io = PDBIO()
            io.set_structure(structure)
            io.save(str(pdb_path))
            return pdb_path
        except ImportError:
            # Try OpenStructure
            try:
                from ost.io import LoadMMCIF, SavePDB
                struct = LoadMMCIF(str(path))
                pdb_path = path.with_suffix(".pdb")
                SavePDB(struct, str(pdb_path))
                return pdb_path
            except ImportError:
                raise EvaluationError(
                    self.name,
                    "Cannot convert CIF to PDB. Install BioPython or OpenStructure."
                )

    def _parse_tmalign_output(self, output: str) -> Dict[str, Any]:
        """Parse TMalign output to extract metrics."""
        result = {
            "tm_score": None,
            "tm_score_chain1": None,
            "tm_score_chain2": None,
            "rmsd": None,
            "aligned_length": None,
            "sequence_identity": None,
            "gdt_ts": None,
            "gdt_ha": None,
        }

        for line in output.split("\n"):
            # TM-score normalized by length of Chain_1
            if "TM-score=" in line and "Chain_1" in line:
                match = re.search(r"TM-score=\s*([\d.]+)", line)
                if match:
                    result["tm_score_chain1"] = float(match.group(1))

            # TM-score normalized by length of Chain_2
            elif "TM-score=" in line and "Chain_2" in line:
                match = re.search(r"TM-score=\s*([\d.]+)", line)
                if match:
                    result["tm_score_chain2"] = float(match.group(1))

            # RMSD
            elif "RMSD=" in line:
                match = re.search(r"RMSD=\s*([\d.]+)", line)
                if match:
                    result["rmsd"] = float(match.group(1))

            # Aligned length
            elif "Aligned length=" in line:
                match = re.search(r"Aligned length=\s*(\d+)", line)
                if match:
                    result["aligned_length"] = int(match.group(1))

            # Sequence identity
            elif "Seq_ID=" in line:
                match = re.search(r"Seq_ID=n_identical/n_aligned=\s*([\d.]+)", line)
                if match:
                    result["sequence_identity"] = float(match.group(1))

            # GDT-TS
            elif "GDT-TS-score=" in line:
                match = re.search(r"GDT-TS-score=\s*([\d.]+)", line)
                if match:
                    result["gdt_ts"] = float(match.group(1))

            # GDT-HA
            elif "GDT-HA-score=" in line:
                match = re.search(r"GDT-HA-score=\s*([\d.]+)", line)
                if match:
                    result["gdt_ha"] = float(match.group(1))

        # Use average TM-score as the main score
        if result["tm_score_chain1"] and result["tm_score_chain2"]:
            result["tm_score"] = (result["tm_score_chain1"] + result["tm_score_chain2"]) / 2
        elif result["tm_score_chain1"]:
            result["tm_score"] = result["tm_score_chain1"]
        elif result["tm_score_chain2"]:
            result["tm_score"] = result["tm_score_chain2"]

        return result

    def _compute_with_biopython(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """Fallback computation using BioPython."""
        try:
            from Bio.PDB import PDBParser, MMCIFParser, Superimposer
            from Bio.PDB.Polypeptide import is_aa
            import numpy as np
        except ImportError:
            raise EvaluationError(
                self.name,
                "TMalign not found and BioPython not available. "
                "Install TMalign or BioPython."
            )

        # Load structures
        if model_path.suffix.lower() in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        model = parser.get_structure("model", str(model_path))
        reference = parser.get_structure("ref", str(reference_path))

        # Extract CA atoms
        model_ca = []
        ref_ca = []

        for chain in model.get_chains():
            for residue in chain:
                if is_aa(residue) and "CA" in residue:
                    model_ca.append(residue["CA"])

        for chain in reference.get_chains():
            for residue in chain:
                if is_aa(residue) and "CA" in residue:
                    ref_ca.append(residue["CA"])

        # Align (using min length)
        min_len = min(len(model_ca), len(ref_ca))
        model_ca = model_ca[:min_len]
        ref_ca = ref_ca[:min_len]

        if min_len < 3:
            raise EvaluationError(self.name, "Not enough aligned residues")

        # Superimpose
        sup = Superimposer()
        sup.set_atoms(ref_ca, model_ca)
        rmsd = sup.rms

        # Approximate TM-score calculation
        d0 = 1.24 * np.cbrt(min_len - 15) - 1.8
        d0 = max(d0, 0.5)

        distances = []
        for m_atom, r_atom in zip(model_ca, ref_ca):
            d = m_atom - r_atom
            distances.append(np.sqrt(sum(d * d)))

        tm_score = sum(1 / (1 + (d / d0) ** 2) for d in distances) / min_len

        return {
            "tm_score": tm_score,
            "rmsd": rmsd,
            "aligned_length": min_len,
        }
