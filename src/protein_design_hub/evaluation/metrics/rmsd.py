"""RMSD (Root Mean Square Deviation) metric."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.core.exceptions import EvaluationError


class RMSDMetric(BaseMetric):
    """
    RMSD computation for structure comparison.

    Computes root mean square deviation between model and reference
    after optimal superposition.
    """

    name = "rmsd"
    description = "Root Mean Square Deviation"
    requires_reference = True

    def __init__(self, atoms: str = "CA"):
        """
        Initialize RMSD metric.

        Args:
            atoms: Atom selection for RMSD calculation.
                   Options: "CA" (alpha carbons), "backbone", "all", "heavy"
        """
        self.atoms = atoms
        self._use_runner = False

    def is_available(self) -> bool:
        """Check if computation is possible (OpenStructure runner, direct OST, or BioPython)."""
        # Check micromamba runner
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            if runner.is_available():
                self._use_runner = True
                return True
        except Exception:
            pass

        # Check direct OpenStructure
        try:
            import ost
            return True
        except ImportError:
            pass

        # Check BioPython
        try:
            import numpy as np
            from Bio.PDB import Superimposer
            return True
        except ImportError:
            pass

        return False

    def get_requirements(self) -> str:
        return "OpenStructure (via micromamba) or BioPython (pip install biopython)"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute RMSD between model and reference.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with RMSD values.
        """
        if reference_path is None:
            raise EvaluationError(self.name, "Reference structure required for RMSD")

        model_path = Path(model_path)
        reference_path = Path(reference_path)

        # Try using the micromamba runner first (if available)
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            if runner.is_available():
                result = runner.compute_rmsd(model_path, reference_path, atoms=self.atoms)
                if "error" in result:
                    raise EvaluationError(self.name, result["error"])
                return result
        except ImportError:
            pass
        except RuntimeError as e:
            raise EvaluationError(self.name, str(e))

        # Try OpenStructure direct import
        try:
            return self._compute_with_openstructure(model_path, reference_path)
        except ImportError:
            pass

        # Fall back to BioPython
        try:
            return self._compute_with_biopython(model_path, reference_path)
        except ImportError:
            raise EvaluationError(
                self.name,
                "Neither OpenStructure nor BioPython available. "
                "Install OpenStructure via micromamba or BioPython via pip."
            )

    def _compute_with_openstructure(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """Compute RMSD using OpenStructure."""
        from ost.io import LoadPDB, LoadMMCIF
        from ost.mol.alg import Superpose

        # Load structures
        if model_path.suffix.lower() in (".cif", ".mmcif"):
            model = LoadMMCIF(str(model_path))
        else:
            model = LoadPDB(str(model_path))

        if reference_path.suffix.lower() in (".cif", ".mmcif"):
            reference = LoadMMCIF(str(reference_path))
        else:
            reference = LoadPDB(str(reference_path))

        # Select atoms based on selection mode
        if self.atoms == "CA":
            model_sel = model.Select("peptide=true and aname=CA")
            ref_sel = reference.Select("peptide=true and aname=CA")
        elif self.atoms == "backbone":
            model_sel = model.Select("peptide=true and aname=CA,C,N,O")
            ref_sel = reference.Select("peptide=true and aname=CA,C,N,O")
        elif self.atoms == "heavy":
            model_sel = model.Select("peptide=true and ele!=H")
            ref_sel = reference.Select("peptide=true and ele!=H")
        else:
            model_sel = model.Select("peptide=true")
            ref_sel = reference.Select("peptide=true")

        # Superpose
        result = Superpose(model_sel, ref_sel)

        return {
            "rmsd": result.rmsd,
            "rmsd_superposed": result.rmsd,
            "num_atoms": model_sel.GetAtomCount(),
            "transformation": {
                "rotation": result.transformation.GetRot().ToList() if hasattr(result.transformation, "GetRot") else None,
                "translation": result.transformation.GetTrans().ToList() if hasattr(result.transformation, "GetTrans") else None,
            },
        }

    def _compute_with_biopython(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """Compute RMSD using BioPython."""
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
        from Bio.PDB.Polypeptide import is_aa

        # Load structures
        if model_path.suffix.lower() in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        model = parser.get_structure("model", str(model_path))

        if reference_path.suffix.lower() in (".cif", ".mmcif"):
            ref_parser = MMCIFParser(QUIET=True)
        else:
            ref_parser = PDBParser(QUIET=True)

        reference = ref_parser.get_structure("ref", str(reference_path))

        # Extract atoms based on selection
        model_atoms = self._extract_atoms(model)
        ref_atoms = self._extract_atoms(reference)

        # Align to minimum length
        min_len = min(len(model_atoms), len(ref_atoms))
        if min_len < 3:
            raise EvaluationError(self.name, "Not enough atoms for superposition")

        model_atoms = model_atoms[:min_len]
        ref_atoms = ref_atoms[:min_len]

        # Superimpose
        sup = Superimposer()
        sup.set_atoms(ref_atoms, model_atoms)

        # Apply transformation
        sup.apply([a for a in model.get_atoms()])

        return {
            "rmsd": sup.rms,
            "rmsd_superposed": sup.rms,
            "num_atoms": min_len,
            "transformation": {
                "rotation": sup.rotran[0].tolist() if hasattr(sup, "rotran") else None,
                "translation": sup.rotran[1].tolist() if hasattr(sup, "rotran") else None,
            },
        }

    def _extract_atoms(self, structure) -> List:
        """Extract atoms based on selection mode."""
        from Bio.PDB.Polypeptide import is_aa

        atoms = []
        for chain in structure.get_chains():
            for residue in chain:
                if not is_aa(residue):
                    continue

                if self.atoms == "CA":
                    if "CA" in residue:
                        atoms.append(residue["CA"])
                elif self.atoms == "backbone":
                    for atom_name in ["CA", "C", "N", "O"]:
                        if atom_name in residue:
                            atoms.append(residue[atom_name])
                elif self.atoms == "heavy":
                    for atom in residue:
                        if atom.element != "H":
                            atoms.append(atom)
                else:
                    for atom in residue:
                        atoms.append(atom)

        return atoms

    def compute_per_residue(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """
        Compute per-residue RMSD after global superposition.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with global and per-residue RMSD.
        """
        try:
            from Bio.PDB import PDBParser, MMCIFParser, Superimposer
            from Bio.PDB.Polypeptide import is_aa
        except ImportError:
            raise EvaluationError(self.name, "BioPython required for per-residue RMSD")

        # Load structures
        if model_path.suffix.lower() in (".cif", ".mmcif"):
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        model = parser.get_structure("model", str(model_path))

        if reference_path.suffix.lower() in (".cif", ".mmcif"):
            ref_parser = MMCIFParser(QUIET=True)
        else:
            ref_parser = PDBParser(QUIET=True)

        reference = ref_parser.get_structure("ref", str(reference_path))

        # Get CA atoms
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

        min_len = min(len(model_ca), len(ref_ca))
        model_ca = model_ca[:min_len]
        ref_ca = ref_ca[:min_len]

        # Superimpose
        sup = Superimposer()
        sup.set_atoms(ref_ca, model_ca)
        sup.apply([a for a in model.get_atoms()])

        # Compute per-residue RMSD
        per_residue = []
        for m_ca, r_ca in zip(model_ca, ref_ca):
            diff = m_ca.coord - r_ca.coord
            rmsd = np.sqrt(np.sum(diff ** 2))
            per_residue.append(rmsd)

        return {
            "rmsd": sup.rms,
            "rmsd_per_residue": per_residue,
            "num_residues": min_len,
        }
