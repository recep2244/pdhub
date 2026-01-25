"""lDDT (local Distance Difference Test) metric using OpenStructure."""

from pathlib import Path
from typing import Dict, Any, Optional

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.core.exceptions import EvaluationError


class LDDTMetric(BaseMetric):
    """
    lDDT score computation using OpenStructure.

    lDDT (local Distance Difference Test) measures the fraction of
    inter-atomic distances that are preserved between model and reference.
    """

    name = "lddt"
    description = "Local Distance Difference Test"
    requires_reference = True

    def __init__(
        self,
        inclusion_radius: float = 15.0,
        sequence_separation: int = 0,
        stereochemistry_check: bool = True,
    ):
        """
        Initialize lDDT metric.

        Args:
            inclusion_radius: Radius for including atoms in lDDT calculation.
            sequence_separation: Minimum sequence separation for distance pairs.
            stereochemistry_check: Whether to check stereochemistry.
        """
        self.inclusion_radius = inclusion_radius
        self.sequence_separation = sequence_separation
        self.stereochemistry_check = stereochemistry_check

    def is_available(self) -> bool:
        """Check if OpenStructure is available."""
        try:
            import ost
            return True
        except ImportError:
            return False

    def get_requirements(self) -> str:
        return "OpenStructure (install via conda: conda install -c bioconda openstructure)"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute lDDT score.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with lDDT scores.
        """
        if reference_path is None:
            raise EvaluationError(self.name, "Reference structure required for lDDT")

        try:
            import ost
            from ost.io import LoadPDB, LoadMMCIF
            from ost.mol.alg import lDDTScorer
        except ImportError:
            raise EvaluationError(
                self.name,
                "OpenStructure not available. Install with: conda install -c bioconda openstructure"
            )

        model_path = Path(model_path)
        reference_path = Path(reference_path)

        try:
            # Load structures
            model = self._load_structure(model_path)
            reference = self._load_structure(reference_path)

            # Select protein atoms only
            model_sel = model.Select("peptide=true and aname=CA,C,N,O")
            ref_sel = reference.Select("peptide=true and aname=CA,C,N,O")

            if model_sel.GetAtomCount() == 0:
                raise EvaluationError(self.name, "No protein atoms found in model")
            if ref_sel.GetAtomCount() == 0:
                raise EvaluationError(self.name, "No protein atoms found in reference")

            # Create lDDT scorer
            scorer = lDDTScorer(
                ref_sel,
                inclusion_radius=self.inclusion_radius,
                sequence_separation=self.sequence_separation,
            )

            # Compute lDDT
            global_lddt = scorer.lDDT(
                model_sel,
                local_lddt_prop="lddt",
            )

            # Extract per-residue scores
            per_residue = []
            for residue in model_sel.residues:
                lddt_val = residue.GetFloatProp("lddt", -1.0)
                if lddt_val >= 0:
                    per_residue.append(lddt_val)

            return {
                "lddt": global_lddt,
                "lddt_per_residue": per_residue,
                "num_residues": len(per_residue),
            }

        except Exception as e:
            if isinstance(e, EvaluationError):
                raise
            raise EvaluationError(self.name, str(e), original_error=e)

    def _load_structure(self, path: Path):
        """Load structure from PDB or mmCIF file."""
        from ost.io import LoadPDB, LoadMMCIF

        suffix = path.suffix.lower()
        if suffix in (".cif", ".mmcif"):
            return LoadMMCIF(str(path))
        else:
            return LoadPDB(str(path))

    def compute_self_lddt(self, model_path: Path) -> Dict[str, Any]:
        """
        Compute self-consistency lDDT (using model as its own reference).

        This can be useful for detecting internal inconsistencies.

        Args:
            model_path: Path to model structure.

        Returns:
            Dictionary with self-lDDT scores.
        """
        return self.compute(model_path, model_path)
