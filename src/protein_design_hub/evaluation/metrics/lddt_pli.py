"""LDDT-PLI metric for protein-ligand interface evaluation."""

from pathlib import Path
from typing import Dict, Any, Optional, List

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.core.exceptions import EvaluationError


class LDDTPLIMetric(BaseMetric):
    """
    LDDT-PLI (Protein-Ligand Interface) metric.

    Evaluates the quality of protein-ligand interfaces by measuring:
    - LDDT-PLI: Local distance difference test for protein-ligand contacts
    - LDDT-LP: Ligand-pocket lDDT
    - BiSyRMSD: Binding site symmetry-corrected RMSD

    This metric is essential for evaluating:
    - Drug-target binding predictions
    - Enzyme-substrate complexes
    - Protein-cofactor interactions
    """

    name = "lddt_pli"
    description = "Protein-Ligand Interface lDDT"
    requires_reference = True

    def __init__(
        self,
        binding_site_radius: float = 10.0,
        include_rmsd: bool = True,
    ):
        """
        Initialize LDDT-PLI metric.

        Args:
            binding_site_radius: Radius (Å) to define binding site around ligand.
            include_rmsd: Whether to also compute ligand RMSD.
        """
        super().__init__()
        self.binding_site_radius = binding_site_radius
        self.include_rmsd = include_rmsd

    def is_available(self) -> bool:
        """Check if OpenStructure is available."""
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            return runner.is_available()
        except Exception:
            return False

    def get_requirements(self) -> str:
        """Get installation requirements."""
        return "OpenStructure with ligand support. Install: micromamba create -n ost -c conda-forge -c bioconda openstructure"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        model_ligands: Optional[List[Path]] = None,
        reference_ligands: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        """
        Compute LDDT-PLI scores.

        Args:
            model_path: Path to model structure (with ligands).
            reference_path: Path to reference structure (with ligands).
            model_ligands: Optional separate ligand files for model.
            reference_ligands: Optional separate ligand files for reference.

        Returns:
            Dictionary with LDDT-PLI scores and components.
        """
        if reference_path is None:
            raise EvaluationError(self.name, "Reference structure required for LDDT-PLI")

        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()

            # First try the full LDDT-PLI computation
            try:
                result = runner.compute_lddt_pli(
                    model_path,
                    reference_path,
                    model_ligands,
                    reference_ligands,
                )

                if result.get("error"):
                    # Fall back to binding site analysis
                    result = runner.compute_binding_site_similarity(
                        model_path,
                        reference_path,
                        radius=self.binding_site_radius,
                    )
            except Exception:
                # Fall back to binding site analysis
                result = runner.compute_binding_site_similarity(
                    model_path,
                    reference_path,
                    radius=self.binding_site_radius,
                )

            return result

        except Exception as e:
            raise EvaluationError(self.name, str(e))

    def interpret_score(self, lddt_pli: float) -> str:
        """
        Interpret LDDT-PLI score.

        Args:
            lddt_pli: LDDT-PLI score (0-1).

        Returns:
            Human-readable interpretation.
        """
        if lddt_pli >= 0.8:
            return "Excellent - High-quality protein-ligand interface"
        elif lddt_pli >= 0.6:
            return "Good - Reasonable interface prediction"
        elif lddt_pli >= 0.4:
            return "Moderate - Some interface features captured"
        else:
            return "Poor - Significant interface deviations"
