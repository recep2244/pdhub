"""QS-score (Quaternary Structure score) metric using OpenStructure."""

from pathlib import Path
from typing import Dict, Any, Optional

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.core.exceptions import EvaluationError


class QSScoreMetric(BaseMetric):
    """
    QS-score computation using OpenStructure.

    QS-score measures the quality of quaternary structure (oligomeric assembly)
    by comparing interface residue contacts between model and reference.
    """

    name = "qs_score"
    description = "Quaternary Structure Score"
    requires_reference = True

    def __init__(self, contact_threshold: float = 12.0):
        """
        Initialize QS-score metric.

        Args:
            contact_threshold: Distance threshold for defining contacts.
        """
        self.contact_threshold = contact_threshold
        self._use_runner = False

    def is_available(self) -> bool:
        """Check if OpenStructure is available (direct or via runner)."""
        # Try direct import first
        try:
            import ost
            from ost.mol.alg import qsscore
            return True
        except ImportError:
            pass

        # Try micromamba runner
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            if runner.is_available():
                self._use_runner = True
                return True
        except Exception:
            pass

        return False

    def get_requirements(self) -> str:
        return "OpenStructure (install via: micromamba create -n ost -c conda-forge -c bioconda openstructure)"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute QS-score.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with QS-score values.
        """
        if reference_path is None:
            raise EvaluationError(self.name, "Reference structure required for QS-score")

        model_path = Path(model_path)
        reference_path = Path(reference_path)

        # Try using the micromamba runner first (if available)
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner
            runner = get_ost_runner()
            if runner.is_available():
                result = runner.compute_qs_score(model_path, reference_path)
                if "error" in result:
                    raise EvaluationError(self.name, result["error"])
                return result
        except ImportError:
            pass
        except RuntimeError as e:
            raise EvaluationError(self.name, str(e))

        # Fall back to direct import
        try:
            import ost
            from ost.io import LoadPDB, LoadMMCIF
            from ost.mol.alg import qsscore
        except ImportError:
            raise EvaluationError(
                self.name,
                "OpenStructure not available. Install with: micromamba create -n ost -c conda-forge -c bioconda openstructure"
            )

        try:
            # Load structures
            model = self._load_structure(model_path)
            reference = self._load_structure(reference_path)

            # Get chain mappings
            model_chains = [ch.GetName() for ch in model.chains]
            ref_chains = [ch.GetName() for ch in reference.chains]

            # QS-score computation
            qs_scorer = qsscore.QSScorer(
                reference.Select("peptide=true"),
                model.Select("peptide=true"),
            )

            # Get global QS-score
            global_qs = qs_scorer.global_score

            # Get per-interface scores if available
            interface_scores = {}
            if hasattr(qs_scorer, "interface_scores"):
                interface_scores = qs_scorer.interface_scores

            return {
                "qs_score": global_qs,
                "qs_best": qs_scorer.best_score if hasattr(qs_scorer, "best_score") else global_qs,
                "model_chains": model_chains,
                "reference_chains": ref_chains,
                "interface_scores": interface_scores,
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

    def compute_interface_quality(
        self,
        model_path: Path,
        reference_path: Path,
    ) -> Dict[str, Any]:
        """
        Compute detailed interface quality metrics.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure.

        Returns:
            Dictionary with detailed interface metrics.
        """
        try:
            import ost
            from ost.io import LoadPDB, LoadMMCIF
            from ost.mol.alg import qsscore
        except ImportError:
            raise EvaluationError(self.name, "OpenStructure not available")

        model = self._load_structure(model_path)
        reference = self._load_structure(reference_path)

        qs_scorer = qsscore.QSScorer(
            reference.Select("peptide=true"),
            model.Select("peptide=true"),
        )

        result = {
            "global_qs": qs_scorer.global_score,
            "chain_mapping": {},
            "interface_details": [],
        }

        # Extract chain mapping if available
        if hasattr(qs_scorer, "chain_mapping"):
            result["chain_mapping"] = qs_scorer.chain_mapping

        return result
