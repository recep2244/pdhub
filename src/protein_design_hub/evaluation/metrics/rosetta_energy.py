"""Optional Rosetta energy metric via PyRosetta (if installed)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric

_PYROSETTA_INITIALIZED = False


class RosettaEnergyMetric(BaseMetric):
    """
    Compute Rosetta all-atom energy using PyRosetta.

    This is optional because PyRosetta is large and requires separate installation/licensing.
    """

    name = "rosetta_energy"
    description = "Rosetta ref2015 total score (optional; requires PyRosetta)"
    requires_reference = False

    def is_available(self) -> bool:
        try:
            import pyrosetta  # noqa: F401

            return True
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "PyRosetta (licensed install; see https://www.pyrosetta.org/)"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")

        if not self.is_available():
            raise EvaluationError(self.name, "PyRosetta not available")

        try:
            import pyrosetta
            from pyrosetta import rosetta
        except Exception as e:
            raise EvaluationError(self.name, "Failed to import PyRosetta", original_error=e)

        global _PYROSETTA_INITIALIZED
        if not _PYROSETTA_INITIALIZED:
            # Keep PyRosetta quiet by default.
            pyrosetta.init(options="-mute all")
            _PYROSETTA_INITIALIZED = True

        pose = rosetta.core.import_pose.pose_from_file(str(model_path))
        scorefxn = rosetta.core.scoring.get_score_function()

        total = float(scorefxn(pose))
        energies = pose.energies().total_energies()

        # Return a small, stable subset of terms (users can inspect full metadata).
        term_names = ["fa_atr", "fa_rep", "fa_sol", "hbond_sr_bb", "hbond_lr_bb", "hbond_sc"]
        terms = {}
        for name in term_names:
            try:
                score_type = getattr(rosetta.core.scoring, name)
                terms[name] = float(energies[score_type])
            except Exception:
                continue

        return {
            "rosetta_total_score": total,
            "terms": terms,
        }
