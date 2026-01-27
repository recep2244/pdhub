"""Rosetta CLI score_jd2 metric (total_score)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.energy.paths import find_rosetta_executable
from protein_design_hub.energy.rosetta import run_score_jd2
from protein_design_hub.evaluation.base import BaseMetric


class RosettaScoreJd2Metric(BaseMetric):
    name = "rosetta_score_jd2"
    description = "Rosetta total_score via score_jd2 (CLI)"
    requires_reference = False

    def is_available(self) -> bool:
        return find_rosetta_executable("score_jd2") is not None

    def get_requirements(self) -> str:
        return "Rosetta3 binaries (set ROSETTA3_HOME) with score_jd2.* executable"

    def compute(
        self, model_path: Path, reference_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")

        out_dir = Path(kwargs.get("work_dir") or model_path.parent / ".pdhub_rosetta_score")
        scores = run_score_jd2(model_path, out_dir)
        return {"rosetta_total_score": scores.get("total_score")}
