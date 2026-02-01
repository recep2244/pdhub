"""CAD-score metric via Voronota."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List

from protein_design_hub.evaluation.base import BaseMetric


class VoronotaCADScoreMetric(BaseMetric):
    """Compute CAD-score using the Voronota CAD-score script."""

    name = "cad_score"
    description = "CAD-score (Voronota)"
    requires_reference = True

    def __init__(self, binary: Optional[str] = None):
        self.binary = binary or "voronota-cadscore"

    def is_available(self) -> bool:
        return shutil.which(self.binary) is not None

    def get_requirements(self) -> str:
        return "Install Voronota and ensure `voronota-cadscore` is on PATH."

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if reference_path is None:
            raise ValueError("reference_path is required for CAD-score")

        model_path = Path(model_path)
        reference_path = Path(reference_path)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            residue_scores_path = Path(tmp.name)

        cmd = [
            self.binary,
            "--input-target",
            str(reference_path),
            "--input-model",
            str(model_path),
            "--output-residue-scores",
            str(residue_scores_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "voronota-cadscore failed"
            )

        cad_score = _parse_cad_score_output(result.stdout)
        residue_scores = _parse_residue_scores(residue_scores_path)

        return {
            "cad_score": cad_score,
            "cad_score_per_residue": residue_scores,
        }


def _parse_cad_score_output(text: str) -> Optional[float]:
    # Standard output format:
    # {target} {model} {query} {num_res} {global_score} ...
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                return float(parts[4])
            except ValueError:
                pass
        match = re.search(r"CAD-score:?\s*([0-9]*\.?[0-9]+)", line)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
    return None


def _parse_residue_scores(path: Path) -> List[float]:
    scores: List[float] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    scores.append(float(parts[-1]))
                except ValueError:
                    continue
    except Exception:
        return []
    return scores
