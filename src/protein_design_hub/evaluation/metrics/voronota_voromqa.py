"""VoroMQA metric via Voronota."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, List

from protein_design_hub.evaluation.base import BaseMetric


class VoronotaVoroMQAMetric(BaseMetric):
    """Compute VoroMQA score using Voronota."""

    name = "voromqa"
    description = "VoroMQA (Voronota)"
    requires_reference = False

    def __init__(self, binary: Optional[str] = None):
        self.binary = binary or _find_binary()

    def is_available(self) -> bool:
        return self.binary is not None and shutil.which(self.binary) is not None

    def get_requirements(self) -> str:
        return "Install Voronota and ensure `voronota-voromqa` is on PATH."

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if self.binary is None:
            raise RuntimeError("voronota-voromqa not available")

        model_path = Path(model_path)

        residue_scores: List[float] = []
        cmd = [self.binary, "--input", str(model_path)]

        output_residue = False
        if "only-global" not in self.binary:
            output_residue = True

        if output_residue:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                residue_scores_path = Path(tmp.name)
            cmd += ["--output-residue-scores", str(residue_scores_path)]
        else:
            residue_scores_path = None

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip() or result.stdout.strip() or "voronota-voromqa failed"
            )

        score, residue_count, atom_count = _parse_voromqa_output(result.stdout)

        if residue_scores_path is not None:
            residue_scores = _parse_residue_scores(residue_scores_path)

        return {
            "voromqa_score": score,
            "voromqa_residue_count": residue_count,
            "voromqa_atom_count": atom_count,
            "voromqa_per_residue": residue_scores,
        }


def _find_binary() -> Optional[str]:
    for name in (
        "voronota-voromqa",
        "voronota-js-voromqa",
        "voronota-js-only-global-voromqa",
    ):
        if shutil.which(name):
            return name
    return None


def _parse_voromqa_output(text: str) -> tuple[Optional[float], Optional[int], Optional[int]]:
    # Standard output format:
    # {input} {global_score} {num_residues} {num_atoms} ...
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            try:
                score = float(parts[1])
            except ValueError:
                score = None
            try:
                residues = int(parts[2])
            except ValueError:
                residues = None
            try:
                atoms = int(parts[3])
            except ValueError:
                atoms = None
            if score is not None:
                return score, residues, atoms
        match = re.search(r"VoroMQA-score:?\s*([0-9]*\.?[0-9]+)", line)
        if match:
            try:
                return float(match.group(1)), None, None
            except ValueError:
                pass
    return None, None, None


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
