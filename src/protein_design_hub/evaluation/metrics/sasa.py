"""Solvent accessible surface area (SASA) metric via BioPython."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.utils import load_structure_biopython


class SASAMetric(BaseMetric):
    """
    Total SASA computed with the Shrake-Rupley algorithm.

    Useful as a proxy for burial/exposure changes during design and for spotting
    overly exposed hydrophobics.
    """

    name = "sasa"
    description = "Total solvent accessible surface area (BioPython Shrake-Rupley)"
    requires_reference = False

    def __init__(self, probe_radius: float = 1.4, n_points: int = 960):
        self.probe_radius = float(probe_radius)
        self.n_points = int(n_points)

    def is_available(self) -> bool:
        try:
            from Bio.PDB.SASA import ShrakeRupley  # noqa: F401

            return True
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "BioPython>=1.79 (pip install biopython)"

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
            raise EvaluationError(self.name, "BioPython SASA module not available")

        from Bio.PDB.SASA import ShrakeRupley

        structure = load_structure_biopython(model_path, structure_id="model")
        sr = ShrakeRupley(probe_radius=self.probe_radius, n_points=self.n_points)
        sr.compute(structure, level="A")  # annotate atoms with .sasa

        total = 0.0
        atom_count = 0
        for atom in structure.get_atoms():
            sasa = getattr(atom, "sasa", None)
            if sasa is None:
                continue
            total += float(sasa)
            atom_count += 1

        return {
            "sasa_total": total,
            "num_atoms": atom_count,
            "probe_radius": self.probe_radius,
            "n_points": self.n_points,
        }
