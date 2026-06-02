"""FoldX interface binding-energy metric (AnalyseComplex).

Reports the inter-chain interaction energy of a complex — the quantitative
binding signal that was missing for antibody/binder work (BuildModel only gives
monomer stability). Requires the FoldX binary and a multi-chain structure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric


class FoldXInterfaceMetric(BaseMetric):
    """Inter-chain interaction (binding) energy via FoldX AnalyseComplex."""

    name = "foldx_interface_energy"
    description = "FoldX AnalyseComplex inter-chain interaction energy (kcal/mol; lower = tighter binding)"
    requires_reference = False

    def __init__(self, chains: Optional[str] = None, repair: bool = True):
        self.chains = chains      # e.g. "A,B"; None → all chain pairs
        self.repair = repair

    def is_available(self) -> bool:
        try:
            from protein_design_hub.energy.paths import find_foldx_executable
            return find_foldx_executable() is not None
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "FoldX binary (set FOLDX_BIN or add to PATH)"

    def compute(self, model_path: Path, reference_path: Optional[Path] = None, **kwargs) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")
        from protein_design_hub.energy.foldx import run_foldx_analyse_complex
        out_dir = model_path.parent / f"_foldx_ac_{model_path.stem}"
        res = run_foldx_analyse_complex(
            model_path, out_dir, chains=self.chains, repair=self.repair,
        )
        return {
            "foldx_interface_energy_kcal_mol": res.get("foldx_interaction_energy_kcal_mol"),
        }
