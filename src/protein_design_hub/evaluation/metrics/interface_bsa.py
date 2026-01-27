"""Interface buried surface area (BSA) metric via SASA."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.sasa import SASAMetric
from protein_design_hub.evaluation.metrics.utils import load_structure_biopython


class InterfaceBSAMetric(BaseMetric):
    """
    Buried surface area (BSA) proxy for multichain complexes:
      BSA_total = sum(SASA(chain_i alone)) - SASA(complex)

    Useful for ranking binders and complex stability.
    """

    name = "interface_bsa"
    description = "Buried surface area proxy (sum chain SASA - complex SASA)"
    requires_reference = False

    def __init__(self, probe_radius: float = 1.4, n_points: int = 960):
        self._sasa = SASAMetric(probe_radius=probe_radius, n_points=n_points)

    def is_available(self) -> bool:
        return self._sasa.is_available()

    def get_requirements(self) -> str:
        return self._sasa.get_requirements()

    def compute(
        self, model_path: Path, reference_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")

        structure = load_structure_biopython(model_path, structure_id="model")
        chains = list(structure.get_chains())
        if len(chains) < 2:
            raise EvaluationError(self.name, "Interface BSA requires at least 2 chains")

        complex_sasa = float(self._sasa.compute(model_path)["sasa_total"])

        sum_chain = 0.0
        chain_sasa = {}
        for chain in chains:
            chain_id = getattr(chain, "id", "?")
            chain_only_path = _write_chain_to_temp_pdb(structure, chain_id)
            try:
                sasa_i = float(self._sasa.compute(chain_only_path)["sasa_total"])
            finally:
                try:
                    chain_only_path.unlink(missing_ok=True)
                except Exception:
                    pass
            chain_sasa[chain_id] = sasa_i
            sum_chain += sasa_i

        bsa = sum_chain - complex_sasa

        return {
            "interface_bsa_total": bsa,
            "complex_sasa": complex_sasa,
            "sum_chain_sasa": sum_chain,
            "chain_sasa": chain_sasa,
        }


def _write_chain_to_temp_pdb(structure, chain_id: str) -> Path:
    from Bio.PDB import PDBIO, Select

    class _ChainSelect(Select):
        def accept_chain(self, chain) -> bool:
            return getattr(chain, "id", None) == chain_id

    io = PDBIO()
    io.set_structure(structure)
    tmp = NamedTemporaryFile(suffix=f"_{chain_id}.pdb", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    io.save(str(tmp_path), select=_ChainSelect())
    return tmp_path
