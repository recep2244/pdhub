"""Salt bridge count metric (simple distance-based proxy)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.utils import (
    iter_protein_residues,
    load_structure_biopython,
)


class SaltBridgeMetric(BaseMetric):
    """
    Count salt bridges between acidic and basic sidechain atoms.

    This is a heuristic:
    - Acidic: ASP (OD1/OD2), GLU (OE1/OE2)
    - Basic:  LYS (NZ), ARG (NH1/NH2/NE), HIS (ND1/NE2) [optional; treated as basic]
    """

    name = "salt_bridges"
    description = "Salt bridge count (distance-based proxy)"
    requires_reference = False

    def __init__(self, cutoff_angstrom: float = 4.0, include_histidine: bool = True):
        self.cutoff = float(cutoff_angstrom)
        self.include_histidine = bool(include_histidine)

    def is_available(self) -> bool:
        try:
            import Bio  # noqa: F401

            return True
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "BioPython (pip install biopython)"

    def compute(
        self, model_path: Path, reference_path: Optional[Path] = None, **kwargs
    ) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")

        structure = load_structure_biopython(model_path, structure_id="model")

        acidic_atoms = []  # (atom, chain_id, res_id, resname)
        basic_atoms = []

        for residue in iter_protein_residues(structure):
            resname = residue.get_resname().upper()
            chain_id = residue.get_parent().id
            res_id = residue.get_id()[1]
            if resname == "ASP":
                for an in ("OD1", "OD2"):
                    if an in residue:
                        acidic_atoms.append((residue[an], chain_id, res_id, resname))
            elif resname == "GLU":
                for an in ("OE1", "OE2"):
                    if an in residue:
                        acidic_atoms.append((residue[an], chain_id, res_id, resname))
            elif resname == "LYS":
                if "NZ" in residue:
                    basic_atoms.append((residue["NZ"], chain_id, res_id, resname))
            elif resname == "ARG":
                for an in ("NH1", "NH2", "NE"):
                    if an in residue:
                        basic_atoms.append((residue[an], chain_id, res_id, resname))
            elif resname == "HIS" and self.include_histidine:
                for an in ("ND1", "NE2"):
                    if an in residue:
                        basic_atoms.append((residue[an], chain_id, res_id, resname))

        if not acidic_atoms or not basic_atoms:
            return {
                "salt_bridge_count": 0,
                "salt_bridge_count_interchain": 0,
                "cutoff_angstrom": self.cutoff,
            }

        from Bio.PDB.NeighborSearch import NeighborSearch

        basic_only = [a[0] for a in basic_atoms]
        ns = NeighborSearch(basic_only)
        basic_index = {id(atom): i for i, atom in enumerate(basic_only)}

        pairs = set()
        inter = 0
        for acid_atom, acid_chain, acid_res, acid_name in acidic_atoms:
            for cand in ns.search(acid_atom.coord, self.cutoff, level="A"):
                j = basic_index[id(cand)]
                base_atom, base_chain, base_res, base_name = basic_atoms[j]
                # Unique by residue pair (not atom pair)
                key = (acid_chain, acid_res, base_chain, base_res)
                if key in pairs:
                    continue
                pairs.add(key)
                if acid_chain != base_chain:
                    inter += 1

        return {
            "salt_bridge_count": len(pairs),
            "salt_bridge_count_interchain": inter,
            "cutoff_angstrom": self.cutoff,
        }
