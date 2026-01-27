"""Steric clash score metric (MolProbity-like, simplified)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.utils import all_heavy_atoms, load_structure_biopython


class ClashScoreMetric(BaseMetric):
    """
    Simplified clash score: number of heavy-atom pairs closer than cutoff.

    Notes:
    - This is *not* a full MolProbity implementation (no atom radii reduction,
      no explicit bonding graph). It’s meant as a fast, dependency-light proxy.
    """

    name = "clash_score"
    description = "Steric clash score (heavy-atom contacts < cutoff Å, normalized per 1000 atoms)"
    requires_reference = False

    def __init__(self, cutoff_angstrom: float = 2.0):
        self.cutoff_angstrom = float(cutoff_angstrom)

    def is_available(self) -> bool:
        try:
            import Bio  # noqa: F401

            return True
        except Exception:
            return False

    def get_requirements(self) -> str:
        return "BioPython (pip install biopython)"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        model_path = Path(model_path)
        if not model_path.exists():
            raise EvaluationError(self.name, f"Model not found: {model_path}")

        structure = load_structure_biopython(model_path, structure_id="model")
        atoms = all_heavy_atoms(structure)
        if len(atoms) < 2:
            raise EvaluationError(self.name, "Not enough atoms for clash computation")

        clash_pairs = _count_clashes(atoms, cutoff=self.cutoff_angstrom)

        num_atoms = len(atoms)
        clash_score = clash_pairs / (num_atoms / 1000.0) if num_atoms else None

        return {
            "clash_score": clash_score,
            "clash_count": clash_pairs,
            "num_atoms": num_atoms,
            "cutoff_angstrom": self.cutoff_angstrom,
        }


def _atom_residue_key(atom) -> Tuple[str, Tuple, str]:
    """Return a stable key for residue identity."""
    parent = atom.get_parent()
    chain = parent.get_parent()
    chain_id = getattr(chain, "id", "?")
    res_id = parent.get_id()  # tuple like (' ', 12, ' ')
    return (chain_id, res_id, parent.get_resname())


def _count_clashes(atoms, cutoff: float) -> int:
    """Count unique atom-atom clashes below cutoff using BioPython NeighborSearch."""
    from Bio.PDB.NeighborSearch import NeighborSearch

    ns = NeighborSearch(atoms)
    atom_index = {id(a): i for i, a in enumerate(atoms)}

    count = 0
    seen = set()

    for atom in atoms:
        i = atom_index[id(atom)]
        for other in ns.search(atom.coord, cutoff, level="A"):
            j = atom_index[id(other)]
            if j <= i:
                continue

            # Skip atoms in the same residue (bonded/nearby by construction).
            if _atom_residue_key(atom) == _atom_residue_key(other):
                continue

            pair = (i, j)
            if pair in seen:
                continue
            seen.add(pair)
            count += 1

    return count
