"""Coarse residue-residue contact energy (Miyazawa–Jernigan, simplified)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.utils import (
    iter_protein_residues,
    load_structure_biopython,
    residue_one_letter,
    residue_representative_atom,
)


class ContactEnergyMetric(BaseMetric):
    """
    Compute a simple contact potential between residues using an MJ-style matrix.

    Implementation:
    - Use Cβ atoms for contacts (Cα for Gly).
    - Contact if distance <= contact_cutoff Å.
    - Skip pairs with |i-j| <= min_seq_separation within the same chain.

    Returns a *relative* energy; useful for ranking designs, not absolute ΔG.
    """

    name = "contact_energy"
    description = "Coarse contact energy (MJ-style, Cβ contacts)"
    requires_reference = False

    def __init__(self, contact_cutoff: float = 8.0, min_seq_separation: int = 1):
        self.contact_cutoff = float(contact_cutoff)
        self.min_seq_separation = int(min_seq_separation)

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

        residues = []
        for residue in iter_protein_residues(structure):
            aa = residue_one_letter(residue)
            if aa is None:
                continue
            atom = residue_representative_atom(residue, ["CB", "CA"])
            if atom is None:
                continue
            chain = residue.get_parent()
            chain_id = getattr(chain, "id", "?")
            resseq = residue.get_id()[1]
            residues.append((aa, chain_id, resseq, atom))

        if len(residues) < 2:
            raise EvaluationError(
                self.name, "Not enough residues with coordinates to compute contacts"
            )

        energy = 0.0
        contact_count = 0

        from Bio.PDB.NeighborSearch import NeighborSearch

        atoms = [r[3] for r in residues]
        ns = NeighborSearch(atoms)
        atom_to_res_idx = {id(atom): i for i, atom in enumerate(atoms)}

        for i, (aa_i, chain_i, res_i, atom_i) in enumerate(residues):
            for atom_j in ns.search(atom_i.coord, self.contact_cutoff, level="A"):
                j = atom_to_res_idx[id(atom_j)]
                if j <= i:
                    continue

                aa_j, chain_j, res_j, _ = residues[j]

                if chain_i == chain_j and abs(res_i - res_j) <= self.min_seq_separation:
                    continue

                contact_count += 1
                energy += MJ_CONTACT_ENERGY[aa_i][aa_j]

        n_res = len(residues)
        energy_per_res = energy / n_res if n_res else None

        return {
            "contact_energy": energy,
            "contact_energy_per_residue": energy_per_res,
            "contact_count": contact_count,
            "num_residues": n_res,
            "contact_cutoff": self.contact_cutoff,
            "min_seq_separation": self.min_seq_separation,
        }


# Symmetric contact matrix (arbitrary energy units).
# Values derived from a Miyazawa–Jernigan-like ordering (coarse-grained); used for ranking only.
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"

_MJ_TRI = [
    # A
    [-0.20],
    # C
    [-0.30, -0.40],
    # D
    [0.10, 0.00, 0.30],
    # E
    [0.10, 0.00, 0.20, 0.30],
    # F
    [-0.60, -0.70, -0.20, -0.20, -1.10],
    # G
    [-0.10, -0.20, 0.10, 0.10, -0.50, -0.10],
    # H
    [-0.10, -0.20, 0.00, 0.00, -0.60, -0.10, -0.30],
    # I
    [-0.70, -0.80, -0.30, -0.30, -1.20, -0.60, -0.70, -1.00],
    # K
    [0.20, 0.10, 0.10, 0.10, -0.20, 0.10, 0.00, -0.30, 0.30],
    # L
    [-0.70, -0.80, -0.30, -0.30, -1.20, -0.60, -0.70, -1.00, -0.30, -1.00],
    # M
    [-0.50, -0.60, -0.20, -0.20, -1.00, -0.40, -0.50, -0.80, -0.20, -0.80, -0.70],
    # N
    [0.00, -0.10, 0.10, 0.10, -0.30, 0.00, -0.10, -0.40, 0.10, -0.40, -0.30, 0.20],
    # P
    [0.10, 0.00, 0.20, 0.20, -0.20, 0.10, 0.00, -0.30, 0.10, -0.30, -0.20, 0.10, 0.10],
    # Q
    [0.00, -0.10, 0.10, 0.10, -0.30, 0.00, -0.10, -0.40, 0.10, -0.40, -0.30, 0.20, 0.20, 0.10],
    # R
    [0.20, 0.10, 0.10, 0.10, -0.20, 0.10, 0.00, -0.30, 0.30, -0.30, -0.20, 0.10, 0.10, 0.10, 0.30],
    # S
    [
        0.00,
        -0.10,
        0.10,
        0.10,
        -0.30,
        0.00,
        -0.10,
        -0.40,
        0.10,
        -0.40,
        -0.30,
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
    ],
    # T
    [
        -0.10,
        -0.20,
        0.00,
        0.00,
        -0.50,
        -0.10,
        -0.20,
        -0.60,
        0.00,
        -0.60,
        -0.50,
        0.00,
        0.00,
        0.00,
        0.00,
        0.00,
        -0.10,
    ],
    # V
    [
        -0.60,
        -0.70,
        -0.20,
        -0.20,
        -1.10,
        -0.50,
        -0.60,
        -0.90,
        -0.20,
        -0.90,
        -0.80,
        -0.30,
        -0.20,
        -0.30,
        -0.20,
        -0.20,
        -0.30,
        -0.60,
    ],
    # W
    [
        -0.70,
        -0.80,
        -0.20,
        -0.20,
        -1.30,
        -0.60,
        -0.70,
        -1.10,
        -0.20,
        -1.10,
        -1.00,
        -0.30,
        -0.20,
        -0.30,
        -0.20,
        -0.20,
        -0.30,
        -0.60,
        -1.20,
    ],
    # Y
    [
        -0.50,
        -0.60,
        -0.10,
        -0.10,
        -1.10,
        -0.40,
        -0.50,
        -0.90,
        -0.10,
        -0.90,
        -0.80,
        -0.20,
        -0.10,
        -0.20,
        -0.10,
        -0.10,
        -0.20,
        -0.50,
        -1.00,
        -1.00,
    ],
]


def _build_symmetric(tri, order: str):
    mat = {a: {} for a in order}
    for i, a in enumerate(order):
        for j in range(i + 1):
            b = order[j]
            v = float(tri[i][j])
            mat[a][b] = v
            mat[b][a] = v
    return mat


MJ_CONTACT_ENERGY = _build_symmetric(_MJ_TRI, _AA_ORDER)
