"""Shared helpers for evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


def load_structure_biopython(path: Path, structure_id: str = "structure"):
    """Load a PDB/mmCIF structure via BioPython."""
    from Bio.PDB import MMCIFParser, PDBParser

    suffix = path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, str(path))


def iter_protein_residues(structure) -> Iterable:
    """Yield protein residues (BioPython Residue objects)."""
    from Bio.PDB.Polypeptide import is_aa

    for chain in structure.get_chains():
        for residue in chain:
            if is_aa(residue, standard=False):
                yield residue


def residue_one_letter(residue) -> Optional[str]:
    """Map a BioPython residue to a one-letter amino-acid code; returns None if unknown."""
    resname = getattr(residue, "resname", None) or residue.get_resname()
    resname = resname.strip().upper()

    mapping = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }

    return mapping.get(resname)


def residue_representative_atom(residue, prefer: List[str]):
    """Pick the first existing atom from prefer list; returns None if missing."""
    for name in prefer:
        if name in residue:
            return residue[name]
    return None


def all_heavy_atoms(structure) -> List:
    """Return heavy atoms (non-hydrogen) for the entire structure."""
    atoms = []
    for atom in structure.get_atoms():
        element = getattr(atom, "element", None)
        if element is None:
            # BioPython sometimes infers element; fall back to atom name.
            element = atom.get_name().strip()[:1]
        if str(element).upper() == "H":
            continue
        atoms.append(atom)
    return atoms
