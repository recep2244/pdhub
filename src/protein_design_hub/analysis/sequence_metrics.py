"""Sequence-level metrics useful for protein engineering/developability."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SequenceMetrics:
    length: int
    molecular_weight: float
    aromaticity: float
    instability_index: float
    gravy: float
    isoelectric_point: float
    net_charge_ph7: float
    aa_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "length": self.length,
            "molecular_weight": self.molecular_weight,
            "aromaticity": self.aromaticity,
            "instability_index": self.instability_index,
            "gravy": self.gravy,
            "isoelectric_point": self.isoelectric_point,
            "net_charge_ph7": self.net_charge_ph7,
            "aa_counts": self.aa_counts,
        }


def compute_sequence_metrics(sequence: str) -> SequenceMetrics:
    """
    Compute common engineering metrics using BioPython ProtParam.

    Notes:
    - Net charge is an estimate at pH 7.0 (Henderson-Hasselbalch model).
    - These are heuristics; not a substitute for stability/solubility experiments.
    """
    seq = "".join(sequence.split()).upper()
    if not seq:
        raise ValueError("Empty sequence")

    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
    except Exception as e:
        raise ImportError("BioPython required: pip install biopython") from e

    pa = ProteinAnalysis(seq)
    counts = pa.count_amino_acids()

    # pI + charge model
    try:
        from Bio.SeqUtils.IsoelectricPoint import IsoelectricPoint as _IP

        ip = _IP(seq)
        pI = float(ip.pi())
        charge7 = float(ip.charge_at_pH(7.0))
    except Exception:
        pI = float(pa.isoelectric_point())
        charge7 = float(pa.charge_at_pH(7.0))

    return SequenceMetrics(
        length=len(seq),
        molecular_weight=float(pa.molecular_weight()),
        aromaticity=float(pa.aromaticity()),
        instability_index=float(pa.instability_index()),
        gravy=float(pa.gravy()),
        isoelectric_point=float(pI),
        net_charge_ph7=float(charge7),
        aa_counts={k: int(v) for k, v in counts.items()},
    )
