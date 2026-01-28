"""Disorder prediction metric based on pLDDT scores."""

from pathlib import Path
from typing import Dict, List, Optional, Any

from protein_design_hub.evaluation.base import BaseMetric


class DisorderMetric(BaseMetric):
    """
    Predict disordered regions from pLDDT scores or structure.

    Regions with pLDDT < 50 are typically considered disordered in AlphaFold predictions.
    """

    def __init__(
        self,
        plddt_threshold: float = 50.0,
        min_region_length: int = 5,
    ):
        """
        Initialize disorder metric.

        Args:
            plddt_threshold: pLDDT threshold below which residues are considered disordered.
            min_region_length: Minimum length for a disordered region.
        """
        self.plddt_threshold = plddt_threshold
        self.min_region_length = min_region_length

    @property
    def name(self) -> str:
        return "disorder"

    @property
    def description(self) -> str:
        return "Disorder prediction from pLDDT scores"

    @property
    def requires_reference(self) -> bool:
        return False

    def is_available(self) -> bool:
        return True

    def get_requirements(self) -> str:
        return "No external requirements"

    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        plddt_values: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute disorder prediction.

        Args:
            model_path: Path to structure file.
            reference_path: Not used.
            plddt_values: Optional direct pLDDT values.

        Returns:
            Dictionary with disorder metrics.
        """
        # Get pLDDT values from structure if not provided
        if plddt_values is None:
            plddt_values = self._extract_plddt(model_path)

        if not plddt_values:
            # Fall back to sequence-based prediction
            sequence = self._extract_sequence(model_path)
            if sequence:
                return self._predict_from_sequence(sequence)
            return {"error": "Could not extract pLDDT values or sequence"}

        # Calculate disorder metrics
        per_residue_disorder = [
            1.0 if p < self.plddt_threshold else 0.0
            for p in plddt_values
        ]

        disorder_fraction = sum(per_residue_disorder) / len(per_residue_disorder)

        # Identify disordered regions
        regions = self._identify_regions(per_residue_disorder)

        # Classify disorder type
        disorder_type = self._classify_disorder(
            per_residue_disorder, plddt_values
        )

        return {
            "disorder_fraction": disorder_fraction,
            "num_disordered_residues": int(sum(per_residue_disorder)),
            "total_residues": len(plddt_values),
            "per_residue_disorder": per_residue_disorder,
            "disordered_regions": regions,
            "disorder_type": disorder_type,
            "plddt_threshold": self.plddt_threshold,
            "mean_plddt_disordered": self._mean_plddt_disordered(plddt_values),
        }

    def _extract_plddt(self, structure_path: Path) -> Optional[List[float]]:
        """Extract pLDDT from structure B-factors."""
        try:
            from Bio.PDB import PDBParser, MMCIFParser

            structure_path = Path(structure_path)

            if structure_path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            structure = parser.get_structure('struct', str(structure_path))

            plddt = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            if 'CA' in residue:
                                plddt.append(residue['CA'].get_bfactor())
                break

            return plddt if plddt else None

        except Exception:
            return None

    def _extract_sequence(self, structure_path: Path) -> Optional[str]:
        """Extract sequence from structure."""
        try:
            from Bio.PDB import PDBParser, MMCIFParser
            from Bio.SeqUtils import seq1

            structure_path = Path(structure_path)

            if structure_path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)

            structure = parser.get_structure('struct', str(structure_path))

            sequence = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ':
                            sequence.append(seq1(residue.resname))
                break

            return "".join(sequence) if sequence else None

        except Exception:
            return None

    def _predict_from_sequence(self, sequence: str) -> Dict[str, Any]:
        """Predict disorder from sequence using biophysical properties."""
        from protein_design_hub.biophysics.stability import calculate_disorder_propensity

        disorder_frac, per_residue, regions = calculate_disorder_propensity(sequence)

        return {
            "disorder_fraction": disorder_frac,
            "num_disordered_residues": sum(1 for d in per_residue if d > 0.5),
            "total_residues": len(sequence),
            "per_residue_disorder": per_residue,
            "disordered_regions": [
                {"start": s + 1, "end": e + 1}
                for s, e in regions
            ],
            "prediction_method": "sequence_based",
        }

    def _identify_regions(self, disorder_scores: List[float]) -> List[Dict]:
        """Identify contiguous disordered regions."""
        regions = []
        in_region = False
        start = 0

        for i, score in enumerate(disorder_scores):
            if score > 0.5 and not in_region:
                in_region = True
                start = i
            elif score <= 0.5 and in_region:
                in_region = False
                if i - start >= self.min_region_length:
                    regions.append({
                        "start": start + 1,
                        "end": i,
                        "length": i - start,
                    })

        if in_region and len(disorder_scores) - start >= self.min_region_length:
            regions.append({
                "start": start + 1,
                "end": len(disorder_scores),
                "length": len(disorder_scores) - start,
            })

        return regions

    def _classify_disorder(
        self, disorder_scores: List[float], plddt: List[float]
    ) -> str:
        """Classify type of disorder."""
        disorder_frac = sum(disorder_scores) / len(disorder_scores)

        if disorder_frac < 0.1:
            return "Ordered"
        elif disorder_frac < 0.3:
            return "Locally disordered"
        elif disorder_frac < 0.6:
            return "Partially disordered"
        else:
            return "Intrinsically disordered"

    def _mean_plddt_disordered(self, plddt: List[float]) -> Optional[float]:
        """Calculate mean pLDDT of disordered regions."""
        disordered = [p for p in plddt if p < self.plddt_threshold]
        return sum(disordered) / len(disordered) if disordered else None
