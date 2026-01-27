"""ESMFold predictors (local + API)."""

from __future__ import annotations

from pathlib import Path
from typing import List

from protein_design_hub.core.exceptions import PredictionError
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    PredictorType,
    StructureScore,
)
from protein_design_hub.predictors.base import BasePredictor
from protein_design_hub.predictors.esmfold.installer import ESMFoldAPIInstaller, ESMFoldInstaller
from protein_design_hub.predictors.registry import PredictorRegistry


def _extract_plddt_from_pdb(pdb_path: Path) -> List[float]:
    """Extract pLDDT values from the B-factor field (CA atoms) in a PDB."""
    plddt_values: List[float] = []
    seen_residues: set[tuple[str, int]] = set()

    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue

        try:
            residue_id = (line[21], int(line[22:26]))
        except Exception:
            continue
        if residue_id in seen_residues:
            continue
        seen_residues.add(residue_id)

        try:
            plddt_values.append(float(line[60:66]))
        except Exception:
            continue

    return plddt_values


@PredictorRegistry.register("esmfold")
class ESMFoldPredictor(BasePredictor):
    """ESMFold predictor for fast single-sequence structure prediction."""

    name = "esmfold"
    predictor_type = PredictorType.ESMFOLD
    description = "ESMFold - Fast single-sequence structure prediction"
    supports_multimer = False
    supports_msa = False
    output_format = "pdb"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer: ToolInstaller = ESMFoldInstaller()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        try:
            import torch
            import esm
        except Exception as e:
            raise PredictionError(
                self.name,
                "ESMFold requires `fair-esm` and `torch`. Install `fair-esm` via `pdhub install predictor esmfold`, and install torch separately.",
                original_error=e,
            )

        if input_data.is_multimer:
            raise PredictionError(self.name, "ESMFold supports single-sequence inputs only")

        if not input_data.sequences:
            raise PredictionError(self.name, "No sequences provided")

        sequence = input_data.sequences[0].sequence.upper().strip()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence if c in valid_aa)
        if not sequence:
            raise PredictionError(self.name, "No valid amino acids in sequence")

        model = esm.pretrained.esmfold_v1().eval()
        if torch.cuda.is_available():
            model = model.cuda()
            # Trade-off: lower chunk size reduces memory use.
            model.set_chunk_size(128)

        with torch.no_grad():
            pdb_text = model.infer_pdb(sequence)

        structure_path = output_dir / f"{input_data.job_id}_{self.name}.pdb"
        structure_path.write_text(pdb_text)

        plddt_values = _extract_plddt_from_pdb(structure_path)
        mean_plddt = (sum(plddt_values) / len(plddt_values)) if plddt_values else None

        return PredictionResult(
            job_id=input_data.job_id,
            predictor=self.predictor_type,
            structure_paths=[structure_path],
            scores=[
                StructureScore(
                    plddt=mean_plddt,
                    plddt_per_residue=plddt_values or None,
                    confidence=mean_plddt,
                )
            ],
            runtime_seconds=0,  # set by BasePredictor
            success=True,
        )


@PredictorRegistry.register("esmfold_api")
class ESMFoldAPIPredictor(BasePredictor):
    """ESMFold predictor using the ESM Metagenomic Atlas API."""

    name = "esmfold_api"
    predictor_type = PredictorType.ESMFOLD_API
    description = "ESMFold API - Remote structure prediction"
    supports_multimer = False
    supports_msa = False
    output_format = "pdb"

    API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer: ToolInstaller = ESMFoldAPIInstaller()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        try:
            import requests
        except Exception as e:
            raise PredictionError(self.name, "Missing dependency: requests", original_error=e)

        if input_data.is_multimer:
            raise PredictionError(self.name, "ESMFold API supports single-sequence inputs only")

        if not input_data.sequences:
            raise PredictionError(self.name, "No sequences provided")

        sequence = input_data.sequences[0].sequence.upper().strip()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence if c in valid_aa)
        if not sequence:
            raise PredictionError(self.name, "No valid amino acids in sequence")

        # Keep conservative limit; API constraints may change.
        if len(sequence) > 400:
            raise PredictionError(
                self.name,
                f"Sequence too long for API ({len(sequence)} > 400). Use local ESMFold.",
            )

        response = requests.post(
            self.API_URL,
            data=sequence,
            headers={"Content-Type": "text/plain"},
            timeout=120,
        )
        if response.status_code != 200:
            raise PredictionError(self.name, f"API error: {response.status_code} - {response.text}")

        structure_path = output_dir / f"{input_data.job_id}_{self.name}.pdb"
        structure_path.write_text(response.text)

        plddt_values = _extract_plddt_from_pdb(structure_path)
        mean_plddt = (sum(plddt_values) / len(plddt_values)) if plddt_values else None

        return PredictionResult(
            job_id=input_data.job_id,
            predictor=self.predictor_type,
            structure_paths=[structure_path],
            scores=[
                StructureScore(
                    plddt=mean_plddt,
                    plddt_per_residue=plddt_values or None,
                    confidence=mean_plddt,
                )
            ],
            runtime_seconds=0,  # set by BasePredictor
            success=True,
        )
