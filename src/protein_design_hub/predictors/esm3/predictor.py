"""ESM3 predictor (local + Forge)."""

from __future__ import annotations

import json
import os
import subprocess
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
from protein_design_hub.predictors.esm3.installer import ESM3Installer
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


@PredictorRegistry.register("esm3")
class ESM3Predictor(BasePredictor):
    """ESM3 predictor for structure generation from sequence."""

    name = "esm3"
    predictor_type = PredictorType.ESM3
    description = "ESM3 - Multimodal generation (structure track)"
    supports_multimer = False
    supports_msa = False
    output_format = "pdb"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer: ToolInstaller = ESM3Installer()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        if input_data.is_multimer:
            raise PredictionError(self.name, "ESM3 predictor supports single-sequence inputs only")

        if not input_data.sequences:
            raise PredictionError(self.name, "No sequences provided")

        sequence = input_data.sequences[0].sequence.upper().strip()
        valid_aa = set("ACDEFGHIKLMNPQRSTVWYX")
        sequence = "".join(c for c in sequence if c in valid_aa)
        if not sequence:
            raise PredictionError(self.name, "No valid amino acids in sequence")

        model_name = os.getenv("ESM3_MODEL", "esm3-open")
        num_steps_raw = os.getenv("ESM3_NUM_STEPS", "8")
        try:
            num_steps = max(1, int(num_steps_raw))
        except ValueError as e:
            raise PredictionError(self.name, f"Invalid ESM3_NUM_STEPS: {num_steps_raw}") from e

        esm3_python = os.getenv("ESM3_PYTHON")
        forge_token = os.getenv("ESM3_FORGE_TOKEN")
        if esm3_python:
            runner = Path(__file__).resolve().parents[4] / "scripts" / "run_esm3.py"
            if not runner.exists():
                raise PredictionError(self.name, f"ESM3 runner not found: {runner}")

            payload = {
                "sequence": sequence,
                "model_name": model_name,
                "num_steps": num_steps,
                "forge_token": forge_token,
                "device": os.getenv("ESM3_DEVICE"),
            }
            input_json = output_dir / f"{input_data.job_id}_{self.name}.json"
            output_json = output_dir / f"{input_data.job_id}_{self.name}.out.json"
            input_json.write_text(json.dumps(payload))

            structure_path = output_dir / f"{input_data.job_id}_{self.name}.pdb"
            result = subprocess.run(
                [
                    esm3_python,
                    str(runner),
                    "--input_json",
                    str(input_json),
                    "--output_pdb",
                    str(structure_path),
                    "--output_json",
                    str(output_json),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise PredictionError(
                    self.name,
                    f"ESM3 runner failed: {result.stderr.strip() or result.stdout.strip()}",
                )

            output = json.loads(output_json.read_text())
            plddt_values = output.get("plddt") or _extract_plddt_from_pdb(structure_path)
            mean_plddt = output.get("mean_plddt")
            if mean_plddt is None and plddt_values:
                mean_plddt = sum(plddt_values) / len(plddt_values)

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
                runtime_seconds=0,
                success=True,
                metadata={
                    "esm3_model": model_name,
                    "esm3_source": "external",
                    "esm3_num_steps": num_steps,
                },
            )

        try:
            import esm
            from esm.sdk.api import ESMProtein, GenerationConfig
        except Exception as e:
            raise PredictionError(
                self.name,
                "ESM3 requires the EvolutionaryScale `esm` package. "
                "Install via `pdhub install predictor esm3`.",
                original_error=e,
            )

        if forge_token:
            try:
                model = esm.sdk.client(model_name, token=forge_token)
            except Exception as e:
                raise PredictionError(
                    self.name,
                    "Failed to initialize ESM3 Forge client. Check ESM3_FORGE_TOKEN and model name.",
                    original_error=e,
                )
            source = "forge"
        else:
            try:
                from esm.models.esm3 import ESM3
            except Exception as e:
                raise PredictionError(
                    self.name,
                    "ESM3 local model not available. Ensure the EvolutionaryScale `esm` package is installed.",
                    original_error=e,
                )

            try:
                import torch
            except Exception as e:
                raise PredictionError(
                    self.name,
                    "ESM3 local inference requires `torch`.",
                    original_error=e,
                )

            device = os.getenv("ESM3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            try:
                model = ESM3.from_pretrained(model_name).to(device)
            except Exception as e:
                raise PredictionError(
                    self.name,
                    "Failed to load ESM3 model. Ensure model weights are available (HF login may be required).",
                    original_error=e,
                )
            source = "local"

        protein = ESMProtein(sequence=sequence)
        gen_config = GenerationConfig(track="structure", num_steps=num_steps)

        try:
            protein = model.generate(protein, gen_config)
        except Exception as e:
            raise PredictionError(self.name, "ESM3 structure generation failed", original_error=e)

        structure_path = output_dir / f"{input_data.job_id}_{self.name}.pdb"
        try:
            protein.to_pdb(str(structure_path))
        except Exception as e:
            raise PredictionError(self.name, "Failed to write ESM3 PDB output", original_error=e)

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
            metadata={
                "esm3_model": model_name,
                "esm3_source": source,
                "esm3_num_steps": num_steps,
            },
        )
