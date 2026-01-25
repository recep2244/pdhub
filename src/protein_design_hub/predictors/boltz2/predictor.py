"""Boltz-2 predictor implementation."""

import json
import subprocess
from pathlib import Path
from typing import List

from protein_design_hub.predictors.base import BasePredictor
from protein_design_hub.predictors.registry import PredictorRegistry
from protein_design_hub.predictors.boltz2.installer import Boltz2Installer
from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    PredictorType,
    StructureScore,
    MoleculeType,
)
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.exceptions import PredictionError


@PredictorRegistry.register("boltz2")
class Boltz2Predictor(BasePredictor):
    """Boltz-2 predictor for biomolecular structure prediction."""

    name = "boltz2"
    predictor_type = PredictorType.BOLTZ2
    description = "Latest biomolecular structure prediction from MIT"
    supports_multimer = True
    supports_templates = False
    supports_msa = True
    output_format = "cif"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer = Boltz2Installer()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        """Run Boltz-2 prediction."""
        config = self.settings.predictors.boltz2

        # Prepare input file in Boltz YAML format
        input_yaml = output_dir / "input.yaml"
        self._write_boltz_input(input_data, input_yaml)

        # Try to use boltz CLI first
        try:
            return self._run_via_cli(input_data, input_yaml, output_dir, config)
        except Exception as cli_error:
            # Fall back to Python API
            try:
                return self._run_via_api(input_data, output_dir, config)
            except Exception as api_error:
                raise PredictionError(
                    self.name,
                    f"CLI error: {cli_error}; API error: {api_error}",
                )

    def _write_boltz_input(self, input_data: PredictionInput, yaml_path: Path) -> None:
        """Write Boltz input YAML file."""
        import yaml

        entities = []
        for seq in input_data.sequences:
            entity = {"id": seq.id}

            if seq.molecule_type == MoleculeType.PROTEIN:
                entity["type"] = "protein"
                entity["sequence"] = seq.sequence
            elif seq.molecule_type == MoleculeType.DNA:
                entity["type"] = "dna"
                entity["sequence"] = seq.sequence
            elif seq.molecule_type == MoleculeType.RNA:
                entity["type"] = "rna"
                entity["sequence"] = seq.sequence
            elif seq.molecule_type == MoleculeType.LIGAND:
                entity["type"] = "ligand"
                entity["smiles"] = seq.sequence  # For ligands, sequence is SMILES
            else:
                entity["type"] = "protein"
                entity["sequence"] = seq.sequence

            entities.append(entity)

        config = {
            "version": 1,
            "sequences": entities,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    def _run_via_cli(
        self,
        input_data: PredictionInput,
        input_yaml: Path,
        output_dir: Path,
        config,
    ) -> PredictionResult:
        """Run prediction via boltz CLI."""
        cmd = [
            "boltz", "predict",
            str(input_yaml),
            "--out_dir", str(output_dir),
            "--recycling_steps", str(config.recycling_steps),
            "--sampling_steps", str(config.sampling_steps),
            "--diffusion_samples", str(config.diffusion_samples),
        ]

        if self._check_gpu():
            cmd.extend(["--accelerator", "gpu"])
        else:
            cmd.extend(["--accelerator", "cpu"])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600 * 24,
        )

        if result.returncode != 0:
            raise PredictionError(self.name, f"Boltz CLI failed: {result.stderr}")

        structure_paths, scores = self._parse_results(output_dir)

        return PredictionResult(
            job_id=input_data.job_id,
            predictor=self.predictor_type,
            structure_paths=structure_paths,
            scores=scores,
            runtime_seconds=0,
            success=True,
        )

    def _run_via_api(
        self,
        input_data: PredictionInput,
        output_dir: Path,
        config,
    ) -> PredictionResult:
        """Run prediction via Boltz Python API."""
        try:
            from boltz.main import predict
            from boltz.data.types import Manifest
        except ImportError:
            raise PredictionError(self.name, "Cannot import boltz API")

        # Build sequences for API
        sequences = []
        for seq in input_data.sequences:
            if seq.molecule_type == MoleculeType.PROTEIN:
                sequences.append({"protein": seq.sequence})
            elif seq.molecule_type == MoleculeType.DNA:
                sequences.append({"dna": seq.sequence})
            elif seq.molecule_type == MoleculeType.RNA:
                sequences.append({"rna": seq.sequence})
            elif seq.molecule_type == MoleculeType.LIGAND:
                sequences.append({"ligand": seq.sequence})
            else:
                sequences.append({"protein": seq.sequence})

        # Run prediction
        predict(
            data=sequences,
            out_dir=output_dir,
            recycling_steps=config.recycling_steps,
            sampling_steps=config.sampling_steps,
            diffusion_samples=config.diffusion_samples,
            accelerator="gpu" if self._check_gpu() else "cpu",
        )

        structure_paths, scores = self._parse_results(output_dir)

        return PredictionResult(
            job_id=input_data.job_id,
            predictor=self.predictor_type,
            structure_paths=structure_paths,
            scores=scores,
            runtime_seconds=0,
            success=True,
        )

    def _parse_results(self, output_dir: Path) -> tuple[List[Path], List[StructureScore]]:
        """Parse Boltz output files."""
        structure_paths = []
        scores = []

        # Find output CIF files - Boltz outputs to predictions/ subdirectory
        predictions_dir = output_dir / "predictions"
        if predictions_dir.exists():
            search_dir = predictions_dir
        else:
            search_dir = output_dir

        for cif_file in sorted(search_dir.glob("**/*.cif")):
            structure_paths.append(cif_file)

        # Parse confidence scores
        for json_file in search_dir.glob("**/confidence*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                scores.append(StructureScore(
                    plddt=data.get("plddt"),
                    ptm=data.get("ptm"),
                    confidence=data.get("confidence") or data.get("plddt"),
                ))
            except Exception:
                continue

        # Also check for scores in ranking file
        ranking_file = search_dir / "ranking.json"
        if ranking_file.exists() and not scores:
            try:
                with open(ranking_file) as f:
                    ranking = json.load(f)
                for model_data in ranking.get("models", []):
                    scores.append(StructureScore(
                        plddt=model_data.get("plddt"),
                        ptm=model_data.get("ptm"),
                        confidence=model_data.get("confidence"),
                    ))
            except Exception:
                pass

        # Ensure we have scores for all structures
        while len(scores) < len(structure_paths):
            scores.append(StructureScore())

        return structure_paths, scores

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def verify_installation(self) -> tuple[bool, str]:
        """Verify Boltz-2 installation."""
        checks = []

        # Check import
        try:
            import boltz
            checks.append(f"boltz v{getattr(boltz, '__version__', 'unknown')} imported")
        except ImportError as e:
            return False, f"Cannot import boltz: {e}"

        # Check CUDA
        cuda_ok, cuda_msg = self._installer.verify_cuda()
        checks.append(cuda_msg)

        # Check model
        model_ok, model_msg = self._installer.verify_model_weights()
        checks.append(model_msg)

        return True, "; ".join(checks)
