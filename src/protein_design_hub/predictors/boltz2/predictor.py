"""Boltz-2 predictor implementation with comprehensive parameters."""

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
    """
    Boltz-2 predictor for biomolecular structure prediction.

    Boltz-2 supports:
    - Proteins, DNA, RNA, and small molecules
    - Multi-chain complexes
    - Affinity prediction
    - Multiple diffusion samples
    - MSA server integration

    Parameters are configured via settings.predictors.boltz2
    """

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
        """Run Boltz-2 prediction with all available parameters."""
        config = self.settings.predictors.boltz2

        # Prepare input file in Boltz YAML format
        input_yaml = output_dir / "input.yaml"
        self._write_boltz_input(input_data, input_yaml)

        # Try to use boltz CLI
        try:
            return self._run_via_cli(input_data, input_yaml, output_dir, config)
        except Exception as e:
            raise PredictionError(self.name, str(e), original_error=e)

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

        boltz_config = {
            "version": 1,
            "sequences": entities,
        }

        with open(yaml_path, "w") as f:
            yaml.dump(boltz_config, f, default_flow_style=False)

    def _run_via_cli(
        self,
        input_data: PredictionInput,
        input_yaml: Path,
        output_dir: Path,
        config,
    ) -> PredictionResult:
        """Run prediction via boltz CLI with all parameters."""
        cmd = [
            "boltz", "predict",
            str(input_yaml),
            "--out_dir", str(output_dir),
            # Model selection
            "--model", config.model,
            # Core prediction parameters
            "--recycling_steps", str(config.recycling_steps),
            "--sampling_steps", str(config.sampling_steps),
            "--diffusion_samples", str(config.diffusion_samples),
            # Device settings
            "--devices", str(config.devices),
            "--accelerator", config.accelerator,
            "--num_workers", str(config.num_workers),
            # Output format
            "--output_format", config.output_format,
        ]

        # Optional parameters
        if config.max_parallel_samples is not None:
            cmd.extend(["--max_parallel_samples", str(config.max_parallel_samples)])

        if config.step_scale is not None:
            cmd.extend(["--step_scale", str(config.step_scale)])

        if config.seed is not None:
            cmd.extend(["--seed", str(config.seed)])

        # MSA settings
        if config.use_msa_server:
            cmd.append("--use_msa_server")
            if config.msa_server_url:
                cmd.extend(["--msa_server_url", config.msa_server_url])
            cmd.extend(["--msa_pairing_strategy", config.msa_pairing_strategy])

        cmd.extend(["--max_msa_seqs", str(config.max_msa_seqs)])

        if config.subsample_msa:
            cmd.append("--subsample_msa")
            cmd.extend(["--num_subsampled_msa", str(config.num_subsampled_msa)])

        # Potentials
        if config.use_potentials:
            cmd.append("--use_potentials")

        # Affinity settings
        if config.affinity_mw_correction:
            cmd.append("--affinity_mw_correction")
            cmd.extend(["--sampling_steps_affinity", str(config.sampling_steps_affinity)])
            cmd.extend(["--diffusion_samples_affinity", str(config.diffusion_samples_affinity)])

        # Output options
        if config.write_full_pae:
            cmd.append("--write_full_pae")

        if config.write_full_pde:
            cmd.append("--write_full_pde")

        if config.write_embeddings:
            cmd.append("--write_embeddings")

        # Performance options
        cmd.extend(["--preprocessing-threads", str(config.preprocessing_threads)])

        if config.no_kernels:
            cmd.append("--no_kernels")

        # Checkpoints
        if config.checkpoint:
            cmd.extend(["--checkpoint", str(config.checkpoint)])

        if config.affinity_checkpoint:
            cmd.extend(["--affinity_checkpoint", str(config.affinity_checkpoint)])

        # Override
        if config.override:
            cmd.append("--override")

        # Run command
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
            metadata={
                "model": config.model,
                "recycling_steps": config.recycling_steps,
                "sampling_steps": config.sampling_steps,
                "diffusion_samples": config.diffusion_samples,
                "accelerator": config.accelerator,
            },
        )

    def _parse_results(self, output_dir: Path) -> tuple[List[Path], List[StructureScore]]:
        """Parse Boltz output files."""
        structure_paths = []
        scores = []

        # Find output CIF/PDB files - Boltz outputs to predictions/ subdirectory
        predictions_dir = output_dir / "predictions"
        if predictions_dir.exists():
            search_dir = predictions_dir
        else:
            search_dir = output_dir

        # Look for structure files
        for pattern in ["**/*.cif", "**/*.pdb"]:
            for struct_file in sorted(search_dir.glob(pattern)):
                if struct_file not in structure_paths:
                    structure_paths.append(struct_file)

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

        # Check for summary JSON
        for summary_file in search_dir.glob("**/summary.json"):
            if scores:
                break
            try:
                with open(summary_file) as f:
                    summary = json.load(f)
                if "confidence" in summary:
                    scores.append(StructureScore(
                        plddt=summary.get("plddt"),
                        ptm=summary.get("ptm"),
                        confidence=summary.get("confidence"),
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

        # Check CLI
        try:
            result = subprocess.run(
                ["boltz", "--help"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                checks.append("CLI available")
        except Exception:
            checks.append("CLI not available")

        # Check CUDA
        cuda_ok, cuda_msg = self._installer.verify_cuda()
        checks.append(cuda_msg)

        return True, "; ".join(checks)

    def get_available_parameters(self) -> dict:
        """Get all available parameters with descriptions."""
        return {
            "model": {
                "type": "str",
                "default": "boltz2",
                "options": ["boltz1", "boltz2"],
                "description": "Model version to use",
            },
            "recycling_steps": {
                "type": "int",
                "default": 3,
                "range": [1, 20],
                "description": "Number of recycling steps",
            },
            "sampling_steps": {
                "type": "int",
                "default": 200,
                "range": [50, 1000],
                "description": "Number of sampling/diffusion steps",
            },
            "diffusion_samples": {
                "type": "int",
                "default": 1,
                "range": [1, 20],
                "description": "Number of diffusion samples to generate",
            },
            "step_scale": {
                "type": "float",
                "default": None,
                "description": "Step scale for diffusion (1.638 for Boltz-1, 1.5 for Boltz-2)",
            },
            "use_msa_server": {
                "type": "bool",
                "default": False,
                "description": "Use MMSeqs2 server for MSA generation",
            },
            "max_msa_seqs": {
                "type": "int",
                "default": 8192,
                "description": "Maximum number of MSA sequences",
            },
            "use_potentials": {
                "type": "bool",
                "default": False,
                "description": "Use potentials for structure steering",
            },
            "output_format": {
                "type": "str",
                "default": "mmcif",
                "options": ["pdb", "mmcif"],
                "description": "Output structure format",
            },
            "write_full_pae": {
                "type": "bool",
                "default": True,
                "description": "Write full PAE matrix to NPZ file",
            },
            "accelerator": {
                "type": "str",
                "default": "gpu",
                "options": ["gpu", "cpu", "tpu"],
                "description": "Accelerator type to use",
            },
            "seed": {
                "type": "int",
                "default": None,
                "description": "Random seed for reproducibility",
            },
        }
