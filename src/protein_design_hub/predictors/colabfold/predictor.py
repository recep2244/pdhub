"""ColabFold predictor implementation."""

import json
import subprocess
from pathlib import Path
from typing import List, Optional

from protein_design_hub.predictors.base import BasePredictor
from protein_design_hub.predictors.registry import PredictorRegistry
from protein_design_hub.predictors.colabfold.installer import ColabFoldInstaller
from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    PredictorType,
    StructureScore,
)
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.exceptions import PredictionError
from protein_design_hub.io.parsers.fasta import FastaParser


@PredictorRegistry.register("colabfold")
class ColabFoldPredictor(BasePredictor):
    """ColabFold predictor using LocalColabFold installation."""

    name = "colabfold"
    predictor_type = PredictorType.COLABFOLD
    description = "AlphaFold2 with MMseqs2 for fast MSA generation"
    supports_multimer = True
    supports_templates = True
    supports_msa = True
    output_format = "pdb"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer = ColabFoldInstaller()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        """Run ColabFold prediction."""
        # Prepare input FASTA
        fasta_path = output_dir / "input.fasta"
        self._write_input_fasta(input_data, fasta_path)

        # Build command
        cmd = self._build_command(input_data, fasta_path, output_dir)

        # Run prediction
        env = self._installer.setup_environment()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=3600 * 24,  # 24 hour timeout
            )

            if result.returncode != 0:
                raise PredictionError(
                    self.name,
                    f"ColabFold failed: {result.stderr}",
                )

        except subprocess.TimeoutExpired:
            raise PredictionError(self.name, "Prediction timed out")

        # Parse results
        structure_paths, scores = self._parse_results(output_dir, input_data)

        return PredictionResult(
            job_id=input_data.job_id,
            predictor=self.predictor_type,
            structure_paths=structure_paths,
            scores=scores,
            runtime_seconds=0,  # Will be set by base class
            success=True,
        )

    def _write_input_fasta(self, input_data: PredictionInput, fasta_path: Path) -> None:
        """Write input sequences to FASTA file."""
        parser = FastaParser()

        if input_data.is_multimer:
            # For multimers, ColabFold uses : to separate chains
            combined_seq = ":".join(seq.sequence for seq in input_data.sequences)
            from protein_design_hub.core.types import Sequence
            sequences = [Sequence(
                id=input_data.job_id,
                sequence=combined_seq,
                description="multimer",
            )]
        else:
            sequences = input_data.sequences

        parser.write(sequences, fasta_path)

    def _build_command(
        self,
        input_data: PredictionInput,
        fasta_path: Path,
        output_dir: Path,
    ) -> List[str]:
        """Build the colabfold_batch command."""
        executable = self._installer.get_executable_path()
        config = self.settings.predictors.colabfold

        cmd = [
            str(executable),
            str(fasta_path),
            str(output_dir),
            "--num-models", str(input_data.num_models),
            "--num-recycle", str(input_data.num_recycles or config.num_recycles),
            "--msa-mode", config.msa_mode,
        ]

        # Add optional flags
        if config.use_amber:
            cmd.append("--amber")

        if config.use_templates or input_data.templates:
            cmd.append("--templates")

        # Add custom templates if provided
        if input_data.templates:
            template_dir = output_dir / "templates"
            template_dir.mkdir(exist_ok=True)
            for template in input_data.templates:
                # Copy template to template directory
                import shutil
                shutil.copy(template.path, template_dir)
            cmd.extend(["--custom-template-path", str(template_dir)])

        # Use pre-computed MSA if provided
        if input_data.msa:
            msa_dir = output_dir / "msa"
            msa_dir.mkdir(exist_ok=True)
            msa_path = msa_dir / f"{input_data.job_id}.a3m"
            from protein_design_hub.io.parsers.a3m import A3MParser
            A3MParser().write(input_data.msa, msa_path)
            cmd.extend(["--msa-only", "false"])

        return cmd

    def _parse_results(
        self,
        output_dir: Path,
        input_data: PredictionInput,
    ) -> tuple[List[Path], List[StructureScore]]:
        """Parse ColabFold output files."""
        structure_paths = []
        scores = []

        # Find output PDB files
        pdb_pattern = f"{input_data.job_id}*.pdb"
        if input_data.is_multimer:
            pdb_pattern = f"{input_data.job_id}*_relaxed_rank_*.pdb"

        for pdb_file in sorted(output_dir.glob("*.pdb")):
            # Skip unrelaxed if relaxed exists
            if "_unrelaxed_" in pdb_file.name:
                relaxed_name = pdb_file.name.replace("_unrelaxed_", "_relaxed_")
                if (output_dir / relaxed_name).exists():
                    continue
            structure_paths.append(pdb_file)

        # Parse scores from JSON files
        for json_file in output_dir.glob("*_scores_rank_*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    score = StructureScore(
                        plddt=data.get("plddt"),
                        ptm=data.get("ptm"),
                        iptm=data.get("iptm"),
                        pae=data.get("pae"),
                        confidence=data.get("ptm") or data.get("plddt"),
                    )
                    # Handle per-residue pLDDT
                    if "plddt" in data and isinstance(data["plddt"], list):
                        score.plddt_per_residue = data["plddt"]
                        score.plddt = sum(data["plddt"]) / len(data["plddt"])
                    scores.append(score)
            except Exception:
                continue

        # If no scores found, try to parse from ranking file
        if not scores:
            ranking_file = output_dir / f"ranking_{input_data.job_id}.json"
            if ranking_file.exists():
                try:
                    with open(ranking_file) as f:
                        ranking = json.load(f)
                    for model_name, model_scores in ranking.get("models", {}).items():
                        scores.append(StructureScore(
                            plddt=model_scores.get("plddt"),
                            ptm=model_scores.get("ptm"),
                            confidence=model_scores.get("ptm") or model_scores.get("plddt"),
                        ))
                except Exception:
                    pass

        # Ensure we have scores for all structures
        while len(scores) < len(structure_paths):
            scores.append(StructureScore())

        return structure_paths, scores

    def verify_installation(self) -> tuple[bool, str]:
        """Verify ColabFold installation."""
        checks = []

        # Check executable
        executable = self._installer.get_executable_path()
        if not executable:
            return False, "colabfold_batch executable not found"
        checks.append(f"Executable: {executable}")

        # Check version
        version = self._installer.get_installed_version()
        if version:
            checks.append(f"Version: {version}")

        # Check GPU
        try:
            import torch
            if torch.cuda.is_available():
                checks.append(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                checks.append("GPU: Not available (will be slow)")
        except ImportError:
            checks.append("PyTorch: Not installed")

        return True, "; ".join(checks)
