"""Chai-1 predictor implementation."""

from pathlib import Path
from typing import List, Optional
import json

from protein_design_hub.predictors.base import BasePredictor
from protein_design_hub.predictors.registry import PredictorRegistry
from protein_design_hub.predictors.chai1.installer import Chai1Installer
from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    PredictorType,
    StructureScore,
    MoleculeType,
)
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.exceptions import PredictionError


@PredictorRegistry.register("chai1")
class Chai1Predictor(BasePredictor):
    """Chai-1 predictor for multi-molecule structure prediction."""

    name = "chai1"
    predictor_type = PredictorType.CHAI1
    description = "Multi-molecule structure prediction from Chai Discovery"
    supports_multimer = True
    supports_templates = False
    supports_msa = False
    output_format = "cif"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer = Chai1Installer()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        """Run Chai-1 prediction."""
        try:
            from chai_lab.chai1 import run_inference
        except ImportError as e:
            raise PredictionError(self.name, f"Failed to import chai_lab: {e}")

        config = self.settings.predictors.chai1

        # Prepare FASTA input for Chai-1
        fasta_path = output_dir / "input.fasta"
        self._write_chai_fasta(input_data, fasta_path)

        try:
            # Run inference
            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                num_trunk_recycles=config.num_trunk_recycles,
                num_diffn_timesteps=config.num_diffusion_timesteps,
                seed=config.seed,
                device=self.settings.gpu.device if self._check_gpu() else "cpu",
                use_esm_embeddings=True,
            )

            # Parse results
            structure_paths, scores = self._parse_results(output_dir, candidates)

            return PredictionResult(
                job_id=input_data.job_id,
                predictor=self.predictor_type,
                structure_paths=structure_paths,
                scores=scores,
                runtime_seconds=0,  # Set by base class
                success=True,
            )

        except Exception as e:
            raise PredictionError(self.name, str(e), original_error=e)

    def _write_chai_fasta(self, input_data: PredictionInput, fasta_path: Path) -> None:
        """Write FASTA file in Chai-1 format with entity type annotations."""
        lines = []

        for seq in input_data.sequences:
            # Determine entity type
            if seq.molecule_type == MoleculeType.PROTEIN:
                entity_type = "protein"
            elif seq.molecule_type == MoleculeType.DNA:
                entity_type = "dna"
            elif seq.molecule_type == MoleculeType.RNA:
                entity_type = "rna"
            elif seq.molecule_type == MoleculeType.LIGAND:
                entity_type = "ligand"
            else:
                entity_type = "protein"

            # Chai-1 FASTA format: >ID|entity_type
            header = f">{seq.id}|{entity_type}"
            lines.append(header)
            lines.append(seq.sequence)

        fasta_path.write_text("\n".join(lines))

    def _parse_results(
        self,
        output_dir: Path,
        candidates: Optional[object] = None,
    ) -> tuple[List[Path], List[StructureScore]]:
        """Parse Chai-1 output files."""
        structure_paths = []
        scores = []

        # Find output CIF files
        for cif_file in sorted(output_dir.glob("pred.model_idx_*.cif")):
            structure_paths.append(cif_file)

        # Parse scores from output
        scores_file = output_dir / "scores.json"
        if scores_file.exists():
            try:
                with open(scores_file) as f:
                    scores_data = json.load(f)
                for model_scores in scores_data.get("models", []):
                    scores.append(StructureScore(
                        plddt=model_scores.get("plddt"),
                        ptm=model_scores.get("ptm"),
                        iptm=model_scores.get("iptm"),
                        confidence=model_scores.get("aggregate_score"),
                    ))
            except Exception:
                pass

        # Try to extract scores from candidates object
        if candidates is not None and hasattr(candidates, "scores"):
            try:
                for i, cand_scores in enumerate(candidates.scores):
                    if i < len(scores):
                        continue
                    scores.append(StructureScore(
                        plddt=float(cand_scores.get("plddt", 0)),
                        ptm=float(cand_scores.get("ptm", 0)),
                        iptm=float(cand_scores.get("iptm", 0)),
                        confidence=float(cand_scores.get("aggregate_score", 0)),
                    ))
            except Exception:
                pass

        # Ensure we have scores for all structures
        while len(scores) < len(structure_paths):
            scores.append(StructureScore())

        # Save scores to file for reference
        if scores and not scores_file.exists():
            scores_data = {
                "models": [
                    {
                        "plddt": s.plddt,
                        "ptm": s.ptm,
                        "iptm": s.iptm,
                        "confidence": s.confidence,
                    }
                    for s in scores
                ]
            }
            try:
                with open(scores_file, "w") as f:
                    json.dump(scores_data, f, indent=2)
            except Exception:
                pass

        return structure_paths, scores

    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def verify_installation(self) -> tuple[bool, str]:
        """Verify Chai-1 installation."""
        checks = []

        # Check import
        try:
            import chai_lab
            checks.append(f"chai_lab v{getattr(chai_lab, '__version__', 'unknown')} imported")
        except ImportError as e:
            return False, f"Cannot import chai_lab: {e}"

        # Check CUDA
        cuda_ok, cuda_msg = self._installer.verify_cuda()
        checks.append(cuda_msg)

        # Check model
        model_ok, model_msg = self._installer.verify_model_weights()
        checks.append(model_msg)

        return True, "; ".join(checks)
