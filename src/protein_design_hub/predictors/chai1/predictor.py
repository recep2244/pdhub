"""Chai-1 predictor implementation with comprehensive parameters."""

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
    """
    Chai-1 predictor for multi-molecule structure prediction.

    Chai-1 supports:
    - Proteins, DNA, RNA, and small molecule ligands
    - Multi-chain complexes
    - Diffusion-based structure generation
    - ESM embeddings for improved accuracy

    Parameters are configured via settings.predictors.chai1
    """

    name = "chai1"
    predictor_type = PredictorType.CHAI1
    description = "Multi-molecule structure prediction from Chai Discovery"
    supports_multimer = True
    supports_templates = True
    supports_msa = True
    output_format = "cif"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer = Chai1Installer()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        """Run Chai-1 prediction with all available parameters."""
        try:
            from chai_lab.chai1 import run_inference
        except ImportError as e:
            raise PredictionError(self.name, f"Failed to import chai_lab: {e}")

        config = self.settings.predictors.chai1

        # Prepare FASTA input for Chai-1
        fasta_path = output_dir / "input.fasta"
        self._write_chai_fasta(input_data, fasta_path)

        # Prepare constraint file if constraints provided
        constraint_path = None
        if input_data.constraints:
            constraint_path = output_dir / "constraints.json"
            self._write_constraints(input_data.constraints, constraint_path)
        elif config.constraint_path:
            constraint_path = config.constraint_path

        # Prepare MSA directory if MSA provided
        msa_directory = None
        if input_data.msa:
            msa_directory = output_dir / "msa"
            msa_directory.mkdir(exist_ok=True)
            self._write_msa(input_data.msa, msa_directory)
        elif config.msa_directory:
            msa_directory = config.msa_directory

        # Prepare template hits if templates provided
        template_hits_path = None
        if input_data.templates:
            template_hits_path = output_dir / "templates"
            template_hits_path.mkdir(exist_ok=True)
            self._prepare_templates(input_data.templates, template_hits_path)
        elif config.template_hits_path:
            template_hits_path = config.template_hits_path

        # Determine device
        device = config.device
        if device is None:
            device = self.settings.gpu.device if self._check_gpu() else "cpu"

        try:
            # Run inference with all parameters
            candidates = run_inference(
                fasta_file=fasta_path,
                output_dir=output_dir,
                # ESM embeddings
                use_esm_embeddings=config.use_esm_embeddings,
                # MSA settings
                use_msa_server=config.use_msa_server,
                msa_server_url=config.msa_server_url,
                msa_directory=msa_directory,
                recycle_msa_subsample=config.recycle_msa_subsample,
                # Template settings
                use_templates_server=config.use_templates_server,
                template_hits_path=template_hits_path,
                # Constraint settings
                constraint_path=constraint_path,
                # Core prediction parameters
                num_trunk_recycles=config.num_trunk_recycles,
                num_diffn_timesteps=config.num_diffn_timesteps,
                num_diffn_samples=config.num_diffn_samples,
                num_trunk_samples=config.num_trunk_samples,
                # Random seed
                seed=config.seed,
                # Device and memory
                device=device,
                low_memory=config.low_memory,
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
                metadata={
                    "num_trunk_recycles": config.num_trunk_recycles,
                    "num_diffn_timesteps": config.num_diffn_timesteps,
                    "num_diffn_samples": config.num_diffn_samples,
                    "use_esm_embeddings": config.use_esm_embeddings,
                    "device": device,
                },
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

    def _write_constraints(self, constraints: list, constraint_path: Path) -> None:
        """Write constraints file for Chai-1."""
        # Chai-1 constraint format
        constraint_data = []
        for c in constraints:
            constraint_data.append({
                "residue_i": c.residue1,
                "residue_j": c.residue2,
                "chain_i": c.chain1,
                "chain_j": c.chain2,
                "distance_min": c.distance_min,
                "distance_max": c.distance_max,
            })

        with open(constraint_path, "w") as f:
            json.dump(constraint_data, f, indent=2)

    def _write_msa(self, msa, msa_directory: Path) -> None:
        """Write MSA files for Chai-1."""
        from protein_design_hub.io.parsers.a3m import A3MParser

        parser = A3MParser()
        msa_path = msa_directory / f"{msa.query_id}.a3m"
        parser.write(msa, msa_path)

    def _prepare_templates(self, templates: list, template_dir: Path) -> None:
        """Prepare template structures for Chai-1."""
        import shutil

        for template in templates:
            dest = template_dir / template.path.name
            shutil.copy(template.path, dest)

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

    def get_available_parameters(self) -> dict:
        """Get all available parameters with descriptions."""
        return {
            "num_trunk_recycles": {
                "type": "int",
                "default": 3,
                "range": [1, 20],
                "description": "Number of trunk recycles for structure refinement",
            },
            "num_diffn_timesteps": {
                "type": "int",
                "default": 200,
                "range": [50, 1000],
                "description": "Number of diffusion timesteps",
            },
            "num_diffn_samples": {
                "type": "int",
                "default": 5,
                "range": [1, 20],
                "description": "Number of diffusion samples to generate",
            },
            "num_trunk_samples": {
                "type": "int",
                "default": 1,
                "range": [1, 10],
                "description": "Number of trunk samples",
            },
            "use_esm_embeddings": {
                "type": "bool",
                "default": True,
                "description": "Use ESM language model embeddings",
            },
            "use_msa_server": {
                "type": "bool",
                "default": False,
                "description": "Use ColabFold MSA server",
            },
            "low_memory": {
                "type": "bool",
                "default": True,
                "description": "Use low memory mode for large proteins",
            },
            "seed": {
                "type": "int",
                "default": None,
                "description": "Random seed for reproducibility",
            },
        }
