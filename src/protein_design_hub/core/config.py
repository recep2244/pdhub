"""Configuration management for Protein Design Hub."""

from pathlib import Path
from typing import Optional, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ColabFoldConfig(BaseModel):
    """ColabFold predictor configuration - comprehensive parameters."""

    enabled: bool = True

    # Basic prediction settings
    num_models: int = Field(default=5, ge=1, le=5, description="Number of models to generate (1-5)")
    num_recycles: int = Field(default=3, ge=1, le=48, description="Number of prediction recycles")
    num_ensemble: int = Field(default=1, ge=1, description="Number of ensembles")
    num_seeds: int = Field(default=1, ge=1, description="Number of random seeds to try")
    random_seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    # MSA settings
    msa_mode: Literal["mmseqs2_uniref_env", "mmseqs2_uniref", "single_sequence"] = Field(
        default="mmseqs2_uniref_env",
        description="MSA generation mode"
    )
    max_seq: int = Field(default=512, description="Maximum number of sequences in MSA")
    max_extra_seq: int = Field(default=1024, description="Maximum extra sequences for templates")
    pair_mode: Literal["unpaired", "paired", "unpaired_paired"] = Field(
        default="unpaired_paired",
        description="MSA pairing mode for multimers"
    )

    # Model settings
    model_type: Literal["auto", "alphafold2", "alphafold2_ptm", "alphafold2_multimer_v1",
                        "alphafold2_multimer_v2", "alphafold2_multimer_v3"] = Field(
        default="auto",
        description="AlphaFold2 model type"
    )
    model_order: Optional[str] = Field(default=None, description="Comma-separated model order")

    # Relaxation settings
    use_amber: bool = Field(default=False, description="Use AMBER for structure relaxation")
    num_relax: int = Field(default=0, description="Number of top structures to relax")
    use_gpu_relax: bool = Field(default=True, description="Use GPU for AMBER relaxation")

    # Template settings
    use_templates: bool = Field(default=False, description="Use structure templates")
    custom_template_path: Optional[Path] = Field(default=None, description="Path to custom templates")

    # Early stopping
    stop_at_score: float = Field(default=100.0, description="Stop if pLDDT reaches this score")
    recycle_early_stop_tolerance: float = Field(
        default=0.5,
        description="Stop recycling if structure converges within this tolerance"
    )

    # Ranking
    rank_by: Literal["auto", "plddt", "ptm", "iptm", "multimer"] = Field(
        default="auto",
        description="Metric to rank models by"
    )

    # Output options
    save_single_representations: bool = Field(default=False, description="Save single representations")
    save_pair_representations: bool = Field(default=False, description="Save pair representations")
    save_all: bool = Field(default=False, description="Save all intermediate outputs")
    use_dropout: bool = Field(default=False, description="Enable dropout for uncertainty estimation")

    # Performance
    recompile_padding: int = Field(default=10, description="Padding for recompilation")
    sort_queries_by: Literal["none", "length", "random"] = Field(
        default="length",
        description="How to sort queries"
    )


class Chai1Config(BaseModel):
    """Chai-1 predictor configuration - comprehensive parameters."""

    enabled: bool = True

    # Core prediction settings
    num_trunk_recycles: int = Field(default=3, ge=1, le=20, description="Number of trunk recycles")
    num_diffn_timesteps: int = Field(default=200, ge=50, le=1000, description="Number of diffusion timesteps")
    num_diffn_samples: int = Field(default=5, ge=1, le=20, description="Number of diffusion samples")
    num_trunk_samples: int = Field(default=1, ge=1, description="Number of trunk samples")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

    # ESM embeddings
    use_esm_embeddings: bool = Field(default=True, description="Use ESM embeddings")

    # MSA settings
    use_msa_server: bool = Field(default=False, description="Use ColabFold MSA server")
    msa_server_url: str = Field(default="https://api.colabfold.com", description="MSA server URL")
    msa_directory: Optional[Path] = Field(default=None, description="Directory with precomputed MSAs")
    recycle_msa_subsample: int = Field(default=0, description="MSA subsampling for recycling")

    # Template settings
    use_templates_server: bool = Field(default=False, description="Use templates server")
    template_hits_path: Optional[Path] = Field(default=None, description="Path to template hits")

    # Constraint settings
    constraint_path: Optional[Path] = Field(default=None, description="Path to constraints file")

    # Memory optimization
    low_memory: bool = Field(default=True, description="Use low memory mode")

    # Device settings
    device: Optional[str] = Field(default=None, description="Device to use (cuda:0, cpu)")


class Boltz2Config(BaseModel):
    """Boltz-2 predictor configuration - comprehensive parameters."""

    enabled: bool = True

    # Model selection
    model: Literal["boltz1", "boltz2"] = Field(default="boltz2", description="Model version to use")

    # Core prediction settings
    recycling_steps: int = Field(default=3, ge=1, le=20, description="Number of recycling steps")
    sampling_steps: int = Field(default=200, ge=50, le=1000, description="Number of sampling steps")
    diffusion_samples: int = Field(default=1, ge=1, le=20, description="Number of diffusion samples")
    max_parallel_samples: Optional[int] = Field(default=None, description="Max parallel samples")

    # Sampling parameters
    step_scale: Optional[float] = Field(
        default=None,
        description="Step scale for diffusion (1.638 for Boltz-1, 1.5 for Boltz-2)"
    )
    seed: Optional[int] = Field(default=None, description="Random seed")

    # MSA settings
    use_msa_server: bool = Field(default=False, description="Use MMSeqs2 server for MSA")
    msa_server_url: Optional[str] = Field(default=None, description="MSA server URL")
    msa_pairing_strategy: Literal["greedy", "complete"] = Field(
        default="greedy",
        description="MSA pairing strategy"
    )
    max_msa_seqs: int = Field(default=8192, description="Maximum MSA sequences")
    subsample_msa: bool = Field(default=True, description="Whether to subsample MSA")
    num_subsampled_msa: int = Field(default=1024, description="Number of subsampled MSA sequences")

    # Potentials/steering
    use_potentials: bool = Field(default=False, description="Use potentials for steering")

    # Affinity prediction
    affinity_mw_correction: bool = Field(
        default=False,
        description="Add molecular weight correction to affinity"
    )
    sampling_steps_affinity: int = Field(default=200, description="Sampling steps for affinity")
    diffusion_samples_affinity: int = Field(default=5, description="Diffusion samples for affinity")

    # Output options
    output_format: Literal["pdb", "mmcif"] = Field(default="mmcif", description="Output format")
    write_full_pae: bool = Field(default=True, description="Write full PAE matrix")
    write_full_pde: bool = Field(default=False, description="Write full PDE matrix")
    write_embeddings: bool = Field(default=False, description="Write S and Z embeddings")

    # Performance
    devices: int = Field(default=1, description="Number of devices")
    accelerator: Literal["gpu", "cpu", "tpu"] = Field(default="gpu", description="Accelerator type")
    num_workers: int = Field(default=2, description="Number of dataloader workers")
    preprocessing_threads: int = Field(default=1, description="Preprocessing threads")
    no_kernels: bool = Field(default=False, description="Disable optimized kernels")

    # Checkpoints
    checkpoint: Optional[Path] = Field(default=None, description="Custom model checkpoint")
    affinity_checkpoint: Optional[Path] = Field(default=None, description="Affinity model checkpoint")

    # Misc
    override: bool = Field(default=False, description="Override existing predictions")


class PredictorConfig(BaseModel):
    """Configuration for all predictors."""

    colabfold: ColabFoldConfig = Field(default_factory=ColabFoldConfig)
    chai1: Chai1Config = Field(default_factory=Chai1Config)
    boltz2: Boltz2Config = Field(default_factory=Boltz2Config)


class OutputConfig(BaseModel):
    """Output configuration."""

    base_dir: Path = Path("./outputs")
    save_all_models: bool = True
    generate_report: bool = True
    save_scores: bool = True
    save_pae: bool = True
    save_plddt: bool = True


class LDDTConfig(BaseModel):
    """lDDT metric configuration."""

    inclusion_radius: float = 15.0
    sequence_separation: int = 0
    stereochemistry_check: bool = True


class TMScoreConfig(BaseModel):
    """TM-score configuration."""

    tmalign_path: Optional[Path] = None
    fast_mode: bool = False


class QSScoreConfig(BaseModel):
    """QS-score configuration."""

    contact_threshold: float = 12.0


class RMSDConfig(BaseModel):
    """RMSD configuration."""

    atoms: Literal["CA", "backbone", "heavy", "all"] = "CA"
    superpose: bool = True


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list[str] = Field(
        default_factory=lambda: ["lddt", "tm_score", "qs_score", "rmsd"]
    )
    lddt: LDDTConfig = Field(default_factory=LDDTConfig)
    tm_score: TMScoreConfig = Field(default_factory=TMScoreConfig)
    qs_score: QSScoreConfig = Field(default_factory=QSScoreConfig)
    rmsd: RMSDConfig = Field(default_factory=RMSDConfig)


class InstallationConfig(BaseModel):
    """Installation configuration."""

    auto_update: bool = False
    check_updates_on_start: bool = True
    colabfold_path: Optional[Path] = None
    tools_dir: Path = Path("~/.protein_design_hub/tools")


class GPUConfig(BaseModel):
    """GPU configuration."""

    device: str = "cuda:0"
    clear_cache_between_jobs: bool = True
    memory_fraction: float = 0.95
    allow_tf32: bool = True


class WebConfig(BaseModel):
    """Web UI configuration."""

    host: str = "localhost"
    port: int = 8501
    theme: str = "light"


class Settings(BaseSettings):
    """Main settings for Protein Design Hub."""

    model_config = SettingsConfigDict(
        env_prefix="PDHUB_",
        env_nested_delimiter="__",
    )

    output: OutputConfig = Field(default_factory=OutputConfig)
    predictors: PredictorConfig = Field(default_factory=PredictorConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    installation: InstallationConfig = Field(default_factory=InstallationConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: Path) -> None:
        """Save settings to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """Load settings from default location or specified path."""
        if config_path and config_path.exists():
            return cls.from_yaml(config_path)

        # Check default locations
        default_paths = [
            Path("config/default.yaml"),
            Path.home() / ".protein_design_hub" / "config.yaml",
            Path("/etc/protein_design_hub/config.yaml"),
        ]

        for path in default_paths:
            if path.exists():
                try:
                    return cls.from_yaml(path)
                except Exception:
                    pass

        # Return default settings
        return cls()


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings


def set_settings(settings: Settings) -> None:
    """Set the global settings instance."""
    global _settings
    _settings = settings
