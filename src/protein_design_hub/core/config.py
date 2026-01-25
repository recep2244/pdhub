"""Configuration management for Protein Design Hub."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ColabFoldConfig(BaseModel):
    """ColabFold predictor configuration."""

    enabled: bool = True
    num_models: int = 5
    num_recycles: int = 3
    use_amber: bool = False
    use_templates: bool = False
    msa_mode: str = "mmseqs2_uniref_env"


class Chai1Config(BaseModel):
    """Chai-1 predictor configuration."""

    enabled: bool = True
    num_trunk_recycles: int = 3
    num_diffusion_timesteps: int = 200
    seed: int = 42


class Boltz2Config(BaseModel):
    """Boltz-2 predictor configuration."""

    enabled: bool = True
    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1


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


class LDDTConfig(BaseModel):
    """lDDT metric configuration."""

    inclusion_radius: float = 15.0
    sequence_separation: int = 0


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""

    metrics: list[str] = Field(default_factory=lambda: ["lddt", "tm_score", "qs_score", "rmsd"])
    lddt: LDDTConfig = Field(default_factory=LDDTConfig)
    tmalign_path: Optional[Path] = None


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
                return cls.from_yaml(path)

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
