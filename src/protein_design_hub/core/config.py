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

    # MSA settings (required for Boltz-2, so enabled by default)
    use_msa_server: bool = Field(default=True, description="Use MMSeqs2 server for MSA")
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
    no_kernels: bool = Field(default=True, description="Disable optimized kernels (set False if cuequivariance_torch is installed)")

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


# Provider presets: name → (base_url, default_model, default_api_key)
#
# Speed tiers for reference (approximate tok/s for generation):
#   LOCAL:  ollama ~30-80 tok/s (GPU-dependent), vllm ~100-200 tok/s
#   FAST:   groq ~300 tok/s, cerebras ~450 tok/s, sambanova ~200 tok/s
#   CLOUD:  deepseek ~50 tok/s, openai ~80 tok/s, gemini ~100 tok/s
LLM_PROVIDER_PRESETS: dict[str, tuple[str, str, str]] = {
    # ── Local (no API key) ──────────────────────────────────────────
    "ollama":    ("http://localhost:11434/v1",  "qwen2.5:14b",      "ollama"),
    "lmstudio":  ("http://localhost:1234/v1",   "default",           "lm-studio"),
    "vllm":      ("http://localhost:8000/v1",   "default",           "vllm"),
    "llamacpp":  ("http://localhost:8080/v1",   "default",           "llamacpp"),
    # ── Fast cloud (free tiers or very cheap) ───────────────────────
    "groq":      ("https://api.groq.com/openai/v1",      "llama-3.3-70b-versatile", ""),
    "cerebras":  ("https://api.cerebras.ai/v1",           "llama-3.3-70b",           ""),
    "sambanova": ("https://api.sambanova.ai/v1",          "Meta-Llama-3.3-70B-Instruct", ""),
    # ── Cloud (API key required) ────────────────────────────────────
    "deepseek":  ("https://api.deepseek.com/v1",          "deepseek-chat",     ""),
    "openai":    ("https://api.openai.com/v1",             "gpt-4o",           ""),
    "gemini":    ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash", ""),
    "kimi":      ("https://api.moonshot.cn/v1",            "kimi-k2",          ""),
    "openrouter": ("https://openrouter.ai/api/v1",        "meta-llama/llama-3.3-70b-instruct", ""),
}


class LLMConfig(BaseModel):
    """LLM configuration for agent meetings.

    Supports any OpenAI-compatible API backend.  Pre-configured
    providers (set ``provider`` and optionally ``model``):

    **Local (no API key needed):**
      - ``ollama``     → http://localhost:11434/v1  (qwen2.5:14b default)
      - ``lmstudio``   → http://localhost:1234/v1
      - ``vllm``       → http://localhost:8000/v1   (fastest local)
      - ``llamacpp``   → http://localhost:8080/v1

    **Fast cloud (free tiers available, ~300+ tok/s):**
      - ``groq``       → llama-3.3-70b  (GROQ_API_KEY, free tier)
      - ``cerebras``   → llama-3.3-70b  (CEREBRAS_API_KEY, free tier)
      - ``sambanova``  → llama-3.3-70b  (SAMBANOVA_API_KEY, free tier)

    **Cloud (API key required):**
      - ``deepseek``   → deepseek-chat  ($0.28/1M in, DEEPSEEK_API_KEY)
      - ``openai``     → gpt-4o         (OPENAI_API_KEY)
      - ``gemini``     → gemini-2.5-flash (GEMINI_API_KEY, free tier)
      - ``kimi``       → kimi-k2        (MOONSHOT_API_KEY)
      - ``openrouter`` → many models    (OPENROUTER_API_KEY)

    Or use ``custom`` and set ``base_url`` manually.
    """

    provider: str = Field(
        default="ollama",
        description=(
            "LLM provider preset: ollama, lmstudio, vllm, llamacpp, "
            "deepseek, openai, gemini, kimi, or custom"
        ),
    )
    base_url: str = Field(
        default="",
        description="Base URL (auto-set from provider if empty)"
    )
    model: str = Field(
        default="",
        description="Model name (auto-set from provider if empty)"
    )
    api_key: str = Field(
        default="",
        description="API key (auto-set from provider if empty; reads env var as fallback)"
    )
    temperature: float = Field(
        default=0.2,
        description="Sampling temperature for meetings"
    )
    max_tokens: Optional[int] = Field(
        default=4096,
        description="Max tokens per response (None = model default)"
    )
    num_rounds: int = Field(
        default=1,
        description="Default number of discussion rounds per meeting"
    )

    def resolve(self) -> "LLMConfig":
        """Return a copy with provider defaults filled in."""
        import os
        preset = LLM_PROVIDER_PRESETS.get(self.provider, (None, None, None))
        preset_url, preset_model, preset_key = preset

        base_url = self.base_url or preset_url or "http://localhost:11434/v1"
        model = self.model or preset_model or "qwen2.5:14b"
        # Auto-migrate legacy Ollama defaults to the current project default.
        # This keeps older user configs from silently sticking to llama3.2.
        if self.provider == "ollama" and model in {"llama3.2", "llama3.2:latest"}:
            model = "qwen2.5:14b"

        # API key: explicit > env var > preset
        api_key = self.api_key
        if not api_key:
            env_map = {
                "openai":     "OPENAI_API_KEY",
                "deepseek":   "DEEPSEEK_API_KEY",
                "gemini":     "GEMINI_API_KEY",
                "kimi":       "MOONSHOT_API_KEY",
                "groq":       "GROQ_API_KEY",
                "cerebras":   "CEREBRAS_API_KEY",
                "sambanova":  "SAMBANOVA_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
            }
            env_var = env_map.get(self.provider, "")
            api_key = os.environ.get(env_var, "") if env_var else ""
        if not api_key:
            api_key = preset_key or "no-key"

        return LLMConfig(
            provider=self.provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            num_rounds=self.num_rounds,
        )


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
    llm: LLMConfig = Field(default_factory=LLMConfig)

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
