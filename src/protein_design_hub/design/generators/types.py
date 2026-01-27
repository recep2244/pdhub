"""Types for backbone generation tools (e.g., RFdiffusion)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class BackboneInput:
    job_id: str
    output_dir: Optional[Path] = None

    # RFdiffusion-style knobs
    num_designs: int = 1
    contigs: Optional[str] = None
    input_pdb: Optional[Path] = None
    config_name: Optional[str] = None  # e.g. "symmetry"
    overrides: list[str] = field(default_factory=list)  # hydra overrides
    ckpt_override_path: Optional[Path] = None
    python_executable: Optional[Path] = None  # which python to run RFdiffusion with

    def __post_init__(self) -> None:
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        if self.input_pdb is not None:
            self.input_pdb = Path(self.input_pdb)
        if self.ckpt_override_path is not None:
            self.ckpt_override_path = Path(self.ckpt_override_path)
        if self.python_executable is not None:
            self.python_executable = Path(self.python_executable)


@dataclass
class BackboneResult:
    job_id: str
    generator: str
    backbone_paths: list[Path] = field(default_factory=list)
    runtime_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
