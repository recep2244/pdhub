"""Types for design workflows (sequence design / backbone generation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from protein_design_hub.core.types import Sequence


@dataclass
class DesignInput:
    """Input data for a design run."""

    job_id: str
    backbone_path: Path
    output_dir: Optional[Path] = None
    # Common sampling knobs
    num_sequences: int = 8
    temperature: float = 0.1
    seed: Optional[int] = None
    # Design constraints
    fixed_positions: Optional[str] = None   # e.g. "1-10, 25, 30-35" (1-indexed, chain A)
    chains_to_design: Optional[str] = None  # e.g. "A" or "A,B"
    omit_aa: Optional[str] = None          # e.g. "CM" — globally excluded residues
    use_soluble_model: bool = False

    def __post_init__(self) -> None:
        self.backbone_path = Path(self.backbone_path)
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)


@dataclass
class DesignResult:
    """Result from a design tool."""

    job_id: str
    designer: str
    sequences: list[Sequence] = field(default_factory=list)
    runtime_seconds: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
