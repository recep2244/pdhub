"""Shared workflow context passed between agents."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from protein_design_hub.core.types import (
    ComparisonResult,
    EvaluationResult,
    PredictionInput,
    PredictionResult,
    Sequence,
)


@dataclass
class WorkflowContext:
    """
    Shared context for the prediction pipeline.

    Each agent reads from and writes to this context. The orchestrator
    passes the same context through the agent chain.
    """

    # Identifiers and paths
    job_id: str = ""
    input_path: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    reference_path: Optional[Path] = None
    job_dir: Optional[Path] = None

    # Step 1: Input
    sequences: List[Sequence] = field(default_factory=list)
    prediction_input: Optional[PredictionInput] = None

    # Step 2: Prediction
    predictors: Optional[List[str]] = None  # None = use all enabled
    prediction_results: Dict[str, PredictionResult] = field(default_factory=dict)

    # Step 3: Evaluation
    evaluation_results: Dict[str, EvaluationResult] = field(default_factory=dict)

    # Step 4: Comparison
    comparison_result: Optional[ComparisonResult] = None

    # Optional: design / mutation / MSA workflow data
    extra: Dict[str, Any] = field(default_factory=dict)

    def with_job_dir(self) -> Path:
        """Return job directory, creating it if needed."""
        if self.job_dir is None:
            self.job_dir = Path(self.output_dir) / self.job_id
        self.job_dir.mkdir(parents=True, exist_ok=True)
        return self.job_dir
