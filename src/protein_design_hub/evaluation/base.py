"""Base class for evaluation metrics."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any


class BaseMetric(ABC):
    """Abstract base class for structure evaluation metrics."""

    name: str = "base"
    description: str = ""
    requires_reference: bool = False

    @abstractmethod
    def compute(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Compute the metric.

        Args:
            model_path: Path to the model structure.
            reference_path: Path to reference structure (if required).
            **kwargs: Additional metric-specific parameters.

        Returns:
            Dictionary with computed metric values.
        """
        pass

    def is_available(self) -> bool:
        """Check if the metric can be computed (dependencies available)."""
        return True

    def get_requirements(self) -> str:
        """Get a description of requirements for this metric."""
        return ""
