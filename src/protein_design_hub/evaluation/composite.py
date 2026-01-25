"""Composite evaluator combining multiple metrics."""

from pathlib import Path
from typing import Dict, List, Optional, Any

from protein_design_hub.evaluation.base import BaseMetric
from protein_design_hub.evaluation.metrics.lddt import LDDTMetric
from protein_design_hub.evaluation.metrics.qs_score import QSScoreMetric
from protein_design_hub.evaluation.metrics.tm_score import TMScoreMetric
from protein_design_hub.evaluation.metrics.rmsd import RMSDMetric
from protein_design_hub.core.types import EvaluationResult
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.exceptions import EvaluationError


class CompositeEvaluator:
    """Evaluator that combines multiple structure quality metrics."""

    AVAILABLE_METRICS = {
        "lddt": LDDTMetric,
        "qs_score": QSScoreMetric,
        "tm_score": TMScoreMetric,
        "rmsd": RMSDMetric,
    }

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize composite evaluator.

        Args:
            metrics: List of metric names to use. Uses all if not specified.
            settings: Configuration settings.
        """
        self.settings = settings or get_settings()

        if metrics is None:
            metrics = self.settings.evaluation.metrics

        self.metrics: Dict[str, BaseMetric] = {}
        for metric_name in metrics:
            metric_name_lower = metric_name.lower().replace("-", "_")
            if metric_name_lower in self.AVAILABLE_METRICS:
                metric_class = self.AVAILABLE_METRICS[metric_name_lower]

                # Pass metric-specific settings
                if metric_name_lower == "lddt":
                    lddt_config = self.settings.evaluation.lddt
                    self.metrics[metric_name_lower] = metric_class(
                        inclusion_radius=lddt_config.inclusion_radius,
                        sequence_separation=lddt_config.sequence_separation,
                    )
                elif metric_name_lower == "tm_score":
                    self.metrics[metric_name_lower] = metric_class(
                        tmalign_path=self.settings.evaluation.tmalign_path,
                    )
                else:
                    self.metrics[metric_name_lower] = metric_class()

    def evaluate(
        self,
        model_path: Path,
        reference_path: Optional[Path] = None,
    ) -> EvaluationResult:
        """
        Evaluate a structure using all configured metrics.

        Args:
            model_path: Path to model structure.
            reference_path: Path to reference structure (required for most metrics).

        Returns:
            EvaluationResult with all computed metrics.
        """
        model_path = Path(model_path)
        if reference_path:
            reference_path = Path(reference_path)

        result = EvaluationResult(
            structure_path=model_path,
            reference_path=reference_path,
        )

        errors = []

        for metric_name, metric in self.metrics.items():
            # Skip metrics that require reference if not provided
            if metric.requires_reference and reference_path is None:
                continue

            if not metric.is_available():
                errors.append(f"{metric_name}: {metric.get_requirements()}")
                continue

            try:
                metric_result = metric.compute(model_path, reference_path)

                # Map results to EvaluationResult fields
                if metric_name == "lddt":
                    result.lddt = metric_result.get("lddt")
                    result.lddt_per_residue = metric_result.get("lddt_per_residue")
                elif metric_name == "qs_score":
                    result.qs_score = metric_result.get("qs_score")
                elif metric_name == "tm_score":
                    result.tm_score = metric_result.get("tm_score")
                    result.gdt_ts = metric_result.get("gdt_ts")
                    result.gdt_ha = metric_result.get("gdt_ha")
                    # Also get RMSD from TMalign if available
                    if result.rmsd is None:
                        result.rmsd = metric_result.get("rmsd")
                elif metric_name == "rmsd":
                    result.rmsd = metric_result.get("rmsd")

                # Store full metric result in metadata
                if "metadata" not in result.__dict__ or result.metadata is None:
                    result.metadata = {}
                result.metadata[metric_name] = metric_result

            except EvaluationError as e:
                errors.append(f"{metric_name}: {str(e)}")
            except Exception as e:
                errors.append(f"{metric_name}: Unexpected error - {str(e)}")

        if errors:
            result.metadata["errors"] = errors

        return result

    def evaluate_batch(
        self,
        model_paths: List[Path],
        reference_path: Optional[Path] = None,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple structures.

        Args:
            model_paths: List of paths to model structures.
            reference_path: Path to reference structure.

        Returns:
            List of EvaluationResult objects.
        """
        results = []
        for model_path in model_paths:
            result = self.evaluate(model_path, reference_path)
            results.append(result)
        return results

    def get_available_metrics(self) -> Dict[str, bool]:
        """
        Check which metrics are available.

        Returns:
            Dictionary mapping metric names to availability status.
        """
        return {name: metric.is_available() for name, metric in self.metrics.items()}

    def get_metric_requirements(self) -> Dict[str, str]:
        """
        Get requirements for unavailable metrics.

        Returns:
            Dictionary mapping metric names to requirement descriptions.
        """
        return {
            name: metric.get_requirements()
            for name, metric in self.metrics.items()
            if not metric.is_available()
        }

    @classmethod
    def list_all_metrics(cls) -> List[Dict[str, Any]]:
        """
        List all available metric types.

        Returns:
            List of metric information dictionaries.
        """
        metrics_info = []
        for name, metric_class in cls.AVAILABLE_METRICS.items():
            metric = metric_class()
            metrics_info.append({
                "name": name,
                "description": metric.description,
                "requires_reference": metric.requires_reference,
                "available": metric.is_available(),
                "requirements": metric.get_requirements() if not metric.is_available() else None,
            })
        return metrics_info
