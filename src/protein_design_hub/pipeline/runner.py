"""Sequential pipeline runner for GPU job execution."""

from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import json

from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    PredictorType,
)
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.exceptions import PredictionError
from protein_design_hub.predictors.registry import PredictorRegistry, get_predictor
from protein_design_hub.predictors.base import BasePredictor


class SequentialPipelineRunner:
    """
    Run predictions sequentially, one GPU job at a time.

    This ensures GPU memory is properly managed between different predictors.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ):
        """
        Initialize the pipeline runner.

        Args:
            settings: Configuration settings.
            progress_callback: Optional callback function(predictor_name, current, total).
        """
        self.settings = settings or get_settings()
        self.progress_callback = progress_callback

    def run_single_predictor(
        self,
        predictor_name: str,
        input_data: PredictionInput,
        auto_install: bool = False,
    ) -> PredictionResult:
        """
        Run a single predictor.

        Args:
            predictor_name: Name of the predictor to run.
            input_data: Input data for prediction.
            auto_install: Whether to auto-install if not installed.

        Returns:
            PredictionResult from the predictor.
        """
        predictor = get_predictor(predictor_name, self.settings)

        # Ensure installed
        if auto_install:
            predictor.installer.ensure_installed(auto_update=False)

        # Run prediction
        result = predictor.predict(input_data)

        # Clear GPU memory after prediction
        self._clear_gpu_memory()

        return result

    def run_all_predictors(
        self,
        input_data: PredictionInput,
        predictors: Optional[List[str]] = None,
        skip_unavailable: bool = True,
    ) -> Dict[str, PredictionResult]:
        """
        Run all specified predictors sequentially.

        Args:
            input_data: Input data for predictions.
            predictors: List of predictor names. Uses all enabled if not specified.
            skip_unavailable: Skip predictors that aren't installed instead of failing.

        Returns:
            Dictionary mapping predictor names to results.
        """
        if predictors is None:
            predictors = self._get_enabled_predictors()

        results = {}
        total = len(predictors)

        for i, predictor_name in enumerate(predictors):
            self._report_progress(predictor_name, i + 1, total)

            try:
                predictor = get_predictor(predictor_name, self.settings)

                # Check if installed
                if not predictor.installer.is_installed():
                    if skip_unavailable:
                        print(f"Skipping {predictor_name}: not installed")
                        results[predictor_name] = PredictionResult(
                            job_id=input_data.job_id,
                            predictor=predictor.predictor_type,
                            structure_paths=[],
                            scores=[],
                            runtime_seconds=0,
                            success=False,
                            error_message="Not installed",
                        )
                        continue
                    else:
                        raise PredictionError(
                            predictor_name,
                            f"Predictor not installed. Run: pdhub install --predictor {predictor_name}"
                        )

                # Run prediction
                print(f"Starting {predictor_name}...")
                result = predictor.predict(input_data)
                results[predictor_name] = result

                if result.success:
                    print(f"Completed {predictor_name} in {result.runtime_seconds:.1f}s")
                else:
                    print(f"Failed {predictor_name}: {result.error_message}")

            except Exception as e:
                results[predictor_name] = PredictionResult(
                    job_id=input_data.job_id,
                    predictor=PredictorType.COLABFOLD,  # Default type
                    structure_paths=[],
                    scores=[],
                    runtime_seconds=0,
                    success=False,
                    error_message=str(e),
                )
                print(f"Error with {predictor_name}: {e}")

            # Clear GPU memory before next predictor
            self._clear_gpu_memory()

        return results

    def run_enabled_predictors(
        self,
        input_data: PredictionInput,
    ) -> Dict[str, PredictionResult]:
        """
        Run all enabled predictors based on settings.

        Args:
            input_data: Input data for predictions.

        Returns:
            Dictionary mapping predictor names to results.
        """
        predictors = self._get_enabled_predictors()
        return self.run_all_predictors(input_data, predictors)

    def _get_enabled_predictors(self) -> List[str]:
        """Get list of enabled predictors from settings."""
        enabled = []
        if self.settings.predictors.colabfold.enabled:
            enabled.append("colabfold")
        if self.settings.predictors.chai1.enabled:
            enabled.append("chai1")
        if self.settings.predictors.boltz2.enabled:
            enabled.append("boltz2")
        return enabled

    def _clear_gpu_memory(self) -> None:
        """Release GPU memory between jobs."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Also try to garbage collect
        import gc
        gc.collect()

    def _report_progress(self, predictor_name: str, current: int, total: int) -> None:
        """Report progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(predictor_name, current, total)

    def get_predictor_status(self) -> Dict[str, Dict]:
        """
        Get status of all predictors.

        Returns:
            Dictionary with predictor status information.
        """
        status = {}
        for name in PredictorRegistry.list_available():
            try:
                predictor = get_predictor(name, self.settings)
                status[name] = predictor.get_status()
            except Exception as e:
                status[name] = {"error": str(e)}
        return status

    def verify_all_predictors(self) -> Dict[str, tuple[bool, str]]:
        """
        Verify all installed predictors.

        Returns:
            Dictionary mapping predictor names to (success, message) tuples.
        """
        results = {}
        for name in PredictorRegistry.list_available():
            try:
                predictor = get_predictor(name, self.settings)
                if predictor.installer.is_installed():
                    results[name] = predictor.verify_installation()
                else:
                    results[name] = (False, "Not installed")
            except Exception as e:
                results[name] = (False, str(e))
        return results

    def save_results(
        self,
        results: Dict[str, PredictionResult],
        output_dir: Path,
    ) -> Path:
        """
        Save all prediction results to a summary file.

        Args:
            results: Dictionary of prediction results.
            output_dir: Output directory.

        Returns:
            Path to the saved summary file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "timestamp": datetime.now().isoformat(),
            "predictors": {},
        }

        for name, result in results.items():
            summary["predictors"][name] = {
                "success": result.success,
                "runtime_seconds": result.runtime_seconds,
                "num_structures": len(result.structure_paths),
                "structure_paths": [str(p) for p in result.structure_paths],
                "error_message": result.error_message,
                "best_plddt": max(
                    (s.plddt for s in result.scores if s.plddt),
                    default=None
                ) if result.scores else None,
            }

        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary_path
