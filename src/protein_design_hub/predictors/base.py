"""Base predictor class for all structure prediction tools."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import time

from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    PredictorType,
    Sequence,
    StructureScore,
)
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.exceptions import PredictorNotInstalledError, PredictionError


class BasePredictor(ABC):
    """Abstract base class for all structure predictors."""

    name: str = "base"
    predictor_type: PredictorType = None
    description: str = ""
    supports_multimer: bool = False
    supports_templates: bool = False
    supports_msa: bool = False
    output_format: str = "cif"  # Default output format

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the predictor.

        Args:
            settings: Configuration settings. Uses global settings if not provided.
        """
        self.settings = settings or get_settings()
        self._installer: Optional[ToolInstaller] = None

    @property
    @abstractmethod
    def installer(self) -> ToolInstaller:
        """Get the installer for this predictor."""
        pass

    @abstractmethod
    def _predict(self, input_data: PredictionInput, output_dir: Path) -> PredictionResult:
        """
        Internal prediction method to be implemented by subclasses.

        Args:
            input_data: Input data for prediction.
            output_dir: Directory to save output files.

        Returns:
            PredictionResult with structure paths and scores.
        """
        pass

    def predict(self, input_data: PredictionInput) -> PredictionResult:
        """
        Run structure prediction.

        Args:
            input_data: Input data for prediction.

        Returns:
            PredictionResult with structure paths and scores.

        Raises:
            PredictorNotInstalledError: If the predictor is not installed.
            PredictionError: If prediction fails.
        """
        # Ensure predictor is installed
        if not self.installer.is_installed():
            raise PredictorNotInstalledError(
                self.name,
                f"Run: pdhub install --predictor {self.name}"
            )

        # Set up output directory
        output_dir = input_data.output_dir
        if output_dir is None:
            output_dir = self.settings.output.base_dir / input_data.job_id / self.name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run prediction with timing
        start_time = time.time()
        try:
            result = self._predict(input_data, output_dir)
            result.runtime_seconds = time.time() - start_time
            return result
        except Exception as e:
            runtime = time.time() - start_time
            return PredictionResult(
                job_id=input_data.job_id,
                predictor=self.predictor_type,
                structure_paths=[],
                scores=[],
                runtime_seconds=runtime,
                success=False,
                error_message=str(e),
            )

    def verify_installation(self) -> Tuple[bool, str]:
        """
        Verify the predictor is correctly installed and functional.

        Returns:
            Tuple of (success, message).
        """
        checks = []

        # Check if installed
        if not self.installer.is_installed():
            return False, f"{self.name} is not installed"

        version = self.installer.get_installed_version()
        if version:
            checks.append(f"Version: {version}")

        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                checks.append(f"GPU: {device_name}")
            else:
                checks.append("GPU: Not available (will use CPU)")
        except ImportError:
            checks.append("PyTorch: Not installed")

        return True, "; ".join(checks)

    def run_test_prediction(self) -> Tuple[bool, str]:
        """
        Run a minimal test prediction to verify functionality.

        Returns:
            Tuple of (success, message).
        """
        # Short test sequence (30 residues)
        test_sequence = "MKFLILLFNILCLFPVLAADNHGVGPQGAS"

        test_input = PredictionInput(
            job_id="test_verification",
            sequences=[Sequence(id="test", sequence=test_sequence)],
            num_models=1,
            num_recycles=1,
        )

        try:
            result = self.predict(test_input)

            if not result.success:
                return False, f"Test failed: {result.error_message}"

            if not result.structure_paths:
                return False, "No structures generated"

            # Verify output file exists
            for path in result.structure_paths:
                if not path.exists():
                    return False, f"Output file not found: {path}"

            return True, f"Test successful ({result.runtime_seconds:.1f}s)"

        except Exception as e:
            return False, f"Test failed with exception: {str(e)}"

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory after prediction."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

    def get_status(self) -> dict:
        """
        Get the current status of this predictor.

        Returns:
            Dictionary with status information.
        """
        install_status = self.installer.get_status()

        return {
            "name": self.name,
            "type": self.predictor_type.value if self.predictor_type else None,
            "description": self.description,
            "installed": install_status.installed,
            "version": install_status.version,
            "latest_version": install_status.latest_version,
            "needs_update": install_status.needs_update,
            "gpu_available": install_status.gpu_available,
            "supports_multimer": self.supports_multimer,
            "supports_templates": self.supports_templates,
            "supports_msa": self.supports_msa,
            "output_format": self.output_format,
        }
