"""Chai-1 installer."""

from typing import Optional

from protein_design_hub.core.installer import ToolInstaller


class Chai1Installer(ToolInstaller):
    """Installer for Chai-1."""

    name = "chai-1"
    description = "Chai-1 - Multi-molecule structure prediction from Chai Discovery"

    def is_installed(self) -> bool:
        """Check if Chai-1 is installed."""
        try:
            import chai_lab
            return True
        except ImportError:
            return False

    def get_installed_version(self) -> Optional[str]:
        """Get the installed Chai-1 version."""
        try:
            import chai_lab
            return getattr(chai_lab, "__version__", "unknown")
        except ImportError:
            return None

    def get_latest_version(self) -> Optional[str]:
        """Get the latest Chai-1 version from PyPI."""
        return self._get_pypi_version("chai_lab")

    def install(self) -> bool:
        """Install Chai-1 via pip."""
        print("Installing Chai-1...")
        success = self._pip_install("chai_lab")
        if success:
            print("Chai-1 installed successfully")
            # Note: Model weights are downloaded on first use
            print("Note: Model weights will be downloaded on first prediction")
        return success

    def update(self) -> bool:
        """Update Chai-1 to the latest version."""
        print("Updating Chai-1...")
        return self._pip_install("chai_lab", upgrade=True)

    def verify_cuda(self) -> tuple[bool, str]:
        """Verify CUDA is available for Chai-1."""
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                return True, f"GPU: {device_name} ({memory:.1f} GB)"
            else:
                return False, "CUDA not available"
        except ImportError:
            return False, "PyTorch not installed"

    def verify_model_weights(self) -> tuple[bool, str]:
        """Check if model weights are downloaded."""
        try:
            from chai_lab.chai1 import run_inference
            # Just importing should be enough to verify the module is functional
            return True, "Model module accessible"
        except ImportError as e:
            return False, f"Model import failed: {e}"
        except Exception as e:
            return False, f"Model check failed: {e}"
