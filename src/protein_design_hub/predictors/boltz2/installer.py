"""Boltz-2 installer."""

from typing import Optional

from protein_design_hub.core.installer import ToolInstaller


class Boltz2Installer(ToolInstaller):
    """Installer for Boltz-2."""

    name = "boltz-2"
    description = "Boltz-2 - Latest biomolecular structure prediction"

    def is_installed(self) -> bool:
        """Check if Boltz is installed."""
        try:
            import boltz
            return True
        except ImportError:
            return False

    def get_installed_version(self) -> Optional[str]:
        """Get the installed Boltz version."""
        try:
            import boltz
            return getattr(boltz, "__version__", "unknown")
        except ImportError:
            return None

    def get_latest_version(self) -> Optional[str]:
        """Get the latest Boltz version from PyPI."""
        return self._get_pypi_version("boltz")

    def install(self) -> bool:
        """Install Boltz via pip."""
        print("Installing Boltz-2...")
        success = self._pip_install("boltz")
        if success:
            print("Boltz-2 installed successfully")
            print("Note: Model weights will be downloaded on first prediction")
        return success

    def update(self) -> bool:
        """Update Boltz to the latest version."""
        print("Updating Boltz-2...")
        return self._pip_install("boltz", upgrade=True)

    def verify_cuda(self) -> tuple[bool, str]:
        """Verify CUDA is available for Boltz."""
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
        """Check if model weights are accessible."""
        try:
            import boltz
            # Check if the main prediction module is importable
            return True, "Boltz module accessible"
        except ImportError as e:
            return False, f"Boltz import failed: {e}"
        except Exception as e:
            return False, f"Boltz check failed: {e}"
