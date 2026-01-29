"""ESM-IF1 installer."""

from pathlib import Path
from typing import Optional, Tuple

from protein_design_hub.core.installer import ToolInstaller


class ESMIFInstaller(ToolInstaller):
    """Installer for ESM-IF1 inverse folding model."""

    @property
    def name(self) -> str:
        return "esmif"

    @property
    def package_name(self) -> str:
        return "fair-esm"

    def is_installed(self) -> bool:
        """Check if ESM-IF1 is installed."""
        try:
            import esm
            # Check if inverse folding model is available
            if hasattr(esm, 'pretrained') and hasattr(esm.pretrained, 'esm_if1_gvp4_t16_142M_UR50'):
                return True
            return False
        except ImportError:
            return False

    def get_installed_version(self) -> Optional[str]:
        """Get installed version."""
        try:
            import esm
            return getattr(esm, '__version__', 'unknown')
        except ImportError:
            return None

    def get_latest_version(self) -> Optional[str]:
        """Get latest available version."""
        # TODO: Implement PyPI check
        return None

    def install(self, force: bool = False) -> Tuple[bool, str]:
        """
        Install ESM-IF1.

        Args:
            force: Force reinstall even if already installed.

        Returns:
            Tuple of (success, message).
        """
        if self.is_installed() and not force:
            return True, "ESM-IF1 already installed"

        try:
            import subprocess
            import sys

            # Install fair-esm package
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "fair-esm"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                return False, f"pip install failed: {result.stderr}"

            # Verify installation
            if self.is_installed():
                return True, "ESM-IF1 installed successfully"
            else:
                return False, "Installation completed but verification failed"

        except subprocess.TimeoutExpired:
            return False, "Installation timed out"
        except Exception as e:
            return False, f"Installation failed: {str(e)}"

    def uninstall(self) -> Tuple[bool, str]:
        """Uninstall ESM-IF1."""
        try:
            import subprocess
            import sys

            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "fair-esm"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                return True, "ESM-IF1 uninstalled"
            return False, f"Uninstall failed: {result.stderr}"

        except Exception as e:
            return False, f"Uninstall failed: {str(e)}"

    def update(self) -> Tuple[bool, str]:
        """Update to latest version."""
        return self.install(force=True)

    def ensure_installed(self, auto_update: bool = False) -> None:
        """Ensure ESM-IF1 is installed."""
        if not self.is_installed():
            success, message = self.install()
            if not success:
                from protein_design_hub.core.exceptions import InstallationError
                raise InstallationError(self.name, message)
