"""Base installer class for auto-installation and updates."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
import subprocess
import sys

from protein_design_hub.core.types import InstallationStatus
from protein_design_hub.core.exceptions import InstallationError


class ToolInstaller(ABC):
    """Base class for auto-installation and updates of prediction tools."""

    name: str = "unknown"
    description: str = ""

    @abstractmethod
    def is_installed(self) -> bool:
        """Check if the tool is installed."""
        pass

    @abstractmethod
    def get_installed_version(self) -> Optional[str]:
        """Get the currently installed version."""
        pass

    @abstractmethod
    def get_latest_version(self) -> Optional[str]:
        """Get the latest available version."""
        pass

    @abstractmethod
    def install(self) -> bool:
        """Install the tool. Returns True on success."""
        pass

    @abstractmethod
    def update(self) -> bool:
        """Update the tool to the latest version. Returns True on success."""
        pass

    def uninstall(self) -> bool:
        """Uninstall the tool. Returns True on success."""
        raise NotImplementedError(f"Uninstall not implemented for {self.name}")

    def get_status(self) -> InstallationStatus:
        """Get the current installation status."""
        try:
            installed = self.is_installed()
            version = self.get_installed_version() if installed else None
            latest = self.get_latest_version()
            gpu_available = self._check_gpu()

            return InstallationStatus(
                name=self.name,
                installed=installed,
                version=version,
                latest_version=latest,
                gpu_available=gpu_available,
            )
        except Exception as e:
            return InstallationStatus(
                name=self.name,
                installed=False,
                error_message=str(e),
            )

    def ensure_installed(self, auto_update: bool = False) -> bool:
        """
        Ensure the tool is installed, optionally updating if outdated.

        Args:
            auto_update: If True, update to latest version if outdated.

        Returns:
            True if tool is installed and ready to use.

        Raises:
            InstallationError: If installation fails.
        """
        if not self.is_installed():
            print(f"Installing {self.name}...")
            if not self.install():
                raise InstallationError(self.name, "Installation failed")
            print(f"Successfully installed {self.name}")
            return True

        if auto_update:
            installed = self.get_installed_version()
            latest = self.get_latest_version()

            if installed and latest and installed != latest:
                print(f"Updating {self.name}: {installed} -> {latest}")
                if not self.update():
                    print(f"Warning: Failed to update {self.name}")
                    return True  # Still installed, just not updated
                print(f"Successfully updated {self.name}")

        return True

    def _check_gpu(self) -> bool:
        """Check if GPU is available for this tool."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _run_command(
        self,
        command: list[str],
        capture_output: bool = True,
        check: bool = False,
        env: dict = None,
    ) -> subprocess.CompletedProcess:
        """Run a shell command."""
        return subprocess.run(
            command,
            capture_output=capture_output,
            check=check,
            env=env,
            text=True,
        )

    def _pip_install(self, package: str, upgrade: bool = False) -> bool:
        """Install a package using pip."""
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package)

        result = self._run_command(cmd)
        return result.returncode == 0

    def _get_pypi_version(self, package: str) -> Optional[str]:
        """Get the latest version of a package from PyPI."""
        try:
            import requests

            resp = requests.get(f"https://pypi.org/pypi/{package}/json", timeout=10)
            if resp.status_code == 200:
                return resp.json()["info"]["version"]
        except Exception:
            pass
        return None

    def _get_installed_pip_version(self, package: str) -> Optional[str]:
        """Get the installed version of a pip package."""
        try:
            result = self._run_command(
                [sys.executable, "-m", "pip", "show", package],
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return None
