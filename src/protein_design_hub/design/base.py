"""Base class for design tools (sequence design, backbone generation)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import time

from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.exceptions import InstallationError
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.design.types import DesignInput, DesignResult


class BaseDesigner(ABC):
    """Abstract base class for all design tools."""

    name: str = "base"
    description: str = ""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._installer: Optional[ToolInstaller] = None

    @property
    @abstractmethod
    def installer(self) -> ToolInstaller:
        """Return the installer for this designer."""
        raise NotImplementedError

    @abstractmethod
    def _design(self, input_data: DesignInput, output_dir: Path) -> DesignResult:
        """Run the design tool."""
        raise NotImplementedError

    def design(self, input_data: DesignInput, auto_install: bool = False) -> DesignResult:
        """Public entry point with output directory + timing."""
        if auto_install:
            try:
                self.installer.ensure_installed(auto_update=False)
            except Exception as e:
                raise InstallationError(self.name, "Auto-install failed", original_error=e)

        if not self.installer.is_installed():
            raise InstallationError(
                self.name,
                f"{self.name} is not installed. Run `pdhub install designer {self.name}`.",
            )

        output_dir = input_data.output_dir
        if output_dir is None:
            output_dir = self.settings.output.base_dir / input_data.job_id / "design" / self.name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        result = self._design(input_data, output_dir)
        result.runtime_seconds = time.time() - start
        return result

    def verify_installation(self) -> Tuple[bool, str]:
        if not self.installer.is_installed():
            return False, "Not installed"
        version = self.installer.get_installed_version()
        return True, f"Installed (v{version or 'unknown'})"
