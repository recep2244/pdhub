"""Base class for backbone generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import time

from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.design.generators.types import BackboneInput, BackboneResult


class BaseBackboneGenerator(ABC):
    name: str = "base"
    description: str = ""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._installer: Optional[ToolInstaller] = None

    @property
    @abstractmethod
    def installer(self) -> ToolInstaller:
        raise NotImplementedError

    @abstractmethod
    def _generate(self, input_data: BackboneInput, output_dir: Path) -> BackboneResult:
        raise NotImplementedError

    def generate(self, input_data: BackboneInput, auto_install: bool = False) -> BackboneResult:
        if auto_install:
            self.installer.ensure_installed(auto_update=False)

        if not self.installer.is_installed():
            return BackboneResult(
                job_id=input_data.job_id,
                generator=self.name,
                success=False,
                error_message=f"{self.name} not installed. Run: pdhub install generator {self.name}",
            )

        output_dir = input_data.output_dir
        if output_dir is None:
            output_dir = self.settings.output.base_dir / input_data.job_id / "backbones" / self.name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        result = self._generate(input_data, output_dir)
        result.runtime_seconds = time.time() - start
        return result

    def verify_installation(self) -> Tuple[bool, str]:
        if not self.installer.is_installed():
            return False, "Not installed"
        version = self.installer.get_installed_version()
        return True, f"Installed (v{version or 'unknown'})"
