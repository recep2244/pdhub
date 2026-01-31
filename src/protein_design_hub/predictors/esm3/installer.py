"""ESM3 installer (local + Forge SDK)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from protein_design_hub.core.installer import ToolInstaller


class ESM3Installer(ToolInstaller):
    """Installer for ESM3 (EvolutionaryScale SDK)."""

    name = "esm3"
    description = "ESM3 - Multimodal protein generation (sequence/structure/function)"

    def is_installed(self) -> bool:
        esm3_python = os.getenv("ESM3_PYTHON")
        if esm3_python and Path(esm3_python).exists():
            return True
        try:
            from esm.models.esm3 import ESM3  # noqa: F401
            from esm.sdk.api import ESMProtein  # noqa: F401
        except Exception:
            return False
        return True

    def get_installed_version(self) -> Optional[str]:
        return self._get_installed_pip_version("esm")

    def get_latest_version(self) -> Optional[str]:
        return self._get_pypi_version("esm")

    def install(self) -> bool:
        # Note: torch is intentionally not auto-installed here; users generally want
        # to install torch matching their CUDA setup.
        if self._get_installed_pip_version("fair-esm"):
            print(
                "Warning: 'fair-esm' is installed. The ESM3 SDK uses the 'esm' package name "
                "and will conflict with ESMFold in the same environment. Consider a separate env."
            )
        return self._pip_install("esm")

    def update(self) -> bool:
        return self._pip_install("esm", upgrade=True)
