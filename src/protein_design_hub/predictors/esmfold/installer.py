"""ESMFold installers (local + API)."""

from __future__ import annotations

from typing import Optional

from protein_design_hub.core.installer import ToolInstaller


class ESMFoldInstaller(ToolInstaller):
    """Installer for local ESMFold (fair-esm + torch)."""

    name = "esmfold"
    description = "ESMFold - Fast single-sequence structure prediction"

    def is_installed(self) -> bool:
        try:
            import torch  # noqa: F401
            import esm  # noqa: F401
            if not hasattr(esm, "pretrained") or not hasattr(esm.pretrained, "esmfold_v1"):
                return False
        except Exception:
            return False
        return True

    def get_installed_version(self) -> Optional[str]:
        return self._get_installed_pip_version("fair-esm")

    def get_latest_version(self) -> Optional[str]:
        return self._get_pypi_version("fair-esm")

    def install(self) -> bool:
        # Note: torch is intentionally not auto-installed here; users generally want
        # to install torch matching their CUDA setup.
        return self._pip_install("fair-esm")

    def update(self) -> bool:
        return self._pip_install("fair-esm", upgrade=True)


class ESMFoldAPIInstaller(ToolInstaller):
    """Installer shim for the remote ESMFold API predictor."""

    name = "esmfold_api"
    description = "ESMFold API - Remote structure prediction"

    def is_installed(self) -> bool:
        try:
            import requests  # noqa: F401
        except Exception:
            return False
        return True

    def get_installed_version(self) -> Optional[str]:
        return None

    def get_latest_version(self) -> Optional[str]:
        return None

    def install(self) -> bool:
        return self._pip_install("requests")

    def update(self) -> bool:
        return self._pip_install("requests", upgrade=True)
