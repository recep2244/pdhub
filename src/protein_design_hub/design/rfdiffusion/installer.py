"""Installer for RFdiffusion (git clone into tools_dir; models optional)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from protein_design_hub.core.config import get_settings
from protein_design_hub.core.installer import ToolInstaller


class RFDiffusionInstaller(ToolInstaller):
    name = "rfdiffusion"
    description = "RFdiffusion - diffusion-based backbone generation"

    def __init__(self):
        self.settings = get_settings()

    @property
    def install_dir(self) -> Path:
        return Path(self.settings.installation.tools_dir).expanduser() / "RFdiffusion"

    @property
    def run_script(self) -> Path:
        return self.install_dir / "scripts" / "run_inference.py"

    @property
    def download_models_script(self) -> Path:
        return self.install_dir / "scripts" / "download_models.sh"

    def is_installed(self) -> bool:
        return self.run_script.exists()

    def get_installed_version(self) -> Optional[str]:
        if not self.is_installed():
            return None
        git = shutil.which("git")
        if not git:
            return None
        try:
            result = subprocess.run(
                [git, "-C", str(self.install_dir), "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def get_latest_version(self) -> Optional[str]:
        return None

    def install(self) -> bool:
        git = shutil.which("git")
        if not git:
            raise RuntimeError("git not found (required to install RFdiffusion)")

        self.install_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.install_dir.exists():
            return self.is_installed()

        repo = "https://github.com/RosettaCommons/RFdiffusion.git"
        result = subprocess.run(
            [git, "clone", "--depth", "1", repo, str(self.install_dir)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        return result.returncode == 0 and self.is_installed()

    def update(self) -> bool:
        git = shutil.which("git")
        if not git:
            return False
        if not self.install_dir.exists():
            return self.install()
        result = subprocess.run(
            [git, "-C", str(self.install_dir), "pull", "--ff-only"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
