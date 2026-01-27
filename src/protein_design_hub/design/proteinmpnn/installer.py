"""Installer for ProteinMPNN (git clone into tools_dir)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from protein_design_hub.core.config import get_settings
from protein_design_hub.core.installer import ToolInstaller


class ProteinMPNNInstaller(ToolInstaller):
    name = "proteinmpnn"
    description = "ProteinMPNN - fixed-backbone sequence design"

    def __init__(self):
        self.settings = get_settings()

    @property
    def install_dir(self) -> Path:
        return Path(self.settings.installation.tools_dir).expanduser() / "ProteinMPNN"

    @property
    def run_script(self) -> Path:
        return self.install_dir / "protein_mpnn_run.py"

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
        # Not stable to compute without hitting network / remotes; omit.
        return None

    def install(self) -> bool:
        git = shutil.which("git")
        if not git:
            raise RuntimeError("git not found (required to install ProteinMPNN)")

        self.install_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.install_dir.exists():
            return self.is_installed()

        repo = "https://github.com/dauparas/ProteinMPNN.git"
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
