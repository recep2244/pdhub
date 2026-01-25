"""ColabFold (LocalColabFold) installer."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.config import get_settings


class ColabFoldInstaller(ToolInstaller):
    """Installer for LocalColabFold."""

    name = "colabfold"
    description = "LocalColabFold - AlphaFold2 with MMseqs2 for fast MSA generation"

    def __init__(self):
        self.settings = get_settings()
        self._colabfold_path: Optional[Path] = None

    def is_installed(self) -> bool:
        """Check if ColabFold is installed."""
        return self._find_colabfold_batch() is not None

    def _find_colabfold_batch(self) -> Optional[Path]:
        """Find the colabfold_batch executable."""
        # Check configured path first
        if self.settings.installation.colabfold_path:
            path = Path(self.settings.installation.colabfold_path)
            if path.exists():
                return path

        # Check PATH
        which_result = shutil.which("colabfold_batch")
        if which_result:
            return Path(which_result)

        # Check common installation locations
        common_paths = [
            Path.home() / "localcolabfold" / "colabfold-conda" / "bin" / "colabfold_batch",
            Path.home() / ".local" / "bin" / "colabfold_batch",
            Path("/usr/local/bin/colabfold_batch"),
            Path("/opt/localcolabfold/colabfold-conda/bin/colabfold_batch"),
        ]

        for path in common_paths:
            if path.exists():
                return path

        return None

    def get_executable_path(self) -> Optional[Path]:
        """Get the path to colabfold_batch executable."""
        if self._colabfold_path is None:
            self._colabfold_path = self._find_colabfold_batch()
        return self._colabfold_path

    def get_installed_version(self) -> Optional[str]:
        """Get the installed ColabFold version."""
        executable = self.get_executable_path()
        if not executable:
            return None

        try:
            result = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            # Parse version from output
            output = result.stdout.strip() or result.stderr.strip()
            # ColabFold version output varies, try to extract version number
            for line in output.split("\n"):
                if "colabfold" in line.lower() or "version" in line.lower():
                    # Try to find version pattern like "1.5.5"
                    import re
                    match = re.search(r"(\d+\.\d+\.\d+)", line)
                    if match:
                        return match.group(1)
            return output.split()[0] if output else None
        except Exception:
            return None

    def get_latest_version(self) -> Optional[str]:
        """Get the latest ColabFold version from GitHub."""
        try:
            import requests
            resp = requests.get(
                "https://api.github.com/repos/YoshitakaMo/localcolabfold/releases/latest",
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("tag_name", "").lstrip("v")
        except Exception:
            pass
        return None

    def install(self) -> bool:
        """
        Install LocalColabFold.

        This downloads and runs the official installation script.
        """
        print("Installing LocalColabFold...")
        print("This may take a while as it downloads model weights and dependencies.")

        # Create installation directory
        install_dir = Path(self.settings.installation.tools_dir).expanduser() / "localcolabfold"
        install_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download installation script
            script_url = "https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh"

            result = subprocess.run(
                ["wget", "-q", script_url, "-O", "install_colabbatch_linux.sh"],
                cwd=str(install_dir),
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"Failed to download installation script: {result.stderr}")
                return False

            # Run installation script
            result = subprocess.run(
                ["bash", "install_colabbatch_linux.sh"],
                cwd=str(install_dir),
                capture_output=False,  # Show output to user
            )

            if result.returncode != 0:
                print("Installation script failed")
                return False

            # Verify installation
            self._colabfold_path = None  # Reset cached path
            if self.is_installed():
                print(f"LocalColabFold installed successfully at {self.get_executable_path()}")
                return True
            else:
                print("Installation completed but colabfold_batch not found")
                return False

        except Exception as e:
            print(f"Installation failed: {e}")
            return False

    def update(self) -> bool:
        """Update LocalColabFold to the latest version."""
        executable = self.get_executable_path()
        if not executable:
            return self.install()

        # Find the update script in the installation directory
        install_dir = executable.parent.parent.parent
        update_script = install_dir / "update_colabbatch.sh"

        if update_script.exists():
            try:
                result = subprocess.run(
                    ["bash", str(update_script)],
                    capture_output=False,
                )
                return result.returncode == 0
            except Exception as e:
                print(f"Update failed: {e}")
                return False

        # If no update script, try reinstalling
        print("Update script not found, reinstalling...")
        return self.install()

    def setup_environment(self) -> dict:
        """
        Get environment variables needed to run ColabFold.

        Returns:
            Dictionary of environment variables.
        """
        executable = self.get_executable_path()
        if not executable:
            return {}

        # Add ColabFold bin directory to PATH
        bin_dir = executable.parent
        env = os.environ.copy()
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"

        return env
