"""RFdiffusion backbone generator wrapper (CLI-style hydra overrides)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List

from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.design.generators.base import BaseBackboneGenerator
from protein_design_hub.design.generators.registry import GeneratorRegistry
from protein_design_hub.design.generators.types import BackboneInput, BackboneResult
from protein_design_hub.design.rfdiffusion.installer import RFDiffusionInstaller


@GeneratorRegistry.register("rfdiffusion")
class RFDiffusionGenerator(BaseBackboneGenerator):
    name = "rfdiffusion"
    description = "RFdiffusion - diffusion-based backbone generation"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer: ToolInstaller = RFDiffusionInstaller()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _generate(self, input_data: BackboneInput, output_dir: Path) -> BackboneResult:
        run_script = self._installer.run_script  # type: ignore[attr-defined]

        python_exe = (
            input_data.python_executable
            or Path(os.environ.get("RFDIFFUSION_PYTHON", "")).expanduser()
            if os.environ.get("RFDIFFUSION_PYTHON")
            else None
        )
        python_cmd = str(python_exe) if python_exe and python_exe.exists() else sys.executable

        output_prefix = output_dir / "rfdiffusion"

        cmd: List[str] = [python_cmd, str(run_script)]
        if input_data.config_name:
            cmd += ["--config-name", input_data.config_name]

        cmd += [
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={int(input_data.num_designs)}",
        ]

        if input_data.input_pdb:
            cmd += [f"inference.input_pdb={Path(input_data.input_pdb)}"]

        if input_data.contigs:
            cmd += [f"contigmap.contigs={input_data.contigs}"]

        if input_data.ckpt_override_path:
            cmd += [f"inference.ckpt_override_path={Path(input_data.ckpt_override_path)}"]

        cmd += list(input_data.overrides or [])

        result = subprocess.run(
            cmd,
            cwd=str(output_dir),
            capture_output=True,
            text=True,
            timeout=24 * 3600,
        )

        if result.returncode != 0:
            return BackboneResult(
                job_id=input_data.job_id,
                generator=self.name,
                success=False,
                error_message=(result.stderr or result.stdout or "RFdiffusion failed").strip(),
                metadata={
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                    "cmd": cmd,
                },
            )

        pdbs = sorted(output_dir.glob("*.pdb")) + sorted(output_dir.rglob("*.pdb"))
        pdbs = sorted(set(pdbs), key=lambda p: p.stat().st_mtime, reverse=True)
        pdbs = pdbs[: int(input_data.num_designs) * 2 + 50]

        return BackboneResult(
            job_id=input_data.job_id,
            generator=self.name,
            backbone_paths=pdbs,
            success=True,
            metadata={
                "cmd": cmd,
                "output_dir": str(output_dir),
                "output_prefix": str(output_prefix),
            },
        )
