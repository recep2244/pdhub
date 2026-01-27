"""ProteinMPNN designer wrapper."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import List

from protein_design_hub.core.exceptions import InstallationError
from protein_design_hub.core.installer import ToolInstaller
from protein_design_hub.core.types import Sequence
from protein_design_hub.design.base import BaseDesigner
from protein_design_hub.design.proteinmpnn.installer import ProteinMPNNInstaller
from protein_design_hub.design.registry import DesignerRegistry
from protein_design_hub.design.types import DesignInput, DesignResult


@DesignerRegistry.register("proteinmpnn")
class ProteinMPNNDesigner(BaseDesigner):
    name = "proteinmpnn"
    description = "ProteinMPNN - fixed-backbone sequence design"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._installer: ToolInstaller = ProteinMPNNInstaller()

    @property
    def installer(self) -> ToolInstaller:
        return self._installer

    def _design(self, input_data: DesignInput, output_dir: Path) -> DesignResult:
        pdb_path = Path(input_data.backbone_path)
        if not pdb_path.exists():
            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                success=False,
                error_message=f"Backbone file not found: {pdb_path}",
            )

        run_script = self._installer.run_script  # type: ignore[attr-defined]
        if not run_script.exists():
            raise InstallationError(self.name, "ProteinMPNN run script missing; reinstall designer")

        out_folder = output_dir / "proteinmpnn_out"
        out_folder.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(run_script),
            "--pdb_path",
            str(pdb_path),
            "--out_folder",
            str(out_folder),
            "--num_seq_per_target",
            str(int(input_data.num_sequences)),
            "--sampling_temp",
            str(float(input_data.temperature)),
        ]
        if input_data.seed is not None:
            cmd += ["--seed", str(int(input_data.seed))]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
            )
        except subprocess.TimeoutExpired:
            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                success=False,
                error_message="ProteinMPNN timed out",
            )

        if result.returncode != 0:
            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                success=False,
                error_message=(result.stderr or result.stdout or "ProteinMPNN failed").strip(),
                metadata={"stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]},
            )

        sequences = _find_and_parse_fasta(out_folder)
        if not sequences:
            return DesignResult(
                job_id=input_data.job_id,
                designer=self.name,
                success=False,
                error_message="No FASTA output found from ProteinMPNN",
                metadata={"stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]},
            )

        return DesignResult(
            job_id=input_data.job_id,
            designer=self.name,
            sequences=sequences,
            success=True,
            metadata={"out_folder": str(out_folder)},
        )


def _find_and_parse_fasta(out_folder: Path) -> List[Sequence]:
    fasta_files = list(out_folder.rglob("*.fa")) + list(out_folder.rglob("*.fasta"))
    fasta_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not fasta_files:
        return []

    fasta_path = fasta_files[0]
    text = fasta_path.read_text(errors="ignore")
    entries: List[Sequence] = []

    header = None
    seq_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None and seq_lines:
                entries.append(Sequence(id=header, sequence="".join(seq_lines)))
            header = line[1:].strip()[:200] or "designed"
            seq_lines = []
        else:
            seq_lines.append(line)

    if header is not None and seq_lines:
        entries.append(Sequence(id=header, sequence="".join(seq_lines)))

    return entries
