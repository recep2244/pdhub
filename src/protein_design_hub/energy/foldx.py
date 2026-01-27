"""FoldX wrapper (ΔΔG via BuildModel)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.energy.paths import find_foldx_executable


def run_foldx_buildmodel(
    pdb_path: Path,
    mutant_file: Path,
    out_dir: Path,
    foldx_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Run FoldX BuildModel. `mutant_file` should be in FoldX individual_list.txt format.

    Returns:
      {"foldx_ddg_kcal_mol": <float>}
    """
    pdb_path = Path(pdb_path)
    mutant_file = Path(mutant_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = foldx_path or find_foldx_executable()
    if exe is None:
        raise EvaluationError("foldx", "FoldX executable not found (set FOLDX_BIN or add to PATH)")

    # FoldX writes outputs into the working directory.
    cmd = [
        str(exe),
        "--command=BuildModel",
        f"--pdb={pdb_path.name}",
        f"--mutant-file={mutant_file.name}",
        "--numberOfRuns=1",
    ]

    # Copy inputs into working directory
    work_pdb = out_dir / pdb_path.name
    work_mut = out_dir / mutant_file.name
    work_pdb.write_bytes(pdb_path.read_bytes())
    work_mut.write_bytes(mutant_file.read_bytes())

    result = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise EvaluationError("foldx", (result.stderr or result.stdout).strip())

    ddg = _find_ddg_from_foldx_outputs(out_dir)
    if ddg is None:
        raise EvaluationError("foldx", "Could not parse ΔΔG from FoldX outputs")
    return {"foldx_ddg_kcal_mol": float(ddg)}


def _find_ddg_from_foldx_outputs(out_dir: Path) -> Optional[float]:
    # Common output: Dif_*.fxout
    candidates = sorted(out_dir.glob("Dif_*.fxout"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        ddg = _parse_dif_fxout(path)
        if ddg is not None:
            return ddg
    # Sometimes: Average_*.fxout
    candidates = sorted(
        out_dir.glob("Average_*.fxout"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for path in candidates:
        ddg = _parse_dif_fxout(path)
        if ddg is not None:
            return ddg
    return None


def _parse_dif_fxout(path: Path) -> Optional[float]:
    text = path.read_text(errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    # FoldX fxout is typically a table; ΔG is often the last column. We'll try to parse the first data row.
    for ln in lines[1:]:
        parts = ln.split("\t") if "\t" in ln else ln.split()
        floats = [p for p in parts if _is_float(p)]
        if floats:
            try:
                return float(floats[-1])
            except Exception:
                continue
    return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False
