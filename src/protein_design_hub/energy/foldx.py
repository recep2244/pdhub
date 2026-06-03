"""FoldX wrapper (ΔΔG via BuildModel, binding ΔG via AnalyseComplex).

FoldX best practice is to run **RepairPDB** before BuildModel/AnalyseComplex so
the energies are not dominated by clashes/strain present in the raw (often
predicted) input structure. ``run_foldx_buildmodel`` repairs by default.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.energy.paths import find_foldx_executable, get_foldx_molecules_dir


def _stage_molecules(out_dir: Path) -> None:
    """FoldX 5 reads its rotamer library from ``molecules/`` in the working
    directory. Symlink (or copy) the library into ``out_dir`` so the wrappers,
    which run in ``out_dir``, can find it."""
    target = out_dir / "molecules"
    if target.exists():
        return
    mol = get_foldx_molecules_dir()
    if mol is None:
        return
    try:
        os.symlink(mol, target)
    except OSError:
        try:
            import shutil
            shutil.copytree(mol, target)
        except Exception:
            pass


def run_foldx_repair(
    pdb_path: Path,
    out_dir: Path,
    foldx_path: Optional[Path] = None,
) -> Path:
    """Run FoldX ``RepairPDB`` and return the repaired structure path.

    RepairPDB resolves clashes/bad torsions so downstream ΔΔG values are
    reliable. Output is ``<stem>_Repair.pdb`` in ``out_dir``.
    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = foldx_path or find_foldx_executable()
    if exe is None:
        raise EvaluationError("foldx", "FoldX executable not found (set FOLDX_BIN or add to PATH)")
    _stage_molecules(out_dir)

    work_pdb = out_dir / pdb_path.name
    if work_pdb.resolve() != pdb_path.resolve():
        work_pdb.write_bytes(pdb_path.read_bytes())

    cmd = [str(exe), "--command=RepairPDB", f"--pdb={work_pdb.name}"]
    result = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise EvaluationError("foldx", (result.stderr or result.stdout).strip())

    repaired = out_dir / f"{work_pdb.stem}_Repair.pdb"
    if not repaired.exists():
        raise EvaluationError("foldx", f"RepairPDB produced no output ({repaired.name})")
    return repaired


def run_foldx_buildmodel(
    pdb_path: Path,
    mutant_file: Path,
    out_dir: Path,
    foldx_path: Optional[Path] = None,
    repair: bool = True,
) -> Dict[str, float]:
    """
    Run FoldX BuildModel. `mutant_file` should be in FoldX individual_list.txt format.

    Args:
        repair: run RepairPDB on the input first (FoldX best practice; default True).

    Returns:
      {"foldx_ddg_kcal_mol": <float>, "foldx_repaired": <bool>}
    """
    pdb_path = Path(pdb_path)
    mutant_file = Path(mutant_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = foldx_path or find_foldx_executable()
    if exe is None:
        raise EvaluationError("foldx", "FoldX executable not found (set FOLDX_BIN or add to PATH)")

    # Copy inputs into working directory
    work_pdb = out_dir / pdb_path.name
    work_mut = out_dir / mutant_file.name
    work_pdb.write_bytes(pdb_path.read_bytes())
    work_mut.write_bytes(mutant_file.read_bytes())

    # RepairPDB first so ΔΔG isn't dominated by raw-model clashes.
    pdb_for_build = work_pdb.name
    if repair:
        repaired = run_foldx_repair(work_pdb, out_dir, exe)
        pdb_for_build = repaired.name

    # FoldX writes outputs into the working directory.
    cmd = [
        str(exe),
        "--command=BuildModel",
        f"--pdb={pdb_for_build}",
        f"--mutant-file={work_mut.name}",
        "--numberOfRuns=1",
    ]

    result = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise EvaluationError("foldx", (result.stderr or result.stdout).strip())

    ddg = _find_ddg_from_foldx_outputs(out_dir)
    if ddg is None:
        raise EvaluationError("foldx", "Could not parse ΔΔG from FoldX outputs")
    return {"foldx_ddg_kcal_mol": float(ddg), "foldx_repaired": bool(repair)}


def run_foldx_analyse_complex(
    pdb_path: Path,
    out_dir: Path,
    chains: Optional[str] = None,
    foldx_path: Optional[Path] = None,
    repair: bool = True,
) -> Dict[str, float]:
    """Run FoldX ``AnalyseComplex`` for interface/binding energy.

    This is what was missing for antibody/binder work — BuildModel only gives
    monomer stability. AnalyseComplex reports the interaction energy between
    chains (a proxy for binding ΔG).

    Args:
        chains: e.g. ``"A,B"`` to analyse the A/B interface. If None, FoldX
            analyses all chain pairs in the structure.
        repair: RepairPDB first (recommended).

    Returns:
      {"foldx_interaction_energy_kcal_mol": <float>}
    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = foldx_path or find_foldx_executable()
    if exe is None:
        raise EvaluationError("foldx", "FoldX executable not found (set FOLDX_BIN or add to PATH)")

    work_pdb = out_dir / pdb_path.name
    work_pdb.write_bytes(pdb_path.read_bytes())

    pdb_for_analysis = work_pdb.name
    if repair:
        repaired = run_foldx_repair(work_pdb, out_dir, exe)
        pdb_for_analysis = repaired.name

    cmd = [str(exe), "--command=AnalyseComplex", f"--pdb={pdb_for_analysis}"]
    if chains:
        cmd.append(f"--analyseComplexChains={chains}")

    result = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        raise EvaluationError("foldx", (result.stderr or result.stdout).strip())

    energy = _find_interaction_energy(out_dir)
    if energy is None:
        raise EvaluationError("foldx", "Could not parse interaction energy from AnalyseComplex outputs")
    return {"foldx_interaction_energy_kcal_mol": float(energy)}


def _find_interaction_energy(out_dir: Path) -> Optional[float]:
    """Parse interface interaction energy from AnalyseComplex Summary_*/Interaction_* files."""
    patterns = ("Summary_*.fxout", "Interaction_*.fxout", "Interaction_*_AC.fxout")
    for pat in patterns:
        for path in sorted(out_dir.glob(pat), key=lambda p: p.stat().st_mtime, reverse=True):
            val = _parse_interaction_fxout(path)
            if val is not None:
                return val
    return None


def _parse_interaction_fxout(path: Path) -> Optional[float]:
    text = path.read_text(errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    # Find the header row containing "Interaction Energy", then read that column
    # from the last data row.
    for hi, ln in enumerate(lines):
        low = ln.lower()
        if "interaction energy" in low:
            sep = "\t" if "\t" in ln else None
            headers = [h.strip().lower() for h in (ln.split(sep) if sep else ln.split("  "))]
            col = next((i for i, h in enumerate(headers) if "interaction energy" in h), None)
            for data in reversed(lines[hi + 1:]):
                parts = data.split(sep) if sep else data.split()
                if col is not None and col < len(parts) and _is_float(parts[col]):
                    return float(parts[col])
                floats = [p for p in parts if _is_float(p)]
                if floats:
                    return float(floats[-1])
    return None


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

    # FoldX Dif_ files have a header row then data rows.
    # The ΔΔG column is named "total energy" or "Stability" in the header.
    # Strategy: find the header, locate the ΔΔG column by name, then read it.
    _DDG_COLUMN_NAMES = {
        "total energy", "totalenergy", "stability", "ddg", "delta g",
        "deltag", "delta stability", "deltastability",
    }

    header_line = lines[0]
    sep = "\t" if "\t" in header_line else None
    headers = [h.strip().lower() for h in (header_line.split(sep) if sep else header_line.split())]

    ddg_col_idx: Optional[int] = None
    for i, h in enumerate(headers):
        if h in _DDG_COLUMN_NAMES:
            ddg_col_idx = i
            break

    for ln in lines[1:]:
        parts = ln.split(sep) if sep else ln.split()
        if ddg_col_idx is not None and ddg_col_idx < len(parts):
            try:
                return float(parts[ddg_col_idx].strip())
            except ValueError:
                continue
        # Fallback: try second numeric column (column 0 is usually the PDB name)
        floats_with_idx = [(i, p) for i, p in enumerate(parts) if _is_float(p)]
        if len(floats_with_idx) >= 1:
            # Prefer second float if available (first is often pdb ID index)
            target = floats_with_idx[1] if len(floats_with_idx) >= 2 else floats_with_idx[0]
            try:
                return float(target[1])
            except Exception:
                continue
    return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False


def run_foldx_stability(
    pdb_path: Path,
    out_dir: Path,
    foldx_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Run FoldX ``Stability`` (monomer total folding energy) on one structure.

    Mirrors the reference ``foldx.sh`` workflow. Returns the total energy parsed
    from the ``<stem>_0_ST.fxout`` report (column 2). Lower = more stable.

    Returns ``{"foldx_total_energy_kcal_mol": <float>}``.
    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = foldx_path or find_foldx_executable()
    if exe is None:
        raise EvaluationError("foldx", "FoldX executable not found (set FOLDX_BIN or add to PATH)")
    _stage_molecules(out_dir)

    work_pdb = out_dir / pdb_path.name
    work_pdb.write_bytes(pdb_path.read_bytes())

    cmd = [str(exe), "--command=Stability", f"--pdb={work_pdb.name}"]
    result = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True, timeout=3600)

    total = _find_stability_total(out_dir, work_pdb.stem)
    if total is None:
        raise EvaluationError(
            "foldx",
            (result.stderr or result.stdout or "Stability produced no _ST.fxout").strip()[:400],
        )
    return {"foldx_total_energy_kcal_mol": float(total)}


def _find_stability_total(out_dir: Path, stem: str) -> Optional[float]:
    """Parse the total folding energy (column 2) from a FoldX ``*_ST.fxout``."""
    candidates = sorted(out_dir.glob(f"{stem}*_ST.fxout"), key=lambda p: p.stat().st_mtime, reverse=True)
    candidates += sorted(out_dir.glob("*_ST.fxout"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            for line in path.read_text(errors="ignore").splitlines():
                parts = line.split("\t") if "\t" in line else line.split()
                # Row format: <pdb>\t<TotalEnergy>\t<...>
                if len(parts) >= 2 and _is_float(parts[1]):
                    return float(parts[1])
        except OSError:
            continue
    return None


def summarize_stability(values: List[float]) -> Dict[str, float]:
    """Descriptive stats over a set of FoldX stability energies (e.g. across a
    mutant library). Ports the reference ``foldx_extraction.py`` summary:
    mean/median/min/max, skewness, kurtosis, max-negative-gradient, overall slope.
    """
    import numpy as np
    arr = np.asarray([v for v in values if isinstance(v, (int, float))], dtype=float)
    if arr.size == 0:
        return {}
    out: Dict[str, float] = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
    if arr.size > 2:
        try:
            from scipy.stats import kurtosis, skew
            out["skewness"] = float(skew(arr))
            if arr.size > 3:
                out["kurtosis"] = float(kurtosis(arr))
        except Exception:
            pass
    diffs = np.diff(arr)
    if diffs.size:
        out["max_negative_gradient"] = float(np.min(diffs))
    if arr.size > 1:
        out["overall_slope"] = float((arr[-1] - arr[0]) / (arr.size - 1))
    return out
