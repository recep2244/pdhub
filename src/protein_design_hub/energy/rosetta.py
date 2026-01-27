"""Rosetta command-line wrappers (no PyRosetta required)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from protein_design_hub.core.exceptions import EvaluationError
from protein_design_hub.energy.paths import find_rosetta_executable


def run_score_jd2(
    pdb_path: Path,
    out_dir: Path,
    rosetta_home: Optional[Path] = None,
    extra_flags: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run Rosetta score_jd2 and parse total_score from the generated scorefile.
    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = find_rosetta_executable("score_jd2", rosetta_home=rosetta_home)
    if exe is None:
        raise EvaluationError("rosetta_score_jd2", "Rosetta score_jd2 executable not found")

    scorefile = out_dir / "score.sc"
    cmd = [
        str(exe),
        "-in:file:s",
        str(pdb_path),
        "-out:file:scorefile",
        str(scorefile),
        "-out:path:all",
        str(out_dir),
        "-mute",
        "all",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise EvaluationError("rosetta_score_jd2", (result.stderr or result.stdout).strip())

    return parse_rosetta_scorefile(scorefile)


def run_relax(
    pdb_path: Path,
    out_dir: Path,
    rosetta_home: Optional[Path] = None,
    nstruct: int = 1,
    extra_flags: Optional[List[str]] = None,
) -> Path:
    """Run Rosetta relax and return the best output PDB path (heuristic)."""
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exe = find_rosetta_executable("relax", rosetta_home=rosetta_home)
    if exe is None:
        raise EvaluationError("rosetta_relax", "Rosetta relax executable not found")

    cmd = [
        str(exe),
        "-in:file:s",
        str(pdb_path),
        "-nstruct",
        str(int(nstruct)),
        "-out:path:all",
        str(out_dir),
        "-out:file:scorefile",
        str(out_dir / "relax.sc"),
        "-mute",
        "all",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=8 * 3600)
    if result.returncode != 0:
        raise EvaluationError("rosetta_relax", (result.stderr or result.stdout).strip())

    # Heuristic: pick newest PDB in out_dir.
    pdbs = sorted(out_dir.glob("*.pdb"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdbs:
        raise EvaluationError("rosetta_relax", "Relax finished but produced no PDB outputs")
    return pdbs[0]


def run_cartesian_ddg(
    pdb_path: Path,
    out_dir: Path,
    mutations_file: Path,
    rosetta_home: Optional[Path] = None,
    extra_flags: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run Rosetta cartesian_ddg (requires a mutfile).

    Returns parsed ddG in Rosetta Energy Units (REU) if found.
    """
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mutations_file = Path(mutations_file)

    exe = find_rosetta_executable("cartesian_ddg", rosetta_home=rosetta_home)
    if exe is None:
        raise EvaluationError("rosetta_cartesian_ddg", "Rosetta cartesian_ddg executable not found")

    out_prefix = out_dir / "cart_ddg"
    cmd = [
        str(exe),
        "-in:file:s",
        str(pdb_path),
        "-ddg:mut_file",
        str(mutations_file),
        "-ddg:out",
        str(out_prefix),
        "-out:path:all",
        str(out_dir),
        "-mute",
        "all",
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=8 * 3600)
    if result.returncode != 0:
        raise EvaluationError("rosetta_cartesian_ddg", (result.stderr or result.stdout).strip())

    # Try to find a ddg output file in out_dir.
    ddg_files = sorted(out_dir.glob("*.ddg"), key=lambda p: p.stat().st_mtime, reverse=True)
    ddg = None
    if ddg_files:
        ddg = _parse_ddg_file(ddg_files[0])

    if ddg is None:
        ddg = _parse_ddg_from_text(result.stdout) or _parse_ddg_from_text(result.stderr)

    if ddg is None:
        raise EvaluationError("rosetta_cartesian_ddg", "Could not parse ddG from Rosetta output")

    return {"cartesian_ddg": float(ddg)}


def parse_rosetta_scorefile(scorefile: Path) -> Dict[str, float]:
    scorefile = Path(scorefile)
    if not scorefile.exists():
        raise EvaluationError("rosetta_score", f"Scorefile not found: {scorefile}")

    header = None
    best = None
    for line in scorefile.read_text(errors="ignore").splitlines():
        if not line.startswith("SCORE:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if parts[1] == "score":
            header = parts[1:]
            continue
        if header is None:
            continue
        values = parts[1:]
        row = dict(zip(header, values))
        if "total_score" in row:
            try:
                score = float(row["total_score"])
            except Exception:
                continue
            if best is None or score < best["total_score"]:
                best = {"total_score": score}
    if best is None:
        raise EvaluationError("rosetta_score", "Failed to parse total_score from scorefile")
    return best


def _parse_ddg_file(path: Path) -> Optional[float]:
    text = path.read_text(errors="ignore")
    # Heuristic: search for a float after "ddg" or "ddG".
    for line in text.splitlines():
        if "ddg" in line.lower():
            vals = [v for v in line.replace("=", " ").split() if _is_float(v)]
            if vals:
                try:
                    return float(vals[-1])
                except Exception:
                    continue
    # Fallback: last float in file
    floats = [float(v) for v in text.replace(",", " ").split() if _is_float(v)]
    return floats[-1] if floats else None


def _parse_ddg_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    for line in text.splitlines():
        if "ddg" in line.lower():
            vals = [v for v in line.replace("=", " ").split() if _is_float(v)]
            if vals:
                try:
                    return float(vals[-1])
                except Exception:
                    continue
    return None


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except Exception:
        return False
