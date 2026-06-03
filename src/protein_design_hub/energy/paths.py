"""Helpers to locate external tools on disk."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional


def which(executable: str) -> Optional[Path]:
    p = shutil.which(executable)
    return Path(p) if p else None


def get_rosetta_home() -> Optional[Path]:
    env = os.environ.get("ROSETTA3_HOME") or os.environ.get("ROSETTA_HOME")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p
    return None


def get_rosetta_bin_dir(rosetta_home: Optional[Path] = None) -> Optional[Path]:
    rosetta_home = rosetta_home or get_rosetta_home()
    if rosetta_home is None:
        return None
    # Standard layout: main/source/bin
    bin_dir = rosetta_home / "main" / "source" / "bin"
    if bin_dir.exists():
        return bin_dir
    # Alternate: source/bin
    bin_dir = rosetta_home / "source" / "bin"
    if bin_dir.exists():
        return bin_dir
    return None


def find_rosetta_executable(
    stem: str, rosetta_home: Optional[Path] = None, bin_dir: Optional[Path] = None
) -> Optional[Path]:
    """
    Find a Rosetta executable by stem.

    Examples:
      stem="relax" -> relax.linuxgccrelease (or other platform suffix)
    """
    bin_dir = bin_dir or get_rosetta_bin_dir(rosetta_home)
    if bin_dir is None or not bin_dir.exists():
        return None

    candidates = sorted(bin_dir.glob(f"{stem}.*"))
    # Prefer linuxgccrelease if present.
    for p in candidates:
        if p.name.endswith("linuxgccrelease") and os.access(p, os.X_OK):
            return p
    for p in candidates:
        if os.access(p, os.X_OK):
            return p
    return None


_FOLDX_NAMES = ("foldx", "FoldX", "foldxLinux64", "FoldXLinux64")
# FoldX 5.x ships a dated binary (e.g. foldx_20251231); match those too.
_FOLDX_GLOBS = ("foldx_*", "foldx5*", "foldx")


def _project_foldx_dir() -> Path:
    """Vendored FoldX directory inside the project (``<repo>/tools/foldx``)."""
    return Path(__file__).resolve().parents[3] / "tools" / "foldx"


def _scan_dir_for_foldx(d: Path) -> Optional[Path]:
    if not d.is_dir():
        return None
    for name in _FOLDX_NAMES:
        cand = d / name
        if cand.exists() and os.access(cand, os.X_OK):
            return cand
    for pat in _FOLDX_GLOBS:
        for cand in sorted(d.glob(pat)):
            if cand.is_file() and os.access(cand, os.X_OK):
                return cand
    return None


def find_foldx_executable() -> Optional[Path]:
    """Locate the FoldX binary.

    Search order: ``$FOLDX_BIN``/``$FOLDX`` (file or dir) → project-vendored
    ``tools/foldx`` → PATH. Matches both classic names and FoldX 5's dated
    binaries (``foldx_YYYYMMDD``).
    """
    env = os.environ.get("FOLDX_BIN") or os.environ.get("FOLDX")
    if env:
        p = Path(env).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return p
        hit = _scan_dir_for_foldx(p)
        if hit:
            return hit

    hit = _scan_dir_for_foldx(_project_foldx_dir())
    if hit:
        return hit

    for name in _FOLDX_NAMES:
        p = which(name)
        if p:
            return p
    return None


def get_foldx_molecules_dir() -> Optional[Path]:
    """Return the FoldX ``molecules/`` library dir (needed in the run cwd).

    Looks next to the resolved binary, then in the vendored project dir.
    """
    exe = find_foldx_executable()
    for base in [exe.parent if exe else None, _project_foldx_dir()]:
        if base and (base / "molecules").is_dir():
            return base / "molecules"
    return None
