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


def find_foldx_executable() -> Optional[Path]:
    """
    FoldX is distributed as a binary and is not freely redistributable.
    We only locate it if already installed.
    """
    env = os.environ.get("FOLDX_BIN") or os.environ.get("FOLDX")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            for name in ("foldx", "FoldX", "foldxLinux64", "FoldXLinux64"):
                cand = p / name
                if cand.exists() and os.access(cand, os.X_OK):
                    return cand
        if p.exists() and os.access(p, os.X_OK):
            return p

    for name in ("foldx", "FoldX", "foldxLinux64", "FoldXLinux64"):
        p = which(name)
        if p:
            return p
    return None
