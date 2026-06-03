"""Foldseek structural-homolog search.

Thin wrapper around the ``foldseek easy-search`` CLI to find structurally
similar proteins (remote homologs that sequence search misses) in PDB100 / AFDB
databases. Degrades gracefully when the binary or a target database is absent —
every entry point returns a status dict rather than raising.

A novelty use-case: searching a *designed* backbone against AFDB and getting few
/ low-TM hits means the design is structurally novel; many high-TM hits mean it
resembles known folds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

# Default report fields (TM-score + LDDT included for structural ranking).
_FORMAT = "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,alntmscore,lddt"
_FORMAT_FIELDS = _FORMAT.split(",")


def foldseek_executable() -> Optional[str]:
    """Path to the foldseek binary, honouring ``$FOLDSEEK_BIN``."""
    env = os.environ.get("FOLDSEEK_BIN")
    if env and Path(env).exists():
        return env
    return shutil.which("foldseek")


def is_available() -> bool:
    return foldseek_executable() is not None


def resolve_database(db: Optional[str] = None) -> Optional[str]:
    """Resolve a target DB path from arg or ``$FOLDSEEK_DB`` (no download)."""
    db = db or os.environ.get("FOLDSEEK_DB")
    if not db:
        return None
    p = Path(db)
    # foldseek DBs are a prefix with sibling files (db, db.dbtype, ...)
    return db if (p.exists() or Path(f"{db}.dbtype").exists()) else None


def search_structure(
    query_pdb: str | Path,
    target_db: Optional[str] = None,
    max_hits: int = 25,
    min_tm: float = 0.0,
    sensitivity: float = 9.5,
    timeout: int = 600,
) -> Dict:
    """Run ``foldseek easy-search`` and return parsed hits.

    Args:
        query_pdb: query structure (.pdb/.cif).
        target_db: foldseek DB prefix (or set ``$FOLDSEEK_DB``).
        max_hits: max alignments to keep.
        min_tm: drop hits below this alignment TM-score.
        sensitivity: foldseek ``-s`` (higher = more sensitive, slower).

    Returns dict: {status, hits[...], n_hits, ...} or {status: 'unavailable'|'error'}.
    """
    exe = foldseek_executable()
    if exe is None:
        return {"status": "unavailable",
                "error": "foldseek binary not found (set $FOLDSEEK_BIN or install foldseek)"}
    query = Path(query_pdb)
    if not query.exists():
        return {"status": "error", "error": f"query not found: {query}"}
    db = resolve_database(target_db)
    if db is None:
        return {"status": "unavailable",
                "error": "no target DB (set $FOLDSEEK_DB or pass target_db; e.g. pdb100/afdb50)"}

    with tempfile.TemporaryDirectory(prefix="foldseek_") as tmp:
        out_m8 = Path(tmp) / "result.m8"
        cmd = [
            exe, "easy-search", str(query), db, str(out_m8), str(Path(tmp) / "work"),
            "--format-output", _FORMAT,
            "-s", str(sensitivity),
            "--max-seqs", str(max(max_hits * 4, 100)),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": f"foldseek timed out after {timeout}s"}
        if proc.returncode != 0 or not out_m8.exists():
            return {"status": "error",
                    "error": (proc.stderr or proc.stdout or "foldseek failed").strip()[:500]}
        hits = _parse_m8(out_m8, min_tm=min_tm, limit=max_hits)

    novelty = _novelty_label(hits)
    return {
        "status": "SUCCESS",
        "query": query.name,
        "database": Path(db).name,
        "n_hits": len(hits),
        "hits": hits,
        "top_tm": hits[0]["alntmscore"] if hits else None,
        "novelty": novelty,
    }


def _parse_m8(path: Path, min_tm: float = 0.0, limit: int = 25) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text().splitlines():
        parts = line.rstrip("\n").split("\t")
        if len(parts) < len(_FORMAT_FIELDS):
            continue
        rec = dict(zip(_FORMAT_FIELDS, parts))
        for k in ("fident", "evalue", "bits", "alntmscore", "lddt"):
            try:
                rec[k] = float(rec[k])
            except (TypeError, ValueError):
                rec[k] = None
        for k in ("alnlen", "qstart", "qend", "tstart", "tend"):
            try:
                rec[k] = int(rec[k])
            except (TypeError, ValueError):
                rec[k] = None
        if (rec.get("alntmscore") or 0.0) >= min_tm:
            rows.append(rec)
    rows.sort(key=lambda r: (r.get("alntmscore") or 0.0), reverse=True)
    return rows[:limit]


def _novelty_label(hits: List[Dict]) -> str:
    if not hits:
        return "🟢 structurally novel — no significant structural hits"
    top = hits[0].get("alntmscore") or 0.0
    if top >= 0.8:
        return f"🔴 close known fold — top TM {top:.2f} ({hits[0]['target']})"
    if top >= 0.5:
        return f"🟡 related to known folds — top TM {top:.2f} ({hits[0]['target']})"
    return f"🟢 largely novel fold — best structural hit only TM {top:.2f}"
