"""AlphaMissense pathogenicity lookup (human canonical missense variants).

AlphaMissense (Cheng et al., Science 2023) ships **pre-computed** pathogenicity
scores for ~71 M human canonical missense variants — there is no inference, only
a lookup keyed by (UniProt accession, ``<WT><pos><MT>``). Scores are 0–1; the
released classification threshold is ~0.564 (≥ → likely_pathogenic).

Scope/caveats (surfaced to callers, never silently mis-applied):
  * **Human canonical proteins only.** Designed / non-human / non-canonical
    sequences return ``None`` → the caller treats AlphaMissense as N/A and falls
    back to the universal ESM-2 zero-shot signal.
  * It is a **pathogenicity** prior, NOT a stability (ΔΔG) measurement.
  * Non-commercial license — for research use.

Backends, in priority order:
  1. A built SQLite index (``pdhub install alphamissense``) over DeepMind's
     ``AlphaMissense_aa_substitutions.tsv.gz`` (~1 GB) — O(1) lookup, no RAM blowup.
  2. A small in-memory cache (the teaching-notebook fallback) so the feature is
     usable without the 1 GB download.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional

AM_THRESHOLD = 0.564

# Notebook-style fallback cache: {(uniprot, variant): {...}}. Extend freely.
_FALLBACK_CACHE: Dict[tuple, Dict] = {
    ("P38398", "A1708E"): {"am_score": 0.94, "am_class": "likely_pathogenic"},
}


def _default_index_path() -> Path:
    env = os.environ.get("PDHUB_ALPHAMISSENSE_INDEX")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "pdhub" / "alphamissense.sqlite"


class AlphaMissenseLookup:
    """Resolve (UniProt, variant) → AlphaMissense score/class, index or fallback."""

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = Path(index_path) if index_path else _default_index_path()
        self._conn: Optional[sqlite3.Connection] = None

    def is_available(self) -> bool:
        """True if either the full index OR the fallback cache can answer lookups."""
        return self.index_path.exists() or bool(_FALLBACK_CACHE)

    def has_full_index(self) -> bool:
        return self.index_path.exists()

    def _connect(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        if not self.index_path.exists():
            return None
        self._conn = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        return self._conn

    def score(self, uniprot_id: Optional[str], variant: str) -> Optional[Dict]:
        """Return ``{am_score, am_class, am_threshold}`` or ``None`` if not covered.

        ``variant`` is ``<WT><pos><MT>`` (e.g. ``"A1708E"``). ``None`` uniprot
        (e.g. a designed protein) → ``None`` (not applicable).
        """
        if not uniprot_id or not variant:
            return None
        conn = self._connect()
        if conn is not None:
            row = conn.execute(
                "SELECT score, class FROM am WHERE uniprot=? AND variant=?",
                (uniprot_id, variant),
            ).fetchone()
            if row:
                return {"am_score": float(row[0]), "am_class": row[1], "am_threshold": AM_THRESHOLD}
            return None
        hit = _FALLBACK_CACHE.get((uniprot_id, variant))
        if hit:
            return {**hit, "am_threshold": AM_THRESHOLD}
        return None


# ── process singleton ──────────────────────────────────────────────────────
_lookup: Optional[AlphaMissenseLookup] = None


def get_alphamissense(index_path: Optional[Path] = None) -> AlphaMissenseLookup:
    global _lookup
    if _lookup is None or index_path is not None:
        _lookup = AlphaMissenseLookup(index_path)
    return _lookup


def build_index(tsv_gz_path: Path, index_path: Optional[Path] = None) -> Path:
    """Build the SQLite index from DeepMind's AlphaMissense substitutions TSV(.gz).

    Expected columns include: uniprot_id, protein_variant, am_pathogenicity, am_class.
    Used by ``pdhub install alphamissense``.
    """
    import csv
    import gzip

    out = Path(index_path) if index_path else _default_index_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    conn = sqlite3.connect(out)
    conn.execute("CREATE TABLE am (uniprot TEXT, variant TEXT, score REAL, class TEXT)")
    opener = gzip.open if str(tsv_gz_path).endswith(".gz") else open
    n = 0
    with opener(tsv_gz_path, "rt") as fh:
        # Skip leading comment lines (the file starts with '#' headers).
        rows = (ln for ln in fh if not ln.startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t")
        batch = []
        for r in reader:
            uni = r.get("uniprot_id") or r.get("uniprot")
            var = r.get("protein_variant") or r.get("variant")
            sc = r.get("am_pathogenicity") or r.get("score")
            cls = r.get("am_class") or r.get("class")
            if not (uni and var and sc):
                continue
            batch.append((uni, var, float(sc), cls))
            if len(batch) >= 50000:
                conn.executemany("INSERT INTO am VALUES (?,?,?,?)", batch)
                n += len(batch); batch = []
        if batch:
            conn.executemany("INSERT INTO am VALUES (?,?,?,?)", batch); n += len(batch)
    conn.execute("CREATE INDEX idx_am ON am (uniprot, variant)")
    conn.commit()
    conn.close()
    return out
