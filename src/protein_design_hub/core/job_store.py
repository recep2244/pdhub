"""Durable, process-shared job store backed by SQLite.

This module provides a tiny, dependency-free job table that can be safely
shared between a *reader* (the Streamlit UI) and a single *writer* (the
out-of-process worker in :mod:`protein_design_hub.services.worker`).

Design notes
------------
* The database is opened in **WAL** mode (``PRAGMA journal_mode=WAL``) so a
  reader and a single writer never block each other with a global lock — a
  read taken mid-write returns the last committed snapshot rather than
  raising ``database is locked``.
* No wall-clock call happens at import time. Timestamps are either passed in
  explicitly or produced by the module-level :data:`_clock` indirection, which
  tests may monkeypatch. The public functions accept an optional ``now``
  argument so callers stay deterministic.
* Status lifecycle: ``queued -> running -> {done, error}``. ``claim_next``
  atomically flips exactly one ``queued`` row to ``running`` (oldest first),
  which is what lets a single worker safely pull work.
* ``params`` and ``result`` are stored as JSON text columns.

The store is intentionally functional (module-level functions taking a
``db_path``) rather than a class, so the worker and the UI can each open their
own short-lived connection without sharing mutable state across threads.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "QUEUED",
    "RUNNING",
    "DONE",
    "ERROR",
    "STATUSES",
    "init_db",
    "enqueue",
    "claim_next",
    "complete",
    "fail",
    "get",
    "list_jobs",
]

# --- Status constants -------------------------------------------------------

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
STATUSES = (QUEUED, RUNNING, DONE, ERROR)

PathLike = Union[str, Path]


# --- Clock indirection ------------------------------------------------------
# Do NOT call datetime.now() at import time. ``_clock`` is invoked lazily and
# can be monkeypatched in tests for determinism.
def _clock() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _now(now: Optional[str]) -> str:
    """Resolve a timestamp: use the caller-supplied value or the module clock."""
    return now if now is not None else _clock()


# --- Connection helpers -----------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    status      TEXT NOT NULL,
    params      TEXT NOT NULL,
    result      TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
)
"""

_INDEX_STATUS = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_status_created "
    "ON jobs(status, created_at)"
)


def _connect(db_path: PathLike) -> sqlite3.Connection:
    """Open a WAL-mode connection to ``db_path``.

    ``isolation_level=None`` puts the connection in autocommit mode so each
    statement (and the explicit ``BEGIN IMMEDIATE`` used by :func:`claim_next`)
    controls its own transaction boundary. A modest ``busy_timeout`` lets a
    reader wait out the brief window a single writer holds the lock.
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a ``jobs`` row to a plain dict, decoding JSON columns."""
    return {
        "id": row["id"],
        "kind": row["kind"],
        "status": row["status"],
        "params": json.loads(row["params"]) if row["params"] else {},
        "result": json.loads(row["result"]) if row["result"] else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


# --- Public API -------------------------------------------------------------

def init_db(path: PathLike) -> None:
    """Create the ``jobs`` table (and index) at ``path`` if absent.

    Enables WAL mode as a side effect; safe to call repeatedly (idempotent).
    The parent directory is created if it does not exist.
    """
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect(p)
    try:
        conn.execute(_SCHEMA)
        conn.execute(_INDEX_STATUS)
    finally:
        conn.close()


def enqueue(
    path: PathLike,
    kind: str,
    params: Dict[str, Any],
    *,
    job_id: Optional[str] = None,
    now: Optional[str] = None,
) -> str:
    """Insert a new ``queued`` job and return its id.

    Args:
        path: SQLite database path (must already be initialised).
        kind: Handler key the worker dispatches on (e.g. ``"predict"``).
        params: JSON-serialisable parameters for the handler.
        job_id: Optional explicit id; a short uuid4 is generated otherwise.
        now: Optional ISO timestamp; the module clock is used if omitted.

    Returns:
        The id of the newly enqueued job.
    """
    jid = job_id or uuid.uuid4().hex[:12]
    ts = _now(now)
    conn = _connect(path)
    try:
        conn.execute(
            "INSERT INTO jobs (id, kind, status, params, result, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, NULL, ?, ?)",
            (jid, kind, QUEUED, json.dumps(params), ts, ts),
        )
    finally:
        conn.close()
    return jid


def claim_next(
    path: PathLike,
    *,
    now: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest ``queued`` job, flipping it to ``running``.

    Uses ``BEGIN IMMEDIATE`` so a single writer's select-then-update is
    serialised against other writers. Returns the claimed job dict (with its
    status already ``running``) or ``None`` when the queue is empty.
    """
    ts = _now(now)
    conn = _connect(path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ? "
                "ORDER BY created_at ASC, id ASC LIMIT 1",
                (QUEUED,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                (RUNNING, ts, row["id"]),
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

        job = _row_to_dict(row)
        job["status"] = RUNNING
        job["updated_at"] = ts
        return job
    finally:
        conn.close()


def complete(
    path: PathLike,
    job_id: str,
    result: Dict[str, Any],
    *,
    now: Optional[str] = None,
) -> None:
    """Mark ``job_id`` as ``done`` and persist its JSON ``result``."""
    ts = _now(now)
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE jobs SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (DONE, json.dumps(result), ts, job_id),
        )
    finally:
        conn.close()


def fail(
    path: PathLike,
    job_id: str,
    error: str,
    *,
    now: Optional[str] = None,
) -> None:
    """Mark ``job_id`` as ``error``, storing ``error`` under ``result.error``."""
    ts = _now(now)
    conn = _connect(path)
    try:
        conn.execute(
            "UPDATE jobs SET status = ?, result = ?, updated_at = ? WHERE id = ?",
            (ERROR, json.dumps({"error": str(error)}), ts, job_id),
        )
    finally:
        conn.close()


def get(path: PathLike, job_id: str) -> Optional[Dict[str, Any]]:
    """Return the job dict for ``job_id`` or ``None`` if it does not exist."""
    conn = _connect(path)
    try:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None
    finally:
        conn.close()


def list_jobs(path: PathLike, limit: int = 50) -> List[Dict[str, Any]]:
    """Return up to ``limit`` jobs, most recently created first."""
    conn = _connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()
