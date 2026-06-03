"""Out-of-process job worker.

Runs a simple claim/dispatch/complete loop against the durable job store in
:mod:`protein_design_hub.core.job_store`. The worker has **no Streamlit
dependency** — it can be launched as a standalone process alongside the web
app::

    python -m protein_design_hub.services.worker /path/to/jobs.db

Each loop iteration claims the oldest ``queued`` job (atomically flipping it to
``running``), looks up a handler by the job's ``kind``, runs it, and records
the outcome via :func:`~protein_design_hub.core.job_store.complete` or
:func:`~protein_design_hub.core.job_store.fail`. Handlers are plain callables
``handler(params: dict) -> dict`` registered in the ``handlers`` mapping.

The loop is cooperative: pass ``max_iter`` to bound the number of iterations
(used by tests to drain a queue and exit cleanly).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Union

from protein_design_hub.core import job_store

__all__ = ["Handler", "run_worker", "main"]

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: A job handler maps a job's ``params`` dict to a JSON-serialisable result.
Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


def run_worker(
    db_path: PathLike,
    handlers: Mapping[str, Handler],
    poll: float = 1.0,
    max_iter: Optional[int] = None,
) -> int:
    """Claim and dispatch queued jobs until the queue drains or ``max_iter``.

    Args:
        db_path: Path to the SQLite job store (initialised if needed).
        handlers: Mapping of job ``kind`` -> callable ``handler(params)->dict``.
        poll: Seconds to sleep when the queue is empty before re-polling. When
            ``max_iter`` is set, an empty queue ends the loop immediately
            instead of sleeping, so tests never block.
        max_iter: Optional cap on loop iterations. ``None`` runs forever; a
            finite value lets the worker exit cleanly (used by tests).

    Returns:
        The number of jobs processed (whether they succeeded or failed).
    """
    job_store.init_db(db_path)

    processed = 0
    iterations = 0
    bounded = max_iter is not None

    while not bounded or iterations < max_iter:
        iterations += 1
        job = job_store.claim_next(db_path)

        if job is None:
            # Nothing to do. In bounded (test) mode we stop as soon as the
            # queue is empty so we never sleep/hang; otherwise back off.
            if bounded:
                break
            time.sleep(poll)
            continue

        _dispatch(db_path, job, handlers)
        processed += 1

    return processed


def _dispatch(
    db_path: PathLike,
    job: Dict[str, Any],
    handlers: Mapping[str, Handler],
) -> None:
    """Run one claimed job and record its outcome on the store."""
    job_id = job["id"]
    kind = job["kind"]
    handler = handlers.get(kind)

    if handler is None:
        msg = f"no handler registered for kind={kind!r}"
        logger.error("[worker] job %s: %s", job_id, msg)
        job_store.fail(db_path, job_id, msg)
        return

    try:
        result = handler(job.get("params", {}))
        if not isinstance(result, dict):
            result = {"value": result}
        job_store.complete(db_path, job_id, result)
        logger.info("[worker] job %s (%s) done", job_id, kind)
    except Exception as exc:  # noqa: BLE001 — any handler error becomes a failed job
        logger.exception("[worker] job %s (%s) failed", job_id, kind)
        job_store.fail(db_path, job_id, str(exc))


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point: ``python -m protein_design_hub.services.worker DB``.

    With no registered handlers this still starts and idles, which is enough to
    verify wiring; real deployments import :func:`run_worker` with a populated
    ``handlers`` mapping. Returns a process exit code.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="protein_design_hub.services.worker",
        description="Out-of-process job worker for the Protein Design Hub.",
    )
    parser.add_argument("db_path", help="Path to the SQLite job store.")
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Seconds to sleep when the queue is empty (default: 1.0).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Optional cap on loop iterations (default: run forever).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # No handlers are wired in here by default — deployments register their own.
    handlers: Dict[str, Handler] = {}
    run_worker(
        args.db_path,
        handlers,
        poll=args.poll,
        max_iter=args.max_iter,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/CLI
    raise SystemExit(main())
