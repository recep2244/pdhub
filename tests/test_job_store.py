"""Unit tests for the durable job store and out-of-process worker.

Covers (per Phase-5 blueprint):
  * enqueue -> claim -> complete round-trips and the queued/running/done/error
    lifecycle;
  * WAL journal mode is actually enabled on the database;
  * a concurrent read taken during an open write transaction does not raise;
  * ``run_worker(max_iter=N)`` drains the queue with a fake handler and exits
    cleanly.

All tests are pure-Python (Tier 1, ``unit``): no torch/GPU/network/binaries.
"""

from __future__ import annotations

import sqlite3

import pytest

from protein_design_hub.core import job_store
from protein_design_hub.services import worker

pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    """An initialised job-store database path."""
    path = tmp_path / "jobs.db"
    job_store.init_db(path)
    return path


# --- Round-trips ------------------------------------------------------------

def test_enqueue_claim_complete_roundtrip(db):
    jid = job_store.enqueue(db, "predict", {"seq": "MKT"})

    queued = job_store.get(db, jid)
    assert queued is not None
    assert queued["status"] == job_store.QUEUED
    assert queued["kind"] == "predict"
    assert queued["params"] == {"seq": "MKT"}
    assert queued["result"] is None

    claimed = job_store.claim_next(db)
    assert claimed is not None
    assert claimed["id"] == jid
    assert claimed["status"] == job_store.RUNNING
    # The persisted row reflects the claim.
    assert job_store.get(db, jid)["status"] == job_store.RUNNING

    job_store.complete(db, jid, {"plddt": 88.0})
    done = job_store.get(db, jid)
    assert done["status"] == job_store.DONE
    assert done["result"] == {"plddt": 88.0}


def test_claim_next_empty_returns_none(db):
    assert job_store.claim_next(db) is None


def test_claim_is_fifo_and_single(db):
    a = job_store.enqueue(db, "k", {}, now="2026-01-01T00:00:00+00:00")
    b = job_store.enqueue(db, "k", {}, now="2026-01-01T00:00:01+00:00")

    first = job_store.claim_next(db)
    second = job_store.claim_next(db)
    third = job_store.claim_next(db)

    assert first["id"] == a  # oldest first
    assert second["id"] == b
    assert third is None  # queue drained, only two existed


def test_fail_records_error(db):
    jid = job_store.enqueue(db, "predict", {})
    job_store.claim_next(db)
    job_store.fail(db, jid, "boom")

    job = job_store.get(db, jid)
    assert job["status"] == job_store.ERROR
    assert job["result"] == {"error": "boom"}


def test_get_missing_returns_none(db):
    assert job_store.get(db, "does-not-exist") is None


def test_list_jobs_orders_newest_first_and_limits(db):
    job_store.enqueue(db, "k", {}, now="2026-01-01T00:00:00+00:00")
    mid = job_store.enqueue(db, "k", {}, now="2026-01-01T00:00:01+00:00")
    newest = job_store.enqueue(db, "k", {}, now="2026-01-01T00:00:02+00:00")

    listed = job_store.list_jobs(db, limit=2)
    assert [j["id"] for j in listed] == [newest, mid]


def test_clock_indirection_used_when_now_omitted(db, monkeypatch):
    monkeypatch.setattr(job_store, "_clock", lambda: "FIXED-TS")
    jid = job_store.enqueue(db, "k", {})
    job = job_store.get(db, jid)
    assert job["created_at"] == "FIXED-TS"
    assert job["updated_at"] == "FIXED-TS"


# --- WAL + concurrency ------------------------------------------------------

def test_wal_mode_enabled(db):
    conn = sqlite3.connect(str(db))
    try:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    finally:
        conn.close()
    assert mode.lower() == "wal"


def test_concurrent_read_during_write_does_not_raise(db):
    """A reader on a second connection must see committed data without locking
    while a writer holds an open (uncommitted) transaction."""
    job_store.enqueue(db, "k", {"n": 1})

    writer = sqlite3.connect(str(db), isolation_level=None, timeout=5.0)
    reader = sqlite3.connect(str(db), timeout=5.0)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO jobs (id, kind, status, params, result, created_at, updated_at) "
            "VALUES ('x', 'k', 'queued', '{}', NULL, 't', 't')"
        )
        # Read concurrently while the write txn is still open — must not raise.
        rows = reader.execute("SELECT COUNT(*) FROM jobs").fetchall()
        assert rows[0][0] == 1  # reader sees last committed snapshot, not the open insert
        writer.execute("COMMIT")
        assert reader.execute("SELECT COUNT(*) FROM jobs").fetchall()[0][0] == 2
    finally:
        writer.close()
        reader.close()


# --- Worker -----------------------------------------------------------------

def test_run_worker_drains_queue_with_fake_handler(db):
    calls = []

    def fake_handler(params):
        calls.append(params)
        return {"echo": params, "ok": True}

    ids = [job_store.enqueue(db, "echo", {"i": i}) for i in range(3)]

    processed = worker.run_worker(db, {"echo": fake_handler}, max_iter=10)

    assert processed == 3
    assert len(calls) == 3
    for jid in ids:
        job = job_store.get(db, jid)
        assert job["status"] == job_store.DONE
        assert job["result"]["ok"] is True


def test_run_worker_marks_handler_exception_as_error(db):
    def boom(_params):
        raise RuntimeError("kaboom")

    jid = job_store.enqueue(db, "boom", {})
    processed = worker.run_worker(db, {"boom": boom}, max_iter=5)

    assert processed == 1
    job = job_store.get(db, jid)
    assert job["status"] == job_store.ERROR
    assert "kaboom" in job["result"]["error"]


def test_run_worker_unknown_kind_fails_job(db):
    jid = job_store.enqueue(db, "mystery", {})
    processed = worker.run_worker(db, {}, max_iter=5)

    assert processed == 1
    job = job_store.get(db, jid)
    assert job["status"] == job_store.ERROR
    assert "mystery" in job["result"]["error"]


def test_run_worker_bounded_empty_queue_exits_cleanly(db):
    # No jobs enqueued; bounded mode must return promptly without sleeping.
    processed = worker.run_worker(db, {}, poll=0.01, max_iter=3)
    assert processed == 0


def test_run_worker_non_dict_result_is_wrapped(db):
    jid = job_store.enqueue(db, "scalar", {})
    worker.run_worker(db, {"scalar": lambda p: 42}, max_iter=2)
    job = job_store.get(db, jid)
    assert job["status"] == job_store.DONE
    assert job["result"] == {"value": 42}
