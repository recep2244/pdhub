"""Tests for the durable batch job handlers + job-store integration."""

import pytest

from protein_design_hub.services import batch_jobs


def test_run_biophysics_complete():
    r = batch_jobs.run_one("biophysics", {"sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGGIE"})
    assert r["status"] == "complete"
    assert r["result"]["mw"] > 0 and r["result"]["pi"] is not None
    assert "solubility_score" in r["result"]


def test_unknown_kind_fails_gracefully():
    r = batch_jobs.run_one("nope", {"sequence": "MKT"})
    assert r["status"] == "failed" and "not implemented" in r["error"]


def test_prediction_rejects_too_long_without_network():
    # >400 aa is rejected before any network call
    r = batch_jobs.run_prediction("A" * 500)
    assert r["status"] == "failed" and "too long" in r["error"]


def test_job_store_roundtrip_with_handler(tmp_path):
    from protein_design_hub.core import job_store
    db = tmp_path / "jobs.db"
    job_store.init_db(db)
    jid = job_store.enqueue(db, "biophysics", {"sequence": "MKTAYIAKQRQISFVK"})
    claimed = job_store.claim_next(db)
    assert claimed and claimed["id"] == jid and claimed["status"] == "running"
    res = batch_jobs.run_one(claimed["kind"], claimed["params"])
    job_store.complete(db, jid, res["result"])
    got = job_store.get(db, jid)
    assert got["status"] == "done" and got["result"]["mw"] > 0
