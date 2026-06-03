"""Tests for the RunManifest provenance model (Phase 5)."""

from __future__ import annotations

import json

from protein_design_hub.core.manifest import (
    Egress,
    InputProvenance,
    RunManifest,
)


def _make_manifest() -> RunManifest:
    return RunManifest(
        run_id="run-0001",
        created_at="2026-06-01T12:00:00Z",  # passed in, never datetime.now()
        track="design",
        inputs={
            "target": InputProvenance(
                source="examples/target.pdb",
                hash="sha256:deadbeef",
                kind="structure",
            ),
        },
        tools={"proteinmpnn": "1.0.1", "esm": "unknown"},
        params={"num_designs": 8, "temperature": 0.1},
        outputs=["out/design_0.pdb", "out/design_1.pdb"],
    )


def test_construct_minimal():
    """A manifest can be built with only the required fields."""
    m = RunManifest(run_id="r", created_at="2026-06-01T00:00:00Z", track="predict")
    assert m.run_id == "r"
    assert m.created_at == "2026-06-01T00:00:00Z"
    assert m.track == "predict"
    assert m.inputs == {}
    assert m.tools == {}
    assert m.outputs == []


def test_egress_defaults_local_only():
    """Egress must default to local-only with no destinations."""
    m = RunManifest(run_id="r", created_at="t", track="design")
    assert m.egress.enabled is False
    assert m.egress.destinations == []

    # Round-tripped through dict the default is preserved.
    assert m.to_dict()["egress"] == {"enabled": False, "destinations": []}


def test_to_json_is_valid_json():
    m = _make_manifest()
    data = json.loads(m.to_json())
    assert data["run_id"] == "run-0001"
    assert data["track"] == "design"
    assert data["inputs"]["target"]["hash"] == "sha256:deadbeef"
    assert data["tools"]["proteinmpnn"] == "1.0.1"
    assert data["params"]["num_designs"] == 8
    assert data["outputs"] == ["out/design_0.pdb", "out/design_1.pdb"]


def test_save_and_load_round_trip(tmp_path):
    """save() writes a file under the dir; load() reconstructs an equal manifest."""
    m = _make_manifest()
    path = m.save(tmp_path)

    assert path.exists()
    assert path.name == "run_manifest.json"
    assert path.parent == tmp_path

    loaded = RunManifest.load(path)
    assert loaded.to_dict() == m.to_dict()

    # Loading via the directory also works.
    loaded_from_dir = RunManifest.load(tmp_path)
    assert loaded_from_dir.to_dict() == m.to_dict()

    # Egress remains local-only after the round trip.
    assert loaded.egress.enabled is False
    assert loaded.egress.destinations == []


def test_egress_with_destinations_round_trips(tmp_path):
    m = RunManifest(
        run_id="r2",
        created_at="t",
        track="batch",
        egress=Egress(enabled=True, destinations=["s3://bucket/results"]),
    )
    loaded = RunManifest.load(m.save(tmp_path))
    assert loaded.egress.enabled is True
    assert loaded.egress.destinations == ["s3://bucket/results"]
