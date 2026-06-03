"""Graceful-degradation tests for external-tool wrappers.

When an external binary (FoldX, Rosetta CLI, Voronota) is ABSENT, or an optional
Python dependency (PyRosetta, OpenMM) is missing, the wrappers must NOT raise.
They must return a status dict ``{"status": "unavailable"|"error", "error": ...}``
(or, for ``run_foldx_repair`` specifically, that dict in place of a ``Path``).

Every test below forces the binary/dependency to look absent by monkeypatching
``shutil.which`` / the executable-finder, then asserts that calling the wrapper
raises nothing and yields a status.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Minimal single-residue PDB so file-existence guards pass without needing a
# real structure (the binary is absent, so it is never actually parsed).
_MINIMAL_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.251   2.390   0.000  1.00  0.00           O
TER
END
"""


def _is_status_dict(obj) -> bool:
    return isinstance(obj, dict) and obj.get("status") in ("unavailable", "error")


@pytest.fixture()
def pdb_file(tmp_path: Path) -> Path:
    p = tmp_path / "model.pdb"
    p.write_text(_MINIMAL_PDB)
    return p


@pytest.fixture()
def mutant_file(tmp_path: Path) -> Path:
    p = tmp_path / "individual_list.txt"
    p.write_text("AA1B;\n")
    return p


@pytest.fixture(autouse=True)
def no_binaries(monkeypatch):
    """Make every external binary look absent.

    - ``shutil.which`` -> None everywhere (covers Voronota + PATH lookups).
    - FoldX / Rosetta finders -> None (covers env-var and vendored-dir paths).
    """
    import shutil

    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: None)

    import protein_design_hub.energy.paths as paths

    monkeypatch.setattr(paths, "find_foldx_executable", lambda: None)
    monkeypatch.setattr(paths, "find_rosetta_executable", lambda *_a, **_k: None)

    # foldx.py imported the symbol directly; patch that reference too.
    import protein_design_hub.energy.foldx as foldx_mod

    monkeypatch.setattr(foldx_mod, "find_foldx_executable", lambda: None)

    # rosetta CLI wrapper imported the finder directly.
    import protein_design_hub.energy.rosetta as rosetta_mod

    monkeypatch.setattr(rosetta_mod, "find_rosetta_executable", lambda *_a, **_k: None)

    # Clear env vars that could resurrect a binary path.
    for var in ("FOLDX_BIN", "FOLDX", "ROSETTA3_HOME", "ROSETTA_HOME"):
        monkeypatch.delenv(var, raising=False)


# --------------------------------------------------------------------------- #
# FoldX low-level wrappers (energy/foldx.py)
# --------------------------------------------------------------------------- #


def test_foldx_repair_returns_status_when_binary_absent(pdb_file, tmp_path):
    from protein_design_hub.energy.foldx import run_foldx_repair

    out = run_foldx_repair(pdb_file, tmp_path / "repair")
    assert _is_status_dict(out)
    assert out["status"] == "unavailable"


def test_foldx_buildmodel_returns_status_when_binary_absent(pdb_file, mutant_file, tmp_path):
    from protein_design_hub.energy.foldx import run_foldx_buildmodel

    out = run_foldx_buildmodel(pdb_file, mutant_file, tmp_path / "bm")
    assert _is_status_dict(out)
    assert "foldx_ddg_kcal_mol" not in out


def test_foldx_analyse_complex_returns_status_when_binary_absent(pdb_file, tmp_path):
    from protein_design_hub.energy.foldx import run_foldx_analyse_complex

    out = run_foldx_analyse_complex(pdb_file, tmp_path / "ac", chains="A,B")
    assert _is_status_dict(out)


def test_foldx_stability_returns_status_when_binary_absent(pdb_file, tmp_path):
    from protein_design_hub.energy.foldx import run_foldx_stability

    out = run_foldx_stability(pdb_file, tmp_path / "st")
    assert _is_status_dict(out)


# --------------------------------------------------------------------------- #
# Metric wrappers (evaluation/metrics/*)
# --------------------------------------------------------------------------- #


def test_foldx_interface_metric_degrades(pdb_file):
    from protein_design_hub.evaluation.metrics.foldx_interface import FoldXInterfaceMetric

    metric = FoldXInterfaceMetric(chains="A,B")
    assert metric.is_available() is False
    out = metric.compute(pdb_file)
    assert _is_status_dict(out)


def test_rosetta_score_jd2_metric_degrades(pdb_file):
    from protein_design_hub.evaluation.metrics.rosetta_score_jd2 import RosettaScoreJd2Metric

    metric = RosettaScoreJd2Metric()
    assert metric.is_available() is False
    out = metric.compute(pdb_file)
    assert _is_status_dict(out)


def test_rosetta_energy_metric_degrades(pdb_file, monkeypatch):
    # Force PyRosetta to look uninstalled regardless of the host environment.
    from protein_design_hub.evaluation.metrics.rosetta_energy import RosettaEnergyMetric

    metric = RosettaEnergyMetric()
    monkeypatch.setattr(metric, "is_available", lambda: False)
    out = metric.compute(pdb_file)
    assert _is_status_dict(out)


def test_openmm_gbsa_metric_degrades(pdb_file, monkeypatch):
    from protein_design_hub.evaluation.metrics.openmm_gbsa import OpenMMGBSAMetric

    metric = OpenMMGBSAMetric()
    monkeypatch.setattr(metric, "is_available", lambda: False)
    out = metric.compute(pdb_file)
    assert _is_status_dict(out)
    assert out["status"] == "unavailable"


def test_voromqa_metric_degrades(pdb_file):
    from protein_design_hub.evaluation.metrics.voronota_voromqa import VoronotaVoroMQAMetric

    metric = VoronotaVoroMQAMetric()
    assert metric.is_available() is False
    out = metric.compute(pdb_file)
    assert _is_status_dict(out)


def test_cadscore_metric_degrades(pdb_file):
    from protein_design_hub.evaluation.metrics.voronota_cadscore import VoronotaCADScoreMetric

    metric = VoronotaCADScoreMetric()
    assert metric.is_available() is False
    # reference_path provided so we exercise the binary-absence path, not the
    # missing-reference guard.
    out = metric.compute(pdb_file, reference_path=pdb_file)
    assert _is_status_dict(out)


def test_no_wrapper_raises_smoke(pdb_file, mutant_file, tmp_path):
    """Belt-and-braces: invoking every wrapper must raise nothing at all."""
    from protein_design_hub.energy.foldx import (
        run_foldx_analyse_complex,
        run_foldx_buildmodel,
        run_foldx_repair,
        run_foldx_stability,
    )
    from protein_design_hub.evaluation.metrics.foldx_interface import FoldXInterfaceMetric
    from protein_design_hub.evaluation.metrics.rosetta_score_jd2 import RosettaScoreJd2Metric
    from protein_design_hub.evaluation.metrics.voronota_cadscore import VoronotaCADScoreMetric
    from protein_design_hub.evaluation.metrics.voronota_voromqa import VoronotaVoroMQAMetric

    calls = [
        lambda: run_foldx_repair(pdb_file, tmp_path / "r"),
        lambda: run_foldx_buildmodel(pdb_file, mutant_file, tmp_path / "b"),
        lambda: run_foldx_analyse_complex(pdb_file, tmp_path / "a"),
        lambda: run_foldx_stability(pdb_file, tmp_path / "s"),
        lambda: FoldXInterfaceMetric().compute(pdb_file),
        lambda: RosettaScoreJd2Metric().compute(pdb_file),
        lambda: VoronotaVoroMQAMetric().compute(pdb_file),
        lambda: VoronotaCADScoreMetric().compute(pdb_file, reference_path=pdb_file),
    ]
    for call in calls:
        result = call()  # must not raise
        assert _is_status_dict(result)
