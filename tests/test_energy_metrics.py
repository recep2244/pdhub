from __future__ import annotations

from pathlib import Path

from protein_design_hub.evaluation.composite import CompositeEvaluator
from protein_design_hub.evaluation.metrics.clash_score import ClashScoreMetric
from protein_design_hub.evaluation.metrics.contact_energy import ContactEnergyMetric
from protein_design_hub.evaluation.metrics.sasa import SASAMetric
from protein_design_hub.evaluation.metrics.interface_bsa import InterfaceBSAMetric
from protein_design_hub.evaluation.metrics.salt_bridges import SaltBridgeMetric


MINI_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00 10.00           C
ATOM      3  C   ALA A   1       1.500   0.000   0.000  1.00 10.00           C
ATOM      4  O   ALA A   1       2.200   0.000   0.000  1.00 10.00           O
ATOM      5  N   ALA A   2       0.500   0.000   0.000  1.00 10.00           N
ATOM      6  CA  ALA A   2       0.500   0.000   0.000  1.00 10.00           C
ATOM      7  C   ALA A   2       2.000   0.000   0.000  1.00 10.00           C
ATOM      8  O   ALA A   2       2.700   0.000   0.000  1.00 10.00           O
TER
END
"""


def test_clash_score_metric(tmp_path: Path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)

    metric = ClashScoreMetric(cutoff_angstrom=2.0)
    result = metric.compute(pdb)

    assert "clash_score" in result
    assert "clash_count" in result
    assert result["clash_count"] >= 1
    assert result["clash_score"] is not None


def test_contact_energy_metric(tmp_path: Path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)

    metric = ContactEnergyMetric(contact_cutoff=8.0, min_seq_separation=0)
    result = metric.compute(pdb)

    assert "contact_energy" in result
    assert "contact_count" in result
    assert result["contact_count"] >= 1
    assert isinstance(result["contact_energy"], float)


def test_composite_evaluator_wires_energy_fields(tmp_path: Path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)

    evaluator = CompositeEvaluator(metrics=["clash_score", "contact_energy", "sasa"])
    res = evaluator.evaluate(pdb, reference_path=None)

    assert res.clash_score is not None
    assert res.contact_energy is not None
    assert res.sasa_total is not None


def test_sasa_metric(tmp_path: Path):
    pdb = tmp_path / "mini.pdb"
    pdb.write_text(MINI_PDB)

    metric = SASAMetric()
    result = metric.compute(pdb)

    assert result["sasa_total"] > 0


def test_interface_bsa_and_salt_bridges(tmp_path: Path):
    # Minimal 2-chain PDB with one Asp-Lys pair near each other
    pdb_text = """\
ATOM      1  N   ASP A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ASP A   1       0.000   0.000   0.000  1.00 10.00           C
ATOM      3  CB  ASP A   1       0.800   0.000   0.000  1.00 10.00           C
ATOM      4  CG  ASP A   1       1.600   0.000   0.000  1.00 10.00           C
ATOM      5  OD1 ASP A   1       2.400   0.000   0.000  1.00 10.00           O
ATOM      6  OD2 ASP A   1       1.600   0.800   0.000  1.00 10.00           O
TER
ATOM      7  N   LYS B   1       3.000   0.000   0.000  1.00 10.00           N
ATOM      8  CA  LYS B   1       3.000   0.000   0.000  1.00 10.00           C
ATOM      9  CB  LYS B   1       3.800   0.000   0.000  1.00 10.00           C
ATOM     10  CG  LYS B   1       4.600   0.000   0.000  1.00 10.00           C
ATOM     11  CD  LYS B   1       5.400   0.000   0.000  1.00 10.00           C
ATOM     12  CE  LYS B   1       6.200   0.000   0.000  1.00 10.00           C
ATOM     13  NZ  LYS B   1       2.900   0.000   0.000  1.00 10.00           N
TER
END
"""
    pdb = tmp_path / "twochain.pdb"
    pdb.write_text(pdb_text)

    sb = SaltBridgeMetric(cutoff_angstrom=4.0)
    sb_res = sb.compute(pdb)
    assert sb_res["salt_bridge_count"] >= 1
    assert sb_res["salt_bridge_count_interchain"] >= 1

    bsa = InterfaceBSAMetric()
    bsa_res = bsa.compute(pdb)
    assert "interface_bsa_total" in bsa_res
