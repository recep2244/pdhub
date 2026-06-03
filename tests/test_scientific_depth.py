"""Tests for Track-2 scientific-depth additions: ipSAE, protein-qc, Foldseek."""

import numpy as np
import pytest

from protein_design_hub.evaluation import ipsae
from protein_design_hub.analysis import protein_qc, foldseek_search


# --------------------------------------------------------------------------- #
# ipSAE
# --------------------------------------------------------------------------- #
def _block_pae(n_a, n_b, intra=2.0, inter=2.0):
    """Build a PAE matrix with given intra-chain and inter-chain error."""
    n = n_a + n_b
    pae = np.full((n, n), intra, dtype=float)
    pae[:n_a, n_a:] = inter
    pae[n_a:, :n_a] = inter
    return pae


def test_ipsae_high_for_confident_interface():
    # realistic interface size (larger N → larger d0); very low inter-chain PAE
    pae = _block_pae(60, 55, intra=0.4, inter=0.4)
    r = ipsae.compute_ipsae(pae, [60, 55], pae_cutoff=10.0)
    assert r["status"] == "SUCCESS"
    assert r["ipsae_min"] > 0.7 and r["passed"]
    assert "high-confidence" in ipsae.verdict(r) or "likely" in ipsae.verdict(r)


def test_ipsae_low_for_no_interface():
    pae = _block_pae(20, 18, intra=1.0, inter=28.0)    # inter-chain PAE above cutoff
    r = ipsae.compute_ipsae(pae, [20, 18], pae_cutoff=10.0)
    assert r["status"] == "SUCCESS"
    assert r["ipsae_min"] < 0.4 and not r["passed"]


def test_ipsae_single_chain_unavailable():
    pae = np.ones((10, 10))
    r = ipsae.compute_ipsae(pae, [10], pae_cutoff=10.0)
    assert r["status"] == "unavailable"
    assert "unavailable" in ipsae.verdict(r)


def test_ipsae_pdockq_with_coords():
    pae = _block_pae(15, 15, intra=1.0, inter=1.0)
    # two chains placed so interface CAs are within 8 Å
    coords = np.zeros((30, 3))
    coords[:15, 0] = np.arange(15) * 3.0
    coords[15:, 0] = np.arange(15) * 3.0
    coords[15:, 1] = 5.0
    plddt = np.full(30, 92.0)
    r = ipsae.compute_ipsae(pae, [15, 15], plddt=plddt, ca_coords=coords)
    assert r["pdockq"] is not None and 0.0 <= r["pdockq"] <= 1.0


# --------------------------------------------------------------------------- #
# protein-qc
# --------------------------------------------------------------------------- #
def test_sequence_liabilities_flags():
    liab = protein_qc.sequence_liabilities("MKTAYIAKQRCNGSKKKRLLLLLLW")
    assert liab["cysteine_count"] == 1 and not liab["cysteine_even"]   # odd Cys
    assert liab["deamidation_sites"] >= 1                               # NG/NS
    assert liab["polybasic_runs"] >= 1                                  # KKKR
    assert liab["hydrophobic_runs"] >= 1                               # LLLLLL...
    assert liab["risk"] > 0 and liab["flags"]


def test_biophysical_properties_basic():
    bp = protein_qc.biophysical_properties("MKTAYIAKQRQISFVKSHFSRQLEERLGG")
    assert bp["length"] > 0
    assert bp["gravy"] is not None and bp["pi"] is not None and bp["mw"] > 0


def test_composite_score_renormalises_on_missing():
    full = protein_qc.composite_score(
        {"plddt": 90, "iptm": 0.8, "pae_interaction": 6, "shape_complementarity": 0.7,
         "esm2_pll_normalized": 0.3})
    assert full is not None and 0.0 <= full <= 1.0
    partial = protein_qc.composite_score({"plddt": 90})     # one term only
    assert partial is not None and partial > 0.8            # high pLDDT dominates
    assert protein_qc.composite_score({}) is None


def test_assess_design_pass_and_fail():
    good = protein_qc.assess_design(
        {"plddt": 92, "iptm": 0.7, "pae_interaction": 8, "sc_rmsd": 1.2,
         "esm2_pll_normalized": 0.3, "instability": 28},
        sequence="MKTAYIAKQRQISFVKSHFSRQLEERLGGIE", level="standard")
    assert good.passed and "Passes QC" in good.verdict
    bad = protein_qc.assess_design(
        {"plddt": 60, "iptm": 0.2, "pae_interaction": 25, "sc_rmsd": 5.0},
        level="standard")
    assert not bad.passed and "Fails QC" in bad.verdict


def test_filter_designs_pipeline_counts():
    rows = [
        {"id": "good", "plddt": 92, "sc_rmsd": 1.0, "iptm": 0.7, "pae_interaction": 8,
         "esm2_pll_normalized": 0.2, "instability": 25, "sequence": "MKTAYIAKQRQ"},
        {"id": "bad_struct", "plddt": 40, "sc_rmsd": 1.0, "iptm": 0.7, "pae_interaction": 8,
         "esm2_pll_normalized": 0.2, "instability": 25, "sequence": "MKTAYIAKQRQ"},
        {"id": "odd_cys", "plddt": 92, "sc_rmsd": 1.0, "iptm": 0.7, "pae_interaction": 8,
         "esm2_pll_normalized": 0.2, "instability": 25, "sequence": "MKTAYIAKQRC"},
    ]
    res = protein_qc.filter_designs(rows, level="standard")
    ids = {r["id"] for r in res["survivors"]}
    assert "good" in ids and "bad_struct" not in ids and "odd_cys" not in ids
    assert res["counts"]["input"] == 3


# --------------------------------------------------------------------------- #
# Foldseek (binary usually absent in CI → must degrade gracefully)
# --------------------------------------------------------------------------- #
def test_foldseek_graceful_without_binary_or_db():
    assert isinstance(foldseek_search.is_available(), bool)
    res = foldseek_search.search_structure("/nonexistent/q.pdb", target_db=None)
    assert res["status"] in ("unavailable", "error")
    assert "error" in res
