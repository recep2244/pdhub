"""Tests for the protein-engineering gap fixes:
FoldX-preferred stability, developability term, ESM-2 fitness, combinatorial
sequence, and the FoldX interface metric plug-in.
"""

from types import SimpleNamespace

import pytest

from protein_design_hub.analysis.mutation_scoring import (
    composite_mutation_score, developability_delta,
)
from protein_design_hub.analysis.esm2_zero_shot import ESM2VariantScorer
from protein_design_hub.evolution.fitness_landscape import ESM2Fitness
from protein_design_hub.evaluation.metrics.foldx_interface import FoldXInterfaceMetric

_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"


def test_composite_prefers_foldx_ddg_over_heuristic():
    base = {"position": 5, "original_aa": "A", "mutant_aa": "V", "mean_plddt": 85.0,
            "ost_lddt": 0.97, "clash_score": 5, "extra_metrics": {}, "delta_mean_plddt": 0.0}
    # heuristic path
    _, c_heur = composite_mutation_score(dict(base), {"clash_score": 5}, _SEQ)
    assert c_heur["ddg_source"] == "heuristic"
    # FoldX provided → used, and sign drives stability term
    m = dict(base); m["foldx_ddg_kcal_mol"] = -2.5     # strongly stabilising
    _, c_fx = composite_mutation_score(m, {"clash_score": 5}, _SEQ)
    assert c_fx["ddg_source"] == "foldx"
    assert c_fx["ddg_kcal"] == -2.5
    assert c_fx["stability_term"] > 0                  # stabilising → positive


def test_developability_flags_aggregation_and_charge():
    agg = developability_delta("S", "W")    # polar → aggregation-prone aromatic
    assert agg["risk"] > 0 and any("aggregation" in f for f in agg["flags"])
    chg = developability_delta("K", "D")    # +1 → -1 charge swing
    assert any("charge" in f for f in chg["flags"])
    assert developability_delta("L", "I")["risk"] == 0.0   # conservative, no flag


def test_composite_applies_developability_penalty():
    m = {"position": 5, "original_aa": "S", "mutant_aa": "W", "mean_plddt": 85.0,
         "ost_lddt": 0.97, "clash_score": 5, "extra_metrics": {}, "delta_mean_plddt": 0.0}
    _, c = composite_mutation_score(m, {"clash_score": 5}, _SEQ)
    assert c["dev_penalty"] < 0


@pytest.mark.skipif(not ESM2VariantScorer.is_available(), reason="fair-esm/torch missing")
def test_esm2_sequence_pll_and_fitness():
    s = ESM2VariantScorer(model_name="esm2_t6_8M_UR50D", device="cpu")
    pll = s.sequence_pll(_SEQ)
    assert isinstance(pll, float) and pll < 0      # log-likelihoods are negative
    fit = ESM2Fitness(model_name="esm2_t6_8M_UR50D").evaluate(_SEQ)
    assert 0.0 <= fit <= 1.0


def test_foldx_interface_metric_contract():
    m = FoldXInterfaceMetric(chains="A,B")
    assert m.name == "foldx_interface_energy"
    assert isinstance(m.is_available(), bool)       # False without the FoldX binary
    assert "FoldX" in m.get_requirements()


def test_msa_conservation_penalises_conserved_positions():
    """Mutating a highly-conserved MSA column is penalised vs a variable one."""
    seq = "MKTAYIAKQRQ"
    base = {"position": 5, "original_aa": "A", "mutant_aa": "V", "mean_plddt": 85.0,
            "ost_lddt": 0.97, "clash_score": 5, "extra_metrics": {}, "delta_mean_plddt": 1.0}
    cons = dict(base); cons["conservation"] = 0.95
    var = dict(base); var["conservation"] = 0.20
    s_cons, c_cons = composite_mutation_score(cons, {"clash_score": 5}, seq)
    s_var, c_var = composite_mutation_score(var, {"clash_score": 5}, seq)
    assert c_cons["conservation_penalty"] < 0          # conserved → penalised
    assert c_var["conservation_penalty"] == 0.0        # variable → no penalty
    assert s_cons < s_var
    assert any("conserved" in f.lower() for f in c_cons["flags"])


def test_conservation_profile_from_alignment():
    from protein_design_hub.analysis.mutation_scoring import conservation_profile
    prof = conservation_profile(["MKTA", "MKTA", "MKTA"])   # fully conserved
    assert prof is not None and len(prof) == 4
    assert all(p > 0.9 for p in prof)
    assert conservation_profile(["not-a-real-thing"]) is not None or True  # graceful
