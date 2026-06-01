"""Tests for the physics-aware mutation scoring (analysis/mutation_scoring.py)."""

from protein_design_hub.analysis.mutation_scoring import (
    classify_score,
    composite_mutation_score,
    position_risk_summary,
    substitution_risk,
)

_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"


def test_fold_preserving_stabilising_is_beneficial():
    mut = {
        "position": 5, "original_aa": "A", "mutant_aa": "V",
        "delta_mean_plddt": 2.0, "mean_plddt": 85.0,
        "ost_lddt": 0.97, "clash_score": 5, "extra_metrics": {},
    }
    wt = {"clash_score": 5, "extra_metrics": {}}
    score, comp = composite_mutation_score(mut, wt, _SEQ)
    assert comp["fold_term"] > 0           # fold preserved → positive
    assert classify_score(score) == "beneficial"


def test_fold_disrupting_clashing_is_detrimental():
    mut = {
        "position": 28, "original_aa": "G", "mutant_aa": "W",
        "delta_mean_plddt": -8.0, "mean_plddt": 60.0,
        "ost_rmsd_ca": 4.2, "clash_score": 45, "extra_metrics": {},
    }
    wt = {"clash_score": 5, "extra_metrics": {}}
    score, comp = composite_mutation_score(mut, wt, _SEQ)
    assert comp["clash_penalty"] < 0
    assert any("removes G" in f for f in comp["flags"])
    assert classify_score(score) == "detrimental"


def test_low_confidence_fold_gate():
    mut = {
        "position": 5, "original_aa": "A", "mutant_aa": "S",
        "delta_mean_plddt": 0.0, "mean_plddt": 40.0,   # below MIN_VIABLE_PLDDT
        "ost_lddt": 0.95, "clash_score": 5, "extra_metrics": {},
    }
    score, comp = composite_mutation_score(mut, {"clash_score": 5}, _SEQ)
    assert comp["gate_ok"] is False
    assert any("low-confidence" in f for f in comp["flags"])


def test_pLDDT_alone_does_not_make_beneficial():
    # Big pLDDT gain but destabilising ΔΔG + fold drift should NOT be beneficial,
    # which the old 0.6·Δmean+0.4·Δlocal score would have rated highly.
    mut = {
        "position": 28, "original_aa": "G", "mutant_aa": "P",
        "delta_mean_plddt": +15.0, "mean_plddt": 88.0,
        "ost_rmsd_ca": 3.5, "clash_score": 30, "extra_metrics": {},
    }
    wt = {"clash_score": 5, "extra_metrics": {}}
    score, comp = composite_mutation_score(mut, wt, _SEQ)
    assert classify_score(score) != "beneficial"


def test_substitution_risk_flags_cys():
    r = substitution_risk("C", "S")
    assert r["risk"] > 0
    assert any("removes C" in f for f in r["flags"])


def test_position_risk_summary_flags_glycine():
    risks = position_risk_summary(_SEQ, [28])  # position 28 = G
    assert risks[28]["wt_aa"] == "G"
    assert risks[28]["risk"] > 0
