"""Tests for ESM-2 zero-shot + AlphaMissense variant-effect integration."""

import gzip
import pytest

from protein_design_hub.analysis.esm2_zero_shot import ESM2VariantScorer, verdict_for_delta
from protein_design_hub.analysis.alphamissense import AlphaMissenseLookup, build_index
from protein_design_hub.analysis.mutation_scoring import composite_mutation_score

_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"


def test_verdict_thresholds():
    assert verdict_for_delta(-8) == "strongly_deleterious"
    assert verdict_for_delta(-6) == "likely_deleterious"
    assert verdict_for_delta(-3) == "ambiguous"
    assert verdict_for_delta(0) == "tolerated"


@pytest.mark.skipif(not ESM2VariantScorer.is_available(), reason="fair-esm/torch not installed")
def test_esm2_masked_marginal_tiny_model():
    s = ESM2VariantScorer(model_name="esm2_t6_8M_UR50D", device="cpu")
    v = s.score_variant(_SEQ, 10, _SEQ[9], "P")
    assert isinstance(v["esm2_delta_ll"], float)
    pos = s.score_position(_SEQ, 10)
    assert len(pos) == 20
    assert pos[_SEQ[9].upper()] == 0.0          # WT delta is zero by definition
    # delta(mt) == score_position[mt] for the same position
    assert abs(v["esm2_delta_ll"] - pos["P"]) < 1e-3


def test_alphamissense_fallback_cache():
    am = AlphaMissenseLookup(index_path="/nonexistent/path.sqlite")
    assert am.is_available()                    # fallback cache present
    hit = am.score("P38398", "A1708E")
    assert hit and hit["am_score"] == 0.94 and hit["am_threshold"] == 0.564
    assert am.score("P38398", "Z9Z") is None    # not covered
    assert am.score(None, "A1E") is None        # no uniprot → N/A


def test_alphamissense_built_index(tmp_path):
    tsv = tmp_path / "am.tsv.gz"
    with gzip.open(tsv, "wt") as fh:
        fh.write("#AlphaMissense\n")
        fh.write("uniprot_id\tprotein_variant\tam_pathogenicity\tam_class\n")
        fh.write("Q99999\tA10G\t0.91\tlikely_pathogenic\n")
        fh.write("Q99999\tL20P\t0.12\tlikely_benign\n")
    idx = tmp_path / "am.sqlite"
    build_index(tsv, idx)
    am = AlphaMissenseLookup(index_path=idx)
    assert am.has_full_index()
    assert am.score("Q99999", "A10G")["am_score"] == pytest.approx(0.91)
    assert am.score("Q99999", "L20P")["am_class"] == "likely_benign"
    assert am.score("Q99999", "X1X") is None


def test_composite_includes_esm2_and_am_terms():
    mut = {
        "position": 5, "original_aa": "A", "mutant_aa": "V", "delta_mean_plddt": 1.0,
        "mean_plddt": 85.0, "ost_lddt": 0.97, "clash_score": 5, "extra_metrics": {},
        "esm2_delta_ll": -6.0, "esm2_verdict": "likely_deleterious",
        "am_score": 0.90, "am_threshold": 0.564,
    }
    score, comp = composite_mutation_score(mut, {"clash_score": 5}, _SEQ)
    assert comp["esm2_term"] < 0               # deleterious ESM-2 lowers score
    assert comp["am_term"] < 0                 # pathogenic AlphaMissense lowers score
    assert any("ESM-2" in f for f in comp["flags"])
    assert any("AlphaMissense" in f for f in comp["flags"])
