"""Tests for MHC-II anchor analysis + deimmunisation suggestions."""

from protein_design_hub.analysis.immunogenicity import (
    mhc2_anchor_analysis, deimmunise_alternatives, MHC2_ANCHORS,
)

_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"


def test_anchor_analysis_basics():
    a = mhc2_anchor_analysis(_SEQ, 12)
    assert a["applicable"] is True
    assert 1 <= a["register"] <= 9
    assert a["is_anchor"] == (a["register"] in MHC2_ANCHORS)
    assert 0 <= a["core_score"] <= 100
    assert isinstance(a["verdict"], str) and a["verdict"]


def test_anchor_short_sequence_not_applicable():
    assert mhc2_anchor_analysis("MKT", 1)["applicable"] is False
    assert deimmunise_alternatives("MKT", 1) == []


def test_deimmunise_ranks_by_lowest_local_score():
    alts = deimmunise_alternatives(_SEQ, 12, top_n=5)
    assert alts and len(alts) <= 5
    scores = [a["max_core"] for a in alts]
    assert scores == sorted(scores)           # most-deimmunising first
    assert all(a["aa"] != _SEQ[11] for a in alts)  # excludes WT
    # at least one option meaningfully lowers the local epitope signal
    assert any(a["delta_vs_wt"] < 0 for a in alts)
