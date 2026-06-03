from __future__ import annotations

import pytest

from protein_design_hub.evaluation.validation_gate import (
    DEFAULT_KPI1_TARGET,
    DEFAULT_KPI3_TARGET,
    INSUFFICIENT_CORPUS_REASON,
    evaluate_gate,
    kpi1_intent_to_shortlist,
    kpi3_shortlist_precision,
)


# --------------------------------------------------------------------------- #
# KPI-1: intent -> shortlist                                                   #
# --------------------------------------------------------------------------- #
def test_kpi1_empty_corpus_is_zero():
    assert kpi1_intent_to_shortlist([]) == 0.0


def test_kpi1_counts_non_empty_shortlists():
    runs = [
        {"shortlist": [{"id": "a"}]},          # reached
        {"shortlist": []},                      # empty -> miss
        {"shortlist_size": 3},                  # reached via count
        {"shortlist_size": 0},                  # zero count -> miss
    ]
    assert kpi1_intent_to_shortlist(runs) == pytest.approx(0.5)


def test_kpi1_exported_flag_vetoes_run():
    runs = [
        {"shortlist": [{"id": "a"}], "exported": False},  # vetoed
        {"shortlist": [{"id": "b"}], "exported": True},   # reached
    ]
    assert kpi1_intent_to_shortlist(runs) == pytest.approx(0.5)


def test_kpi1_string_shortlist_not_treated_as_sequence():
    # A stray string should not be mistaken for a populated shortlist.
    assert kpi1_intent_to_shortlist([{"shortlist": "nope"}]) == 0.0


# --------------------------------------------------------------------------- #
# KPI-3: shortlist precision                                                   #
# --------------------------------------------------------------------------- #
def test_kpi3_empty_shortlist_is_zero():
    assert kpi3_shortlist_precision([], {"a": True}) == 0.0


def test_kpi3_precision_with_dict_entries():
    shortlist = [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}]
    labels = {"a": True, "b": True, "c": True, "d": False}
    assert kpi3_shortlist_precision(shortlist, labels) == pytest.approx(0.75)


def test_kpi3_precision_with_scalar_ids_and_missing_labels():
    shortlist = ["a", "b", "z"]  # "z" absent from labels -> graded negative
    labels = {"a": True, "b": True}
    assert kpi3_shortlist_precision(shortlist, labels) == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# evaluate_gate                                                                #
# --------------------------------------------------------------------------- #
def _strong_corpus():
    # 9/10 runs reach a shortlist -> KPI-1 = 0.9 (>= 0.70).
    runs = [{"shortlist": [{"id": f"r{i}"}]} for i in range(9)]
    runs.append({"shortlist": []})
    # 10-entry shortlist, 1 false positive -> precision 0.9 (>= 0.90).
    shortlist = [{"id": f"c{i}"} for i in range(10)]
    labels = {f"c{i}": True for i in range(9)}
    labels["c9"] = False
    return runs, shortlist, labels


def test_evaluate_gate_strong_corpus_passes():
    runs, shortlist, labels = _strong_corpus()
    res = evaluate_gate(runs, shortlist, labels)

    assert res["kpi1_intent_to_shortlist"] == pytest.approx(0.9)
    assert res["kpi3_shortlist_precision"] == pytest.approx(0.9)
    assert res["kpi1_pass"] is True
    assert res["kpi3_pass"] is True
    assert res["release_ok"] is True
    assert "reason" not in res
    assert "RELEASE OK" in res["verdict"]
    assert res["kpi1_target"] == DEFAULT_KPI1_TARGET
    assert res["kpi3_target"] == DEFAULT_KPI3_TARGET


def test_evaluate_gate_weak_corpus_fails_both_kpis():
    # Only 1/4 runs reach a shortlist; shortlist is half wrong.
    runs = [
        {"shortlist": [{"id": "a"}]},
        {"shortlist": []},
        {"shortlist": []},
        {"shortlist_size": 0},
    ]
    shortlist = [{"id": "a"}, {"id": "b"}]
    labels = {"a": True, "b": False}
    res = evaluate_gate(runs, shortlist, labels)

    assert res["kpi1_intent_to_shortlist"] == pytest.approx(0.25)
    assert res["kpi3_shortlist_precision"] == pytest.approx(0.5)
    assert res["kpi1_pass"] is False
    assert res["kpi3_pass"] is False
    assert res["release_ok"] is False
    assert res.get("reason") != INSUFFICIENT_CORPUS_REASON
    assert "BLOCKED" in res["verdict"]


def test_evaluate_gate_empty_corpus_insufficient():
    res = evaluate_gate([], [], {})
    assert res["release_ok"] is False
    assert res["reason"] == INSUFFICIENT_CORPUS_REASON
    assert "insufficient corpus" in res["verdict"]


def test_evaluate_gate_partial_corpus_missing_labels_insufficient():
    # Runs and shortlist exist but no external labels -> cannot grade precision.
    runs = [{"shortlist": [{"id": "a"}]}]
    shortlist = [{"id": "a"}]
    res = evaluate_gate(runs, shortlist, {})
    assert res["release_ok"] is False
    assert res["reason"] == INSUFFICIENT_CORPUS_REASON


def test_evaluate_gate_custom_targets():
    runs, shortlist, labels = _strong_corpus()  # kpi1=0.9, kpi3=0.9
    # Raise KPI-3 target above the achieved 0.9 -> should block.
    res = evaluate_gate(runs, shortlist, labels, kpi3_target=0.95)
    assert res["kpi3_pass"] is False
    assert res["release_ok"] is False
