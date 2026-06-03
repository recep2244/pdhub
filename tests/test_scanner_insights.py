"""Regression tests for the mutation-scanner analytics: tri-expert consensus,
biophysicist verdict, developability assessment, and the data-science feature
frame. The page is loaded via importlib (its digit-prefixed filename blocks a
normal import); Streamlit-rendering functions are not exercised here.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_PAGE = Path(__file__).resolve().parents[1] / "src/protein_design_hub/web/app_pages/10_mutation_scanner.py"


def _load_page():
    spec = importlib.util.spec_from_file_location("mutation_scanner_page", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


PAGE = _load_page()
_SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGGIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK"


def _mut(wt, pos, mt, dlocal, rmsd, clash=5):
    return SimpleNamespace(success=True, original_aa=wt, mutant_aa=mt, position=pos,
                           mutation_code=f"{wt}{pos}{mt}", delta_local_plddt=dlocal,
                           delta_mean_plddt=dlocal / 5.0, rmsd_to_base=rmsd,
                           clash_score=clash, sasa_total=1000.0)


def test_helpers_present():
    for h in ("_consensus_recommendation", "_biophysicist_verdict", "_assess_candidate",
              "_mutation_feature_frame", "_ost_compare", "_effect_badge", "_mut3"):
        assert hasattr(PAGE, h), f"missing {h}"


def test_biophysicist_verdict():
    stable = _mut("A", 5, "V", 4.0, 0.4)
    bad = _mut("G", 28, "W", -9.0, 3.5, clash=40)
    v_ok, _ = PAGE._biophysicist_verdict(stable, esm2_delta=1.0, ost_lddt=0.98)
    v_bad, _ = PAGE._biophysicist_verdict(bad, esm2_delta=-6.0, ost_lddt=0.6)
    assert "🟢" in v_ok
    assert "🔴" in v_bad


def test_consensus_picks_all_agree_mutant():
    muts = [_mut("A", 5, "V", 5.0, 0.4), _mut("S", 33, "W", -9.0, 3.5, clash=40),
            _mut("K", 48, "R", 2.0, 0.8)]
    res = SimpleNamespace(mutations=muts, sequence=_SEQ, ranked_mutations=muts)
    esm2 = {"V": 1.5, "W": -6.0, "R": 0.5}
    rec = PAGE._consensus_recommendation(res, esm2)
    assert rec is not None
    assert rec["consensus"] is not None
    assert rec["consensus"]["code"] == "A5V"          # all three experts agree
    s33 = next(r for r in rec["rows"] if r["code"] == "S33W")
    assert s33["consensus"] is False                  # vetoed (high-risk + destabilising)


def test_consensus_handles_no_esm2():
    """Graceful degradation: ESM-2 disabled (deltas None) must not crash."""
    muts = [_mut("A", 5, "V", 5.0, 0.4), _mut("K", 48, "R", 2.0, 0.8)]
    res = SimpleNamespace(mutations=muts, sequence=_SEQ, ranked_mutations=muts)
    rec = PAGE._consensus_recommendation(res, None)
    assert rec is not None and len(rec["rows"]) == 2


def test_feature_frame_has_engineered_columns():
    muts = [_mut("A", 5, a, 1.0, 0.5) for a in "VLWDKG"]
    res = SimpleNamespace(mutations=muts, sequence=_SEQ)
    df = PAGE._mutation_feature_frame(res, {a: 0.0 for a in "VLWDKG"})
    for col in ("Δhydrophobicity", "Δvolume(MW)", "Δcharge", "BLOSUM62", "aromatic_intro"):
        assert col in df.columns
    assert len(df) == 6


def test_assessment_flags_high_risk():
    a = PAGE._assess_candidate(_mut("S", 33, "W", -9.0, 3.5), _SEQ, esm2_delta=-6.0)
    assert "High-risk" in a["verdict"]
    assert a["liabilities"] >= 2
