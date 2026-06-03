"""Tests for the canonical biophysical-properties core.

Covers:
* ``biophysics.properties.compute_properties`` vs known reference values,
* pure-Python fallbacks (BioPython forced absent),
* the empty / dirty sequence contract,
* backward-compatible delegation from ``analysis.protein_qc.biophysical_properties``
  (return-dict keys must stay identical).
"""

from __future__ import annotations

import pytest

from protein_design_hub.biophysics import properties as props
from protein_design_hub.analysis import protein_qc

# A 330-residue reference (E. coli AsnB N-terminal fragment style). Values below
# come from BioPython's ProtParam, the validated reference implementation.
REF_SEQ = (
    "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHS"
    "LAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGI"
    "KATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAP"
    "DYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMP"
    "QTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
)


# --------------------------------------------------------------------------- #
# Individual canonical calculators
# --------------------------------------------------------------------------- #
def test_clean_sequence_drops_noise():
    assert props.clean_sequence("  ac-dx*Z9 ") == "ACD"
    assert props.clean_sequence("") == ""
    assert props.clean_sequence(None) == ""


def test_gravy_simple():
    # All-alanine: GRAVY == Kyte-Doolittle value for Ala (1.8).
    assert props.gravy("AAAA") == pytest.approx(1.8)
    assert props.gravy("") is None


def test_molecular_weight_and_charge_signs():
    # A poly-K peptide should be net positive at pH 7; poly-E net negative.
    assert props.net_charge("KKKKK", 7.0) > 0
    assert props.net_charge("EEEEE", 7.0) < 0
    assert props.molecular_weight("") is None
    assert props.net_charge("") is None


def test_isoelectric_point_monotonic():
    pi_basic = props.isoelectric_point("KKKKRRRR")
    pi_acidic = props.isoelectric_point("DDDDEEEE")
    assert pi_basic > pi_acidic
    # Net charge at the computed pI should be ~0.
    assert props.net_charge(REF_SEQ, props.isoelectric_point(REF_SEQ)) == pytest.approx(0.0, abs=1e-2)


def test_aromaticity_fraction():
    assert props.aromaticity("FWYA") == pytest.approx(0.75)
    assert props.aromaticity("AAAA") == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Aggregate vs reference
# --------------------------------------------------------------------------- #
def test_compute_properties_matches_biopython_reference():
    pytest.importorskip("Bio")
    p = props.compute_properties(REF_SEQ)
    assert p["length"] == 330
    assert p["gravy"] == pytest.approx(-0.294, abs=1e-3)
    assert p["pi"] == pytest.approx(5.45, abs=0.05)
    assert p["mw"] == pytest.approx(36586.0, rel=1e-3)
    assert p["aromaticity"] == pytest.approx(0.07, abs=1e-3)
    assert p["instability"] == pytest.approx(34.2, abs=0.2)


def test_compute_properties_empty():
    p = props.compute_properties("")
    assert p["length"] == 0
    for k in ("gravy", "mw", "pi", "aromaticity", "instability", "net_charge"):
        assert p[k] is None


def test_compute_properties_fallback_without_biopython(monkeypatch):
    """With BioPython forced absent, pure-Python paths must still return numbers
    that are close to the validated reference."""
    monkeypatch.setattr(props, "_has_biopython", lambda: False)
    p = props.compute_properties(REF_SEQ)
    assert p["length"] == 330
    assert p["gravy"] == pytest.approx(-0.294, abs=0.02)
    assert 4.5 < p["pi"] < 6.5
    assert p["aromaticity"] == pytest.approx(0.07, abs=1e-3)
    # Pure-Python instability still produced (not None) via the DIWV fallback.
    assert p["instability"] is not None
    assert p["mw"] == pytest.approx(36586.0, rel=5e-3)


def test_instability_short_sequence_is_none():
    assert props.canonical_instability_index("A") is None
    assert props.canonical_instability_index("") is None


# --------------------------------------------------------------------------- #
# Backward-compatible delegation in protein_qc
# --------------------------------------------------------------------------- #
def test_protein_qc_keys_preserved_nonempty():
    out = protein_qc.biophysical_properties(REF_SEQ)
    # These keys must ALWAYS be present (other code imports/relies on them).
    for k in ("length", "gravy", "pi", "mw", "instability"):
        assert k in out
    assert out["length"] == 330
    # aromaticity is present iff BioPython is installed (historical contract).
    if props._has_biopython():
        assert "aromaticity" in out


def test_protein_qc_empty_contract():
    out = protein_qc.biophysical_properties("")
    assert out == {"gravy": None, "pi": None, "mw": None,
                   "instability": None, "length": 0}


def test_protein_qc_delegates_to_canonical():
    """protein_qc values must agree with the canonical implementation."""
    qc = protein_qc.biophysical_properties(REF_SEQ)
    canon = props.compute_properties(REF_SEQ)
    for k in ("length", "gravy", "pi", "mw", "instability"):
        assert qc[k] == canon[k]
