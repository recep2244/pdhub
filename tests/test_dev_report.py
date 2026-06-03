from __future__ import annotations

from protein_design_hub.analysis.dev_report import DevReport, developability_report


def test_clean_sequence_is_orderable():
    # No cysteines (so no free-thiol risk), balanced charge and hydropathy.
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    report = developability_report(seq)
    assert isinstance(report, DevReport)
    assert report.orderable is True
    assert report.blockers == []
    assert report.properties.get("length") == len(seq)


def test_odd_cysteine_is_not_orderable():
    # Single cysteine → free thiol → hard NO-GO.
    report = developability_report("ACDEFGHIKLMNPQRSTVWY")
    assert report.orderable is False
    assert any("free thiol" in b.lower() for b in report.blockers)


def test_even_cysteine_clears_free_thiol_blocker():
    report = developability_report("ACDEFGHIKLMNPQRSTVWYC")  # two cysteines
    assert not any("free thiol" in b.lower() for b in report.blockers)


def test_invalid_residue_is_blocked():
    report = developability_report("ACDEFXZB")
    assert report.orderable is False
    assert any("invalid" in b.lower() or "forbidden" in b.lower() for b in report.blockers)


def test_empty_sequence_is_blocked():
    report = developability_report("")
    assert report.orderable is False
    assert report.blockers


def test_lowercase_and_whitespace_tolerated():
    report = developability_report(" acd efg hik ")
    # Valid residues, even/odd cysteine aside, should not raise and should
    # normalise length correctly.
    assert isinstance(report, DevReport)
    assert report.properties.get("length") == 9
