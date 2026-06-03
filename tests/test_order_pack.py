"""Tests for the Phase 3 Order Pack exporter (design/order_pack.py)."""

from __future__ import annotations

import os

from protein_design_hub.design.order_pack import build_order_pack, write_order_pack


def _fake_candidates():
    return [
        {
            "id": "weak_binder",
            "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ",
            "score": 0.41,
            "liabilities": ["N12D", "M1A"],
        },
        {
            "id": "strong_binder",
            "sequence": "GSDKIHLQVNGKWVTDDEFAKLLGYKSARELRT",
            "score": 0.92,
            "liabilities": [{"code": "C7S", "note": "free cysteine"}],
        },
    ]


def test_pack_has_all_artifacts():
    pack = build_order_pack(_fake_candidates(), track="binder", host="ecoli")
    for key in ("ranked_csv", "construct_fasta", "construct_map", "assay_plan", "risk_sheet"):
        assert key in pack
        assert isinstance(pack[key], str) and pack[key].strip()
    assert pack["track"] == "binder"
    assert pack["n_candidates"] == 2


def test_csv_is_ranked_best_first():
    pack = build_order_pack(_fake_candidates(), track="binder")
    lines = pack["ranked_csv"].strip().splitlines()
    header, body = lines[0], lines[1:]
    assert header.startswith("rank,id,score")
    # Highest score (strong_binder, 0.92) must rank above weak_binder (0.41).
    ids_in_order = [row.split(",")[1] for row in body]
    assert ids_in_order[0] == "strong_binder"
    assert ids_in_order[1] == "weak_binder"
    # Ranks are 1..N, ascending.
    ranks = [int(row.split(",")[0]) for row in body]
    assert ranks == [1, 2]


def test_fasta_is_valid_dna():
    pack = build_order_pack(_fake_candidates(), track="enzyme")
    fasta = pack["construct_fasta"]
    headers = [ln for ln in fasta.splitlines() if ln.startswith(">")]
    seq_lines = [ln for ln in fasta.splitlines() if ln and not ln.startswith(">")]
    assert len(headers) == 2
    joined = "".join(seq_lines)
    assert joined  # non-empty
    assert set(joined) <= set("ACGT"), f"non-ACGT chars: {set(joined) - set('ACGT')}"


def test_risk_sheet_uses_three_letter_notation():
    pack = build_order_pack(_fake_candidates(), track="binder")
    sheet = pack["risk_sheet"]
    # A123K-style codes are rendered as Ala123Lys-style three-letter notation.
    assert "Asn12Asp" in sheet
    assert "Cys7Ser" in sheet
    assert "free cysteine" in sheet


def test_assay_plan_is_track_specific():
    binder = build_order_pack(_fake_candidates(), track="binder")["assay_plan"]
    stability = build_order_pack(_fake_candidates(), track="stability")["assay_plan"]
    assert "SPR" in binder or "BLI" in binder
    assert "Tm" in stability
    assert binder != stability


def test_unknown_track_falls_back_to_generic():
    pack = build_order_pack(_fake_candidates(), track="weird")
    assert pack["track"] == "weird"
    assert "LC-MS" in pack["assay_plan"]  # generic plan marker


def test_write_order_pack_writes_files(tmp_path):
    pack = build_order_pack(_fake_candidates(), track="binder")
    written = write_order_pack(pack, str(tmp_path))
    assert set(written) == {
        "ranked_csv",
        "construct_fasta",
        "construct_map",
        "assay_plan",
        "risk_sheet",
    }
    for path in written.values():
        assert os.path.isabs(path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    # FASTA file content round-trips to valid DNA.
    with open(written["construct_fasta"], encoding="utf-8") as fh:
        body = "".join(ln for ln in fh.read().splitlines() if ln and not ln.startswith(">"))
    assert set(body) <= set("ACGT")


def test_empty_candidates_still_builds():
    pack = build_order_pack([], track="binder")
    assert pack["n_candidates"] == 0
    assert pack["ranked_csv"].startswith("rank,id,score")
    assert "no candidates" in pack["risk_sheet"]
