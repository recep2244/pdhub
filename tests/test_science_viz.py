"""Tests for the canonical scientific-viz helpers (web/science_viz.py).

Runs with heavy deps absent: science_viz imports only the stdlib.
"""

from __future__ import annotations

import pytest

from protein_design_hub.web import science_viz as sv


# ---------------------------------------------------------------------------
# Confidence ramp
# ---------------------------------------------------------------------------

def test_ramp_anchor_colors_present():
    hexes = {h for _, h in sv.PLDDT_RAMP}
    assert {"#0053d6", "#65cbf3", "#ffdb13", "#ff7d45"} <= hexes


def test_ramp_thresholds_monotonic():
    lowers = [lo for lo, _ in sv.PLDDT_RAMP]
    assert lowers == sorted(lowers)
    assert lowers == [0.0, 50.0, 70.0, 90.0]


def test_confidence_color_monotonic_bands():
    # Walking up the scale should hit the four distinct anchor colours in order.
    assert sv.confidence_color(10, "plddt") == "#ff7d45"
    assert sv.confidence_color(60, "plddt") == "#ffdb13"
    assert sv.confidence_color(80, "plddt") == "#65cbf3"
    assert sv.confidence_color(95, "plddt") == "#0053d6"


def test_confidence_color_distinct_per_band():
    samples = [sv.confidence_color(v, "plddt") for v in (10, 60, 80, 95)]
    assert len(set(samples)) == 4


def test_confidence_color_iptm_scaled():
    # iptm in 0-1 should mirror plddt at *100
    assert sv.confidence_color(0.95, "iptm") == sv.confidence_color(95, "plddt")
    assert sv.confidence_color(0.10, "iptm") == sv.confidence_color(10, "plddt")


def test_confidence_color_pae_inverted():
    # Low PAE -> high confidence (blue); high PAE -> low confidence (orange).
    assert sv.confidence_color(0.0, "pae") == "#0053d6"
    assert sv.confidence_color(30.0, "pae") == "#ff7d45"


def test_confidence_color_clamps():
    assert sv.confidence_color(999, "plddt") == "#0053d6"
    assert sv.confidence_color(-5, "plddt") == "#ff7d45"


def test_confidence_color_bad_kind():
    with pytest.raises(ValueError):
        sv.confidence_color(50, "nonsense")


# ---------------------------------------------------------------------------
# Bands
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,band",
    [
        (95, "very_high"),
        (90, "very_high"),
        (89.9, "high"),
        (70, "high"),
        (69.9, "low"),
        (50, "low"),
        (49.9, "very_low"),
        (0, "very_low"),
    ],
)
def test_confidence_band_thresholds(value, band):
    assert sv.confidence_band(value) == band


# ---------------------------------------------------------------------------
# Property palette
# ---------------------------------------------------------------------------

def test_property_color_known_classes():
    assert sv.property_color("L") == sv.PROPERTY_PALETTE["hydrophobic"]
    assert sv.property_color("W") == sv.PROPERTY_PALETTE["aromatic"]
    assert sv.property_color("K") == sv.PROPERTY_PALETTE["positive"]
    assert sv.property_color("D") == sv.PROPERTY_PALETTE["negative"]
    assert sv.property_color("G") == sv.PROPERTY_PALETTE["special"]


def test_property_color_three_letter_and_case():
    assert sv.property_color("Lys") == sv.property_color("K")
    assert sv.property_color("asp") == sv.property_color("D")


def test_property_color_unknown_fallback():
    assert sv.property_color("X") == sv.PROPERTY_PALETTE["unknown"]
    assert sv.property_color("") == sv.PROPERTY_PALETTE["unknown"]


def test_all_twenty_aas_covered():
    assert len(sv.AA_PROPERTY_CLASS) == 20
    for cls in set(sv.AA_PROPERTY_CLASS.values()):
        assert cls in sv.PROPERTY_PALETTE


# ---------------------------------------------------------------------------
# Effect scale — must be DISTINCT from confidence ramp
# ---------------------------------------------------------------------------

def test_effect_scale_distinct_from_confidence_ramp():
    ramp_hexes = {h.lower() for _, h in sv.PLDDT_RAMP}
    effect_hexes = {h.lower() for _, h in sv.EFFECT_SCALE}
    assert ramp_hexes.isdisjoint(effect_hexes)


def test_effect_color_diverging():
    stabilising = sv.effect_color(-3.0)
    neutral = sv.effect_color(0.0)
    destabilising = sv.effect_color(3.0)
    assert stabilising != neutral != destabilising
    assert stabilising != destabilising
    # ramp colours never leak into the effect encoding
    assert stabilising not in {h for _, h in sv.PLDDT_RAMP}


# ---------------------------------------------------------------------------
# PAE domain
# ---------------------------------------------------------------------------

def test_pae_domain_engine_aware():
    assert sv.pae_domain("alphafold") == (0.0, 31.75)
    assert sv.pae_domain("boltz") == (0.0, 32.0)
    assert sv.pae_domain("esmfold") == (0.0, 30.0)


def test_pae_domain_explicit_max_overrides():
    assert sv.pae_domain("alphafold", max_error=25.0) == (0.0, 25.0)
    assert sv.pae_domain(28.0) == (0.0, 28.0)


def test_pae_domain_unknown_default():
    assert sv.pae_domain("mystery_engine") == (0.0, 30.0)
    assert sv.pae_domain() == (0.0, 30.0)


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def test_takeaway_banner_non_empty_html():
    html = sv.takeaway_banner("Designs passed QC", "success")
    assert isinstance(html, str) and html.strip()
    assert html.startswith("<div")
    assert "Designs passed QC" in html


def test_takeaway_banner_escapes_html():
    html = sv.takeaway_banner("<script>alert(1)</script>", "warning")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_takeaway_banner_levels_differ():
    a = sv.takeaway_banner("x", "success")
    b = sv.takeaway_banner("x", "danger")
    assert a != b


def test_campaign_funnel_non_empty_html():
    stages = [
        {"label": "Generated", "count": 1000},
        {"label": "Passed QC", "count": 120},
        {"label": "Ordered", "count": 24},
    ]
    html = sv.campaign_funnel(stages)
    assert isinstance(html, str) and html.strip()
    assert "Generated" in html and "Passed QC" in html
    assert html.count("pdhub-funnel__row") == 3


def test_campaign_funnel_empty_stages_valid():
    html = sv.campaign_funnel([])
    assert isinstance(html, str) and html.strip()
    assert "pdhub-funnel" in html


def test_campaign_funnel_handles_bad_counts():
    html = sv.campaign_funnel([{"label": "S", "count": None}])
    assert isinstance(html, str) and html.strip()


def test_no_streamlit_import():
    import sys
    # science_viz must not have pulled in streamlit
    sv_mod = sys.modules["protein_design_hub.web.science_viz"]
    assert "streamlit" not in getattr(sv_mod, "__dict__", {})
