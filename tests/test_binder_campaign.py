"""Tests for Track-3 binder-design campaign planning & health."""

from protein_design_hub.design.campaign import (
    plan_campaign, assess_health, recommend_stack, PASS_RATE,
)


def test_plan_scales_with_target_and_difficulty():
    small = plan_campaign(5, "easy")
    big = plan_campaign(50, "easy")
    assert big.backbones > small.backbones
    assert big.sequences > small.sequences
    # harder target → lower pass rate → more passing-needed headroom
    easy = plan_campaign(10, "easy")
    hard = plan_campaign(10, "difficult")
    assert easy.pass_rate > hard.pass_rate
    assert PASS_RATE["easy"] > PASS_RATE["difficult"]


def test_plan_has_funnel_cost_and_stack():
    p = plan_campaign(10, "medium", predictor="chai")
    assert p.funnel and p.funnel[0]["count"] == p.backbones
    assert p.funnel[-1]["count"] == p.expected_finals
    assert p.est_cost_usd > 0 and p.est_hours > 0
    assert p.recommended_stack and p.rationale
    # funnel expands backbones→sequences (8 seq/backbone), then narrows to finals
    counts = [s["count"] for s in p.funnel]
    assert counts[1] > counts[0]                       # MPNN expansion
    assert counts[1:] == sorted(counts[1:], reverse=True)   # narrows thereafter


def test_recommend_stack_priorities():
    assert "BindCraft" in recommend_stack(priority="higher_success")["stack"]
    assert "BoltzGen" in recommend_stack(priority="all_atom")["stack"]
    # difficult target upgrades the standard stack
    assert "ColabFold" in recommend_stack("difficult", "standard")["stack"]


def test_assess_health_bands():
    good = assess_health(0.6, 0.5, 0.7)            # high → product > 0.15
    assert "EXCELLENT" in good["health"] or "GOOD" in good["health"]
    poor = assess_health(0.1, 0.1, 0.3)
    assert "POOR" in poor["health"] and poor["diagnostics"]
    # low ipTM should be diagnosed
    diag = assess_health(0.5, 0.05, 0.6)
    assert any("ipTM" in d for d in diag["diagnostics"])
