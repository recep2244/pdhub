from protein_design_hub.agents.orchestrator import AgentOrchestrator
from protein_design_hub.agents.registry import AgentRegistry
from protein_design_hub.agents.base import BaseAgent, AgentResult
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.llm_guided import (
    _parse_verdict_from_summary,
    _parse_mutation_plan_from_summary,
)


def test_step_pipeline_has_5_computational_steps():
    orchestrator = AgentOrchestrator(mode="step")
    steps = orchestrator.describe_pipeline()
    assert len(steps) == 5
    assert [s["name"] for s in steps] == [
        "input",
        "prediction",
        "evaluation",
        "comparison",
        "report",
    ]
    assert all(s["type"] == "step" for s in steps)


def test_llm_pipeline_has_12_integrated_steps():
    orchestrator = AgentOrchestrator(mode="llm")
    steps = orchestrator.describe_pipeline()
    names = [s["name"] for s in steps]
    assert len(steps) >= 10

    required_order = [
        "input",
        "llm_planning",
        "prediction",
        "llm_prediction_review",
        "evaluation",
        "comparison",
        "llm_evaluation_review",
        "llm_refinement_review",
        "llm_mutagenesis_planning",
        "report",
    ]
    indices = [names.index(name) for name in required_order]
    assert indices == sorted(indices)

    # Optional enhanced pipeline steps must be in sensible positions.
    if "llm_input_review" in names:
        assert names.index("input") < names.index("llm_input_review") < names.index("llm_planning")
    if "llm_report_narrative" in names:
        assert names.index("llm_mutagenesis_planning") < names.index("llm_report_narrative") < names.index("report")

    llm_steps = [s for s in steps if s["type"] == "llm"]
    assert len(llm_steps) >= 4


def test_registry_exposes_llm_agents():
    names = AgentRegistry.list_names()
    assert "llm_planning" in names
    assert "llm_prediction_review" in names
    assert "llm_evaluation_review" in names
    assert "llm_refinement_review" in names


def test_parse_verdict_uses_structured_contract():
    summary = (
        "Findings...\n"
        'VERDICT_JSON: {"step":"evaluation_review","status":"FAIL",'
        '"key_findings":["clash score too high"],'
        '"thresholds":{"clash_score":"< 20"},'
        '"actions":["run refinement"]}'
    )
    verdict = _parse_verdict_from_summary(summary, "evaluation_review")
    assert verdict["status"] == "FAIL"
    assert verdict["key_findings"] == ["clash score too high"]
    assert verdict["source"] == "verdict_json"


def test_parse_verdict_missing_contract_falls_back_to_warn():
    verdict = _parse_verdict_from_summary("No structured footer", "planning")
    assert verdict["status"] == "WARN"
    assert verdict["source"] == "fallback"


class _FailVerdictAgent(BaseAgent):
    name = "llm_evaluation_review"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.step_verdicts["evaluation_review"] = {
            "status": "FAIL",
            "key_findings": ["test failure verdict"],
        }
        return AgentResult.ok(context, "fail verdict emitted")


class _TailAgent(BaseAgent):
    name = "report"

    def run(self, context: WorkflowContext) -> AgentResult:
        context.extra["tail_ran"] = True
        return AgentResult.ok(context, "tail")


def test_pipeline_policy_halts_on_fail_verdict_without_override():
    orchestrator = AgentOrchestrator(
        agents=[_FailVerdictAgent(), _TailAgent()],
        allow_failed_llm_steps=False,
    )
    context = WorkflowContext(job_id="jobx")
    result = orchestrator.run_with_context(context)
    assert result.success is False
    assert result.context is not None
    assert "halted by policy" in result.message.lower()
    assert not result.context.extra.get("tail_ran", False)
    assert len(result.context.extra.get("policy_log", [])) == 1


def test_pipeline_policy_allows_override_on_fail_verdict():
    orchestrator = AgentOrchestrator(
        agents=[_FailVerdictAgent(), _TailAgent()],
        allow_failed_llm_steps=True,
    )
    context = WorkflowContext(job_id="jobx")
    result = orchestrator.run_with_context(context)
    assert result.success is True
    assert result.context is not None
    assert result.context.extra.get("tail_ran", False)
    assert len(result.context.extra.get("policy_log", [])) == 1


# ── Mutagenesis pipeline tests ──────────────────────────────────


def test_mutagenesis_pre_pipeline_has_7_steps():
    orchestrator = AgentOrchestrator(mode="mutagenesis_pre")
    steps = orchestrator.describe_pipeline()
    assert len(steps) == 7

    names = [s["name"] for s in steps]
    assert names == [
        "input",
        "llm_input_review",
        "prediction",
        "evaluation",
        "comparison",
        "llm_baseline_review",
        "llm_mutation_suggestion",
    ]


def test_mutagenesis_post_pipeline_has_4_steps():
    orchestrator = AgentOrchestrator(mode="mutagenesis_post")
    steps = orchestrator.describe_pipeline()
    assert len(steps) == 4

    names = [s["name"] for s in steps]
    assert names == [
        "mutation_execution",
        "mutation_comparison",
        "llm_mutation_results",
        "mutagenesis_report",
    ]


def test_mutagenesis_pre_step_order():
    orchestrator = AgentOrchestrator(mode="mutagenesis_pre")
    steps = orchestrator.describe_pipeline()
    names = [s["name"] for s in steps]

    # Input must come before prediction
    assert names.index("input") < names.index("prediction")
    # Prediction before evaluation
    assert names.index("prediction") < names.index("evaluation")
    # Baseline review after evaluation
    assert names.index("evaluation") < names.index("llm_baseline_review")
    # Suggestion after baseline review
    assert names.index("llm_baseline_review") < names.index("llm_mutation_suggestion")


def test_mutagenesis_post_step_order():
    orchestrator = AgentOrchestrator(mode="mutagenesis_post")
    steps = orchestrator.describe_pipeline()
    names = [s["name"] for s in steps]

    assert names.index("mutation_execution") < names.index("mutation_comparison")
    assert names.index("mutation_comparison") < names.index("llm_mutation_results")
    assert names.index("llm_mutation_results") < names.index("mutagenesis_report")


def test_registry_exposes_mutagenesis_agents():
    names = AgentRegistry.list_names()
    assert "llm_baseline_review" in names
    assert "llm_mutation_suggestion" in names
    assert "llm_mutation_results" in names
    assert "mutation_execution" in names
    assert "mutation_comparison" in names
    assert "mutagenesis_report" in names


def test_parse_mutation_plan_valid():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    summary = (
        "Discussion...\n"
        'MUTATION_PLAN_JSON: {"positions": ['
        '{"residue": 3, "wt_aa": "D", "targets": ["A", "G"], "rationale": "low pLDDT"},'
        '{"residue": 7, "wt_aa": "G", "targets": ["A"], "rationale": "flexible"}'
        '], "strategy": "targeted", "rationale": "stabilise weak regions"}'
    )
    plan = _parse_mutation_plan_from_summary(summary, sequence)
    assert plan is not None
    assert len(plan["positions"]) == 2
    assert plan["positions"][0]["residue"] == 3
    assert plan["positions"][0]["wt_aa"] == "D"
    assert "A" in plan["positions"][0]["targets"]
    assert plan["strategy"] == "targeted"


def test_parse_mutation_plan_corrects_wrong_wt_aa():
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    summary = (
        'MUTATION_PLAN_JSON: {"positions": ['
        '{"residue": 3, "wt_aa": "X", "targets": ["A"], "rationale": "test"}'
        '], "strategy": "targeted", "rationale": ""}'
    )
    plan = _parse_mutation_plan_from_summary(summary, sequence)
    assert plan is not None
    # Should auto-correct to actual residue D at position 3
    assert plan["positions"][0]["wt_aa"] == "D"


def test_parse_mutation_plan_missing_returns_none():
    plan = _parse_mutation_plan_from_summary("No JSON here", "ACDEF")
    assert plan is None


def test_parse_mutation_plan_invalid_position_skipped():
    sequence = "ACDEF"
    summary = (
        'MUTATION_PLAN_JSON: {"positions": ['
        '{"residue": 99, "wt_aa": "A", "targets": ["G"], "rationale": "out of range"}'
        '], "strategy": "targeted", "rationale": ""}'
    )
    plan = _parse_mutation_plan_from_summary(summary, sequence)
    # Position 99 > len(sequence), so all positions skipped → None
    assert plan is None


def test_parse_mutation_plan_removes_wt_from_targets():
    sequence = "ACDEF"
    summary = (
        'MUTATION_PLAN_JSON: {"positions": ['
        '{"residue": 1, "wt_aa": "A", "targets": ["A", "G", "V"], "rationale": "test"}'
        '], "strategy": "targeted", "rationale": ""}'
    )
    plan = _parse_mutation_plan_from_summary(summary, sequence)
    assert plan is not None
    # "A" should be removed from targets (it's the WT)
    assert "A" not in plan["positions"][0]["targets"]
    assert "G" in plan["positions"][0]["targets"]


def test_mutagenesis_report_agent_importable_by_both_names():
    """Canonical class and backward-compat alias must both be importable."""
    from protein_design_hub.agents.mutagenesis_agents import (
        MutagenesisPipelineReportAgent,
        MutagenesiReportAgent,
    )
    assert MutagenesiReportAgent is MutagenesisPipelineReportAgent
