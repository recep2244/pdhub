from protein_design_hub.agents.orchestrator import AgentOrchestrator
from protein_design_hub.agents.registry import AgentRegistry


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
    assert len(steps) == 12
    assert [s["name"] for s in steps] == [
        "input",
        "llm_input_review",
        "llm_planning",
        "prediction",
        "llm_prediction_review",
        "evaluation",
        "comparison",
        "llm_evaluation_review",
        "llm_refinement_review",
        "llm_mutagenesis_planning",
        "llm_report_narrative",
        "report",
    ]
    llm_steps = [s for s in steps if s["type"] == "llm"]
    assert len(llm_steps) == 7


def test_registry_exposes_llm_agents():
    names = AgentRegistry.list_names()
    assert "llm_input_review" in names
    assert "llm_planning" in names
    assert "llm_prediction_review" in names
    assert "llm_evaluation_review" in names
    assert "llm_refinement_review" in names
    assert "llm_mutagenesis_planning" in names
    assert "llm_report_narrative" in names
