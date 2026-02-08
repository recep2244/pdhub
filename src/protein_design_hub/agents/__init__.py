"""Multi-agent pipeline for protein structure prediction and evaluation.

Two modes are available:

**Step agents** (default ``mode="step"``)
  Each pipeline step has one agent: Input → Prediction → Evaluation →
  Comparison → Report.

**LLM-guided agents** (``mode="llm"``)
  Inspired by the Virtual Lab (https://github.com/zou-group/virtual-lab),
  LLM "scientist" agents are interleaved with the computational steps.
  They hold **team meetings** and **individual meetings** to plan, review,
  and interpret every stage.

  Pipeline: Input → *Planning Meeting* → Prediction → *Prediction Review* →
  Evaluation → Comparison → *Evaluation Review* → Report

Usage (step mode):

    from pathlib import Path
    from protein_design_hub.agents import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    result = orchestrator.run(input_path=Path("input.fasta"))

Usage (LLM-guided mode):

    orchestrator = AgentOrchestrator(mode="llm")
    result = orchestrator.run(
        input_path=Path("input.fasta"),
        reference_path=Path("native.pdb"),
    )
"""

# Core
from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.orchestrator import AgentOrchestrator
from protein_design_hub.agents.registry import AgentRegistry, DEFAULT_PIPELINE_AGENTS

# Step agents
from protein_design_hub.agents.input_agent import InputAgent
from protein_design_hub.agents.prediction_agent import PredictionAgent
from protein_design_hub.agents.evaluation_agent import EvaluationAgent
from protein_design_hub.agents.comparison_agent import ComparisonAgent
from protein_design_hub.agents.report_agent import ReportAgent

# LLM layer
from protein_design_hub.agents.llm_agent import LLMAgent
from protein_design_hub.agents.meeting import run_meeting
from protein_design_hub.agents.llm_guided import (
    LLMPlanningAgent,
    LLMPredictionReviewAgent,
    LLMEvaluationReviewAgent,
    LLMRefinementReviewAgent,
    LLMMutagenesisPlanningAgent,
)

# Scientists & teams
from protein_design_hub.agents.scientists import (  # noqa: F401
    ALL_AGENTS,
    ALL_TEAMS,
)

__all__ = [
    # Core
    "AgentResult",
    "BaseAgent",
    "WorkflowContext",
    "AgentOrchestrator",
    "AgentRegistry",
    "DEFAULT_PIPELINE_AGENTS",
    # Step agents
    "InputAgent",
    "PredictionAgent",
    "EvaluationAgent",
    "ComparisonAgent",
    "ReportAgent",
    # LLM layer
    "LLMAgent",
    "run_meeting",
    "LLMPlanningAgent",
    "LLMPredictionReviewAgent",
    "LLMEvaluationReviewAgent",
    "LLMRefinementReviewAgent",
    "LLMMutagenesisPlanningAgent",
    # Scientists & teams
    "ALL_AGENTS",
    "ALL_TEAMS",
]
