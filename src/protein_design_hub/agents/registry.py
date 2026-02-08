"""Registry of pipeline agents by step name."""

from typing import Dict, List, Type

from protein_design_hub.agents.base import BaseAgent
from protein_design_hub.agents.input_agent import InputAgent
from protein_design_hub.agents.prediction_agent import PredictionAgent
from protein_design_hub.agents.evaluation_agent import EvaluationAgent
from protein_design_hub.agents.comparison_agent import ComparisonAgent
from protein_design_hub.agents.report_agent import ReportAgent


# Default agents for the prediction pipeline (order matters)
DEFAULT_PIPELINE_AGENTS: List[Type[BaseAgent]] = [
    InputAgent,
    PredictionAgent,
    EvaluationAgent,
    ComparisonAgent,
    ReportAgent,
]


class AgentRegistry:
    """Registry to create and lookup agents by name."""

    _agents: Dict[str, Type[BaseAgent]] = {
        "input": InputAgent,
        "prediction": PredictionAgent,
        "evaluation": EvaluationAgent,
        "comparison": ComparisonAgent,
        "report": ReportAgent,
    }

    @classmethod
    def register(cls, name: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class under a step name."""
        cls._agents[name.lower()] = agent_class

    @classmethod
    def get(cls, name: str, **kwargs) -> BaseAgent:
        """Create an agent instance by name."""
        name_lower = name.lower()
        if name_lower not in cls._agents:
            raise KeyError(f"Unknown agent: {name}. Available: {list(cls._agents)}")
        return cls._agents[name_lower](**kwargs)

    @classmethod
    def list_names(cls) -> List[str]:
        """Return registered agent names in default pipeline order."""
        order = [a.name for a in DEFAULT_PIPELINE_AGENTS]
        return [n for n in order if n in cls._agents] + [
            n for n in cls._agents if n not in order
        ]

    @classmethod
    def get_default_pipeline(cls, **kwargs) -> List[BaseAgent]:
        """Return agent instances for the default prediction pipeline."""
        return [agent_cls(**kwargs) for agent_cls in DEFAULT_PIPELINE_AGENTS]


def _register_llm_agents() -> None:
    """Register LLM-guided agents into the global registry."""
    from protein_design_hub.agents.llm_guided import (
        LLMPlanningAgent,
        LLMPredictionReviewAgent,
        LLMEvaluationReviewAgent,
        LLMRefinementReviewAgent,
    )
    AgentRegistry.register("llm_planning", LLMPlanningAgent)
    AgentRegistry.register("llm_prediction_review", LLMPredictionReviewAgent)
    AgentRegistry.register("llm_evaluation_review", LLMEvaluationReviewAgent)
    AgentRegistry.register("llm_refinement_review", LLMRefinementReviewAgent)


# Auto-register LLM agents on import
try:
    _register_llm_agents()
except Exception:
    pass  # LLM deps may not be installed
