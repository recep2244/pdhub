"""Orchestrator that runs a chain of agents with shared context.

Supports three modes:

* **step-only** (default) – the 5 computational step agents.
* **llm-guided** – LLM discussion agents are interleaved with the
  computational steps so that scientist LLM agents plan, review, and
  interpret every stage.  Inspired by the Virtual-Lab approach.
* **custom** – caller supplies any list of agents.
"""

from pathlib import Path
from typing import Callable, List, Optional

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.registry import AgentRegistry


# Agent display names for progress
_AGENT_LABELS = {
    "input": "Parsing input sequences",
    "llm_planning": "Planning meeting (LLM team discussion)",
    "prediction": "Running structure predictors",
    "llm_prediction_review": "Prediction review (LLM team: Structural Biologist + Liam)",
    "evaluation": "Evaluating structures",
    "comparison": "Comparing and ranking results",
    "llm_evaluation_review": "Evaluation review (LLM team: Biophysicist + Liam)",
    "llm_refinement_review": "Refinement strategy (LLM team: Digital Recep + Liam)",
    "llm_mutagenesis_planning": "Mutagenesis planning (LLM team: Protein Engineer + ML Specialist)",
    "report": "Generating reports",
}


def _build_llm_guided_pipeline(
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> List[BaseAgent]:
    """Build the LLM-guided pipeline (step agents + meeting agents)."""
    from protein_design_hub.agents.input_agent import InputAgent
    from protein_design_hub.agents.prediction_agent import PredictionAgent
    from protein_design_hub.agents.evaluation_agent import EvaluationAgent
    from protein_design_hub.agents.comparison_agent import ComparisonAgent
    from protein_design_hub.agents.report_agent import ReportAgent
    from protein_design_hub.agents.llm_guided import (
        LLMPlanningAgent,
        LLMPredictionReviewAgent,
        LLMEvaluationReviewAgent,
        LLMRefinementReviewAgent,
        LLMMutagenesisPlanningAgent,
    )

    return [
        # 1. Parse input
        InputAgent(progress_callback=progress_callback),
        # 2. Team meeting: plan predictors, metrics, params
        LLMPlanningAgent(progress_callback=progress_callback, **kwargs),
        # 3. Run predictions
        PredictionAgent(progress_callback=progress_callback),
        # 4. Team meeting: review prediction quality (Structural Bio + Liam + Critic)
        LLMPredictionReviewAgent(progress_callback=progress_callback, **kwargs),
        # 5. Evaluate structures
        EvaluationAgent(progress_callback=progress_callback),
        # 6. Compare and rank
        ComparisonAgent(progress_callback=progress_callback),
        # 7. Team meeting: interpret evaluation (Biophysicist + Liam + Critic)
        LLMEvaluationReviewAgent(progress_callback=progress_callback, **kwargs),
        # 8. Team meeting: refinement strategy (Digital Recep + Liam + Critic)
        LLMRefinementReviewAgent(progress_callback=progress_callback, **kwargs),
        # 9. Team meeting: mutagenesis & design planning (Protein Engineer + ML + Biophysicist)
        LLMMutagenesisPlanningAgent(progress_callback=progress_callback, **kwargs),
        # 10. Write reports
        ReportAgent(progress_callback=progress_callback),
    ]


class AgentOrchestrator:
    """
    Runs a sequence of agents, passing a single WorkflowContext through
    the chain. Stops on first failure unless configured otherwise.
    """

    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        mode: str = "step",
        stop_on_failure: bool = True,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
        **kwargs,
    ):
        """
        Args:
            agents: Explicit list of agents (overrides *mode*).
            mode: ``"step"`` for computational-only pipeline,
                  ``"llm"`` for LLM-guided pipeline with meetings.
            stop_on_failure: If True, stop as soon as an agent fails.
            progress_callback: Optional callback(stage, item, current, total).
            **kwargs: Forwarded to LLM meeting agents (e.g. num_rounds).
        """
        if agents is not None:
            self.agents = agents
        elif mode == "llm":
            self.agents = _build_llm_guided_pipeline(
                progress_callback=progress_callback, **kwargs,
            )
        else:
            self.agents = AgentRegistry.get_default_pipeline(
                progress_callback=progress_callback,
            )

        self.mode = mode
        self.stop_on_failure = stop_on_failure
        self.progress_callback = progress_callback

    def describe_pipeline(self) -> List[dict]:
        """Return a list of {name, label, type} for each agent in the chain."""
        steps = []
        for agent in self.agents:
            name = getattr(agent, "name", type(agent).__name__)
            label = _AGENT_LABELS.get(name, name)
            agent_type = "llm" if name.startswith("llm_") else "step"
            steps.append({"name": name, "label": label, "type": agent_type})
        return steps

    def run(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        reference_path: Optional[Path] = None,
        predictors: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> AgentResult:
        """
        Run the full agent pipeline.

        Args:
            input_path: Path to input FASTA.
            output_dir: Output directory (default from settings).
            reference_path: Optional reference structure for evaluation.
            predictors: List of predictor names; None = all enabled.
            job_id: Optional job ID (otherwise derived from input path).

        Returns:
            Final AgentResult (with context containing comparison_result on success).
        """
        from protein_design_hub.core.config import get_settings

        settings = get_settings()
        if output_dir is None:
            output_dir = Path(settings.output.base_dir)

        context = WorkflowContext(
            job_id=job_id or "",
            input_path=Path(input_path),
            output_dir=Path(output_dir),
            reference_path=Path(reference_path) if reference_path else None,
            predictors=predictors,
        )

        return self.run_with_context(context)

    def run_with_context(self, context: WorkflowContext) -> AgentResult:
        """
        Run the agent chain with an existing context (e.g. for custom workflows).
        """
        total = len(self.agents)
        for i, agent in enumerate(self.agents):
            agent_name = getattr(agent, "name", type(agent).__name__)
            if self.progress_callback:
                self.progress_callback("agent", agent_name, i + 1, total)

            result = agent.run(context)
            if not result.success and self.stop_on_failure:
                return result
            if result.context is not None:
                context = result.context

        return AgentResult.ok(context, "Pipeline completed")
