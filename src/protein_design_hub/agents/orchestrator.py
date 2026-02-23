"""Orchestrator that runs a chain of agents with shared context.

Supports three modes:

* **step-only** (default) – the 5 computational step agents.
* **llm-guided** – LLM discussion agents are interleaved with the
  computational steps so that scientist LLM agents plan, review, and
  interpret every stage.  Inspired by the Virtual-Lab approach.
* **custom** – caller supplies any list of agents.
"""

from pathlib import Path
from datetime import datetime
from typing import Callable, List, Optional

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.registry import AgentRegistry


# Agent display names for progress
_AGENT_LABELS = {
    "input": "Parsing input sequences",
    "llm_input_review": "Input review (LLM team: sequence validation)",
    "llm_planning": "Planning meeting (LLM team discussion)",
    "prediction": "Running structure predictors",
    "llm_prediction_review": "Prediction review (LLM team: Structural Biologist + Liam)",
    "evaluation": "Evaluating structures",
    "comparison": "Comparing and ranking results",
    "llm_evaluation_review": "Evaluation review (LLM team: Biophysicist + Liam)",
    "llm_refinement_review": "Refinement strategy (LLM team: Digital Recep + Liam)",
    "llm_mutagenesis_planning": "Mutagenesis planning (LLM team: Protein Engineer + ML Specialist)",
    "llm_report_narrative": "Executive summary (LLM team: synthesising all results)",
    "report": "Generating reports",
    # Mutagenesis pipeline agents
    "llm_baseline_review": "Baseline review (LLM team)",
    "llm_mutation_suggestion": "Mutation suggestion (LLM team)",
    "mutation_execution": "Executing mutations",
    "mutation_comparison": "Comparing mutants vs WT",
    "llm_mutation_results": "Results interpretation (LLM team)",
    "mutagenesis_report": "Generating mutagenesis report",
}

_LLM_VERDICT_KEYS = {
    "llm_input_review": "input_review",
    "llm_planning": "planning",
    "llm_prediction_review": "prediction_review",
    "llm_evaluation_review": "evaluation_review",
    "llm_refinement_review": "refinement_review",
    "llm_mutagenesis_planning": "mutagenesis_planning",
    "llm_report_narrative": "executive_summary",
    # Mutagenesis pipeline verdict keys
    "llm_baseline_review": "baseline_review",
    "llm_mutation_suggestion": "mutation_suggestion",
    "llm_mutation_results": "mutation_results_review",
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
        LLMInputReviewAgent,
        LLMPlanningAgent,
        LLMPredictionReviewAgent,
        LLMEvaluationReviewAgent,
        LLMRefinementReviewAgent,
        LLMMutagenesisPlanningAgent,
        LLMReportNarrativeAgent,
    )

    return [
        # 1. Parse input
        InputAgent(progress_callback=progress_callback),
        # 2. Team meeting: validate input sequences & identify characteristics
        LLMInputReviewAgent(progress_callback=progress_callback, **kwargs),
        # 3. Team meeting: plan predictors, metrics, params
        LLMPlanningAgent(progress_callback=progress_callback, **kwargs),
        # 4. Run predictions
        PredictionAgent(progress_callback=progress_callback),
        # 5. Team meeting: review prediction quality (Structural Bio + Liam + Critic)
        LLMPredictionReviewAgent(progress_callback=progress_callback, **kwargs),
        # 6. Evaluate structures
        EvaluationAgent(progress_callback=progress_callback),
        # 7. Compare and rank
        ComparisonAgent(progress_callback=progress_callback),
        # 8. Team meeting: interpret evaluation (Biophysicist + Liam + Critic)
        LLMEvaluationReviewAgent(progress_callback=progress_callback, **kwargs),
        # 9. Team meeting: refinement strategy (Digital Recep + Liam + Critic)
        LLMRefinementReviewAgent(progress_callback=progress_callback, **kwargs),
        # 10. Team meeting: mutagenesis & design planning (Protein Engineer + ML + Biophysicist)
        LLMMutagenesisPlanningAgent(progress_callback=progress_callback, **kwargs),
        # 11. Team meeting: synthesise all results into executive summary
        LLMReportNarrativeAgent(progress_callback=progress_callback, **kwargs),
        # 12. Write reports
        ReportAgent(progress_callback=progress_callback),
    ]


def _build_nanobody_llm_pipeline(
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> List[BaseAgent]:
    """Build the nanobody-specialised LLM-guided pipeline (12 agents).

    Identical to ``_build_llm_guided_pipeline`` but forces
    ``NANOBODY_TEAM_MEMBERS`` onto every LLM meeting agent so that the
    Immunologist is always present.  Caller overrides of *team_members* are
    intentionally ignored — the nanobody team is the defining characteristic
    of this mode.
    """
    from protein_design_hub.agents.input_agent import InputAgent
    from protein_design_hub.agents.prediction_agent import PredictionAgent
    from protein_design_hub.agents.evaluation_agent import EvaluationAgent
    from protein_design_hub.agents.comparison_agent import ComparisonAgent
    from protein_design_hub.agents.report_agent import ReportAgent
    from protein_design_hub.agents.llm_guided import (
        LLMInputReviewAgent,
        LLMPlanningAgent,
        LLMPredictionReviewAgent,
        LLMEvaluationReviewAgent,
        LLMRefinementReviewAgent,
        LLMMutagenesisPlanningAgent,
        LLMReportNarrativeAgent,
    )
    from protein_design_hub.agents.scientists import (
        DEFAULT_TEAM_LEAD,
        NANOBODY_TEAM_MEMBERS,
    )

    nb_kwargs = dict(kwargs)
    # Direct assignment — nanobody team is mandatory for this pipeline mode
    nb_kwargs["team_members"] = NANOBODY_TEAM_MEMBERS
    nb_kwargs.setdefault("team_lead", DEFAULT_TEAM_LEAD)

    return [
        # 1. Parse input
        InputAgent(progress_callback=progress_callback),
        # 2. Team meeting: validate input sequences & identify characteristics
        LLMInputReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        # 3. Team meeting: plan predictors, metrics, params
        LLMPlanningAgent(progress_callback=progress_callback, **nb_kwargs),
        # 4. Run predictions
        PredictionAgent(progress_callback=progress_callback),
        # 5. Team meeting: review prediction quality (nanobody team)
        LLMPredictionReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        # 6. Evaluate structures
        EvaluationAgent(progress_callback=progress_callback),
        # 7. Compare and rank
        ComparisonAgent(progress_callback=progress_callback),
        # 8. Team meeting: interpret evaluation (nanobody team)
        LLMEvaluationReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        # 9. Team meeting: refinement strategy (nanobody team)
        LLMRefinementReviewAgent(progress_callback=progress_callback, **nb_kwargs),
        # 10. Team meeting: mutagenesis & design planning (nanobody team)
        LLMMutagenesisPlanningAgent(progress_callback=progress_callback, **nb_kwargs),
        # 11. Team meeting: synthesise all results into executive summary
        LLMReportNarrativeAgent(progress_callback=progress_callback, **nb_kwargs),
        # 12. Write reports
        ReportAgent(progress_callback=progress_callback),
    ]


def _build_mutagenesis_pre_approval_pipeline(
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> List[BaseAgent]:
    """Build Phase 1: understanding + baseline + suggestion (7 agents)."""
    from protein_design_hub.agents.input_agent import InputAgent
    from protein_design_hub.agents.prediction_agent import PredictionAgent
    from protein_design_hub.agents.evaluation_agent import EvaluationAgent
    from protein_design_hub.agents.comparison_agent import ComparisonAgent
    from protein_design_hub.agents.llm_guided import (
        LLMInputReviewAgent,
        LLMBaselineReviewAgent,
        LLMMutationSuggestionAgent,
    )

    return [
        # 1. Parse input
        InputAgent(progress_callback=progress_callback),
        # 2. LLM review input sequences
        LLMInputReviewAgent(progress_callback=progress_callback, **kwargs),
        # 3. Run structure predictions (WT baseline)
        PredictionAgent(progress_callback=progress_callback),
        # 4. Evaluate WT structures
        EvaluationAgent(progress_callback=progress_callback),
        # 5. Compare and rank predictors
        ComparisonAgent(progress_callback=progress_callback),
        # 6. LLM team reviews WT baseline in detail
        LLMBaselineReviewAgent(progress_callback=progress_callback, **kwargs),
        # 7. LLM team suggests specific mutations
        LLMMutationSuggestionAgent(progress_callback=progress_callback, **kwargs),
    ]


def _build_mutagenesis_post_approval_pipeline(
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> List[BaseAgent]:
    """Build Phase 2: execution + analysis + interpretation (4 agents)."""
    from protein_design_hub.agents.mutagenesis_agents import (
        MutationExecutionAgent,
        MutationComparisonAgent,
        MutagenesisPipelineReportAgent,
    )
    from protein_design_hub.agents.llm_guided import LLMMutationResultsAgent

    return [
        # 8. Execute approved mutations
        MutationExecutionAgent(progress_callback=progress_callback),
        # 9. Compare mutants vs WT
        MutationComparisonAgent(progress_callback=progress_callback),
        # 10. LLM team interprets results
        LLMMutationResultsAgent(progress_callback=progress_callback, **kwargs),
        # 11. Generate mutagenesis report
        MutagenesisPipelineReportAgent(progress_callback=progress_callback),
    ]


def _build_binding_affinity_pipeline(
    progress_callback: Optional[Callable] = None,
    **kwargs,
) -> List[BaseAgent]:
    """Build a focused binding-affinity analysis pipeline (6 agents).

    Runs structure prediction and evaluation, then brings in the Biophysicist
    (+ Scientific Critic) to interpret binding-relevant metrics via
    ``LLMEvaluationReviewAgent``.  Callers may override *team_members* or
    *team_lead* via keyword arguments.
    """
    from protein_design_hub.agents.input_agent import InputAgent
    from protein_design_hub.agents.prediction_agent import PredictionAgent
    from protein_design_hub.agents.evaluation_agent import EvaluationAgent
    from protein_design_hub.agents.comparison_agent import ComparisonAgent
    from protein_design_hub.agents.report_agent import ReportAgent
    from protein_design_hub.agents.llm_guided import LLMEvaluationReviewAgent
    from protein_design_hub.agents.scientists import (
        DEFAULT_TEAM_LEAD,
        BIOPHYSICIST,
        SCIENTIFIC_CRITIC,
    )

    ba_kwargs = dict(kwargs)
    ba_kwargs.setdefault("team_lead", DEFAULT_TEAM_LEAD)
    ba_kwargs.setdefault("team_members", (BIOPHYSICIST, SCIENTIFIC_CRITIC))

    return [
        # 1. Parse input
        InputAgent(progress_callback=progress_callback),
        # 2. Run structure predictions
        PredictionAgent(progress_callback=progress_callback),
        # 3. Evaluate structures
        EvaluationAgent(progress_callback=progress_callback),
        # 4. Compare and rank
        ComparisonAgent(progress_callback=progress_callback),
        # 5. Biophysicist-led evaluation review (interprets binding metrics)
        LLMEvaluationReviewAgent(progress_callback=progress_callback, **ba_kwargs),
        # 6. Write reports
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
        allow_failed_llm_steps: bool = False,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
        **kwargs,
    ):
        """
        Args:
            agents: Explicit list of agents (overrides *mode*).
            mode: ``"step"`` for computational-only pipeline,
                  ``"llm"`` for LLM-guided pipeline with meetings,
                  ``"mutagenesis_pre"`` for Phase 1 mutagenesis (7 agents),
                  ``"mutagenesis_post"`` for Phase 2 mutagenesis (4 agents),
                  ``"nanobody_llm"`` for 12-step nanobody-specialised pipeline
                  (NANOBODY_TEAM_MEMBERS forced on all LLM agents),
                  ``"binding_affinity"`` for 6-step binding affinity analysis
                  (Biophysicist-led evaluation review).
            stop_on_failure: If True, stop as soon as an agent fails.
            allow_failed_llm_steps: If True, continue even when an LLM step
                returns a FAIL verdict (this is logged in context.extra["policy_log"]).
            progress_callback: Optional callback(stage, item, current, total).
            **kwargs: Forwarded to LLM meeting agents (e.g. num_rounds).
        """
        if agents is not None:
            self.agents = agents
        elif mode == "llm":
            self.agents = _build_llm_guided_pipeline(
                progress_callback=progress_callback, **kwargs,
            )
        elif mode == "mutagenesis_pre":
            self.agents = _build_mutagenesis_pre_approval_pipeline(
                progress_callback=progress_callback, **kwargs,
            )
        elif mode == "mutagenesis_post":
            self.agents = _build_mutagenesis_post_approval_pipeline(
                progress_callback=progress_callback, **kwargs,
            )
        elif mode == "nanobody_llm":
            self.agents = _build_nanobody_llm_pipeline(
                progress_callback=progress_callback, **kwargs,
            )
        elif mode == "binding_affinity":
            self.agents = _build_binding_affinity_pipeline(
                progress_callback=progress_callback, **kwargs,
            )
        else:
            self.agents = AgentRegistry.get_default_pipeline(
                progress_callback=progress_callback,
            )

        self.mode = mode
        self.stop_on_failure = stop_on_failure
        self.allow_failed_llm_steps = allow_failed_llm_steps
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

            verdict_key = _LLM_VERDICT_KEYS.get(agent_name)
            if verdict_key:
                verdict = context.step_verdicts.get(verdict_key, {})
                status = str(verdict.get("status", "")).upper()
                if status == "FAIL":
                    policy_log = context.extra.setdefault("policy_log", [])
                    event = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "step": agent_name,
                        "verdict_key": verdict_key,
                        "status": "FAIL",
                        "message": (
                            f"LLM verdict for '{verdict_key}' is FAIL; "
                            + ("continuing due to explicit override." if self.allow_failed_llm_steps else "halting pipeline.")
                        ),
                    }
                    policy_log.append(event)
                    if not self.allow_failed_llm_steps:
                        return AgentResult(
                            success=False,
                            message=(
                                f"Pipeline halted by policy: step '{verdict_key}' returned FAIL. "
                                "Use explicit override to continue."
                            ),
                            context=context,
                        )

        return AgentResult.ok(context, "Pipeline completed")
