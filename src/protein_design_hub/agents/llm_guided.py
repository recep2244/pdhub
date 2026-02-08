"""LLM-guided pipeline agents.

These agents wrap the existing step agents (InputAgent, PredictionAgent, …)
and add an LLM "team meeting" or "individual meeting" **before** and/or
**after** the computational step.

Pattern
-------
1. **Pre-step meeting** – LLM agents discuss *what* to do (e.g. which
   predictors, which metrics, parameter advice).
2. **Computational step** – the inner step agent runs the actual computation.
3. **Post-step meeting** – LLM agents interpret the results and decide
   next actions.

This mirrors the Virtual-Lab approach where LLM agents discuss an agenda,
then a script is run, then the results are reviewed.

Reference: https://github.com/zou-group/virtual-lab
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.agents.llm_agent import LLMAgent
from protein_design_hub.agents.meeting import run_meeting
from protein_design_hub.agents.scientists import (
    DEFAULT_TEAM_LEAD,
    DEFAULT_TEAM_MEMBERS,
    EVALUATION_TEAM_MEMBERS,
    REFINEMENT_TEAM_MEMBERS,
    MUTAGENESIS_TEAM_MEMBERS,
    SCIENTIFIC_CRITIC,
)

# Step agents
from protein_design_hub.agents.input_agent import InputAgent
from protein_design_hub.agents.prediction_agent import PredictionAgent
from protein_design_hub.agents.evaluation_agent import EvaluationAgent
from protein_design_hub.agents.comparison_agent import ComparisonAgent
from protein_design_hub.agents.report_agent import ReportAgent


class _LLMGuidedMixin:
    """Shared helper to run a meeting around a step."""

    def _run_meeting_if_enabled(
        self,
        meeting_type: str,
        agenda: str,
        context: WorkflowContext,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        team_member: LLMAgent | None = None,
        critic: LLMAgent | None = None,
        questions: Sequence[str] = (),
        rules: Sequence[str] = (),
        summaries: Sequence[str] = (),
        contexts: Sequence[str] = (),
        num_rounds: int = 1,
        save_name: str = "discussion",
    ) -> str | None:
        """Run a meeting and return its summary."""
        save_dir = (context.job_dir or Path("./outputs")) / "meetings"
        return run_meeting(
            meeting_type=meeting_type,  # type: ignore[arg-type]
            agenda=agenda,
            save_dir=save_dir,
            save_name=save_name,
            team_lead=team_lead,
            team_members=team_members,
            team_member=team_member,
            critic=critic,
            agenda_questions=questions,
            agenda_rules=rules,
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
            return_summary=True,
        )


# ── LLM-guided step agents ─────────────────────────────────────────

class LLMPlanningAgent(BaseAgent, _LLMGuidedMixin):
    """Team meeting to plan the entire pipeline before any computation.

    The team discusses the protein(s), selects predictors, metrics, and
    parameters.  The summary is stored in ``context.extra["plan"]``.
    """

    name = "llm_planning"
    description = "LLM team meeting to plan the prediction pipeline"

    def __init__(
        self,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        num_rounds: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.team_lead = team_lead or DEFAULT_TEAM_LEAD
        self.team_members = team_members or DEFAULT_TEAM_MEMBERS
        self.num_rounds = num_rounds

    def run(self, context: WorkflowContext) -> AgentResult:
        try:
            context.with_job_dir()
            seq_info = ""
            if context.sequences:
                seq_info = (
                    f"\n\nSequences provided ({len(context.sequences)}):\n"
                    + "\n".join(
                        f"  - {s.id}: {len(s.sequence)} residues"
                        for s in context.sequences
                    )
                )
            agenda = (
                "Plan the protein structure prediction and evaluation pipeline. "
                "Consider protein type (monomer, complex, antibody, de novo), "
                "expected structural features (domains, disordered regions, "
                "disulfide bonds, ligand binding), and downstream application "
                "(drug target, enzyme design, structural biology). "
                "Select predictors, evaluation metrics, and quality thresholds."
                f"{seq_info}"
            )
            questions = (
                "Which predictors should we run? Consider: ESMFold (fast, single-sequence), "
                "ColabFold (MSA-based, higher accuracy), Chai-1/Boltz-2 (complexes, "
                "diffusion-based). What order and why?",
                "Which evaluation metrics are most critical? Consider: pLDDT (local "
                "confidence), pTM (global fold), RMSD/TM-score (if reference available), "
                "clash score, Ramachandran quality, and energy-based scores.",
                "What are the protein-specific considerations? Is this an IDP, membrane "
                "protein, multi-domain, or protein with known flexible loops?",
                "What is the success criterion? Define concrete thresholds (e.g. "
                "pLDDT > 80, TM-score > 0.7, clash score < 20).",
            )
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                save_name="planning_meeting",
            )
            context.extra["plan"] = summary or ""
            return AgentResult.ok(context, "Planning meeting completed")
        except Exception as e:
            return AgentResult.fail(f"Planning meeting failed: {e}", error=e)


class LLMPredictionReviewAgent(BaseAgent, _LLMGuidedMixin):
    """Team meeting to review prediction results.

    Runs **after** PredictionAgent.  The Structural Biologist, Liam
    (quality assessment), and the Scientific Critic review the outputs.
    The team lead synthesises into a recommendation for evaluation.
    """

    name = "llm_prediction_review"
    description = "LLM team meeting to review prediction quality"

    def __init__(
        self,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        num_rounds: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.team_lead = team_lead or DEFAULT_TEAM_LEAD
        if team_members is None:
            from protein_design_hub.agents.scientists import (
                STRUCTURAL_BIOLOGIST, LIAM, SCIENTIFIC_CRITIC,
            )
            self.team_members = (STRUCTURAL_BIOLOGIST, LIAM, SCIENTIFIC_CRITIC)
        else:
            self.team_members = team_members
        self.num_rounds = num_rounds

    def run(self, context: WorkflowContext) -> AgentResult:
        if not context.prediction_results:
            return AgentResult.ok(context, "No predictions to review")
        try:
            context.with_job_dir()
            lines = []
            for name, res in context.prediction_results.items():
                if res.success:
                    n = len(res.structure_paths)
                    best = ""
                    if res.scores:
                        plddts = [s.plddt for s in res.scores if s.plddt]
                        if plddts:
                            best = f", best pLDDT = {max(plddts):.1f}"
                    lines.append(f"  - {name}: {n} structures{best}")
                else:
                    lines.append(f"  - {name}: FAILED ({res.error_message})")
            pred_summary = "\n".join(lines)

            agenda = (
                "Review the prediction results and assess quality.\n\n"
                f"Prediction results:\n{pred_summary}\n\n"
                "Assess model quality using pLDDT distributions, identify any "
                "concerns, suggest which structures to prioritise for evaluation, "
                "flag any predictors whose output is suspect, and recommend "
                "whether any structures need refinement before evaluation."
            )
            questions = (
                "Which predictions show the highest quality and should be prioritised?",
                "Are there any quality red flags (low pLDDT regions, failed predictors)?",
                "Should any structures be sent for refinement before evaluation?",
                "What quality assessment metrics should be applied during evaluation?",
            )
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                summaries=tuple(
                    v for v in [context.extra.get("plan")] if v
                ),
                save_name="prediction_review",
            )
            context.extra["prediction_review"] = summary or ""
            return AgentResult.ok(context, "Prediction review completed")
        except Exception as e:
            return AgentResult.fail(f"Prediction review failed: {e}", error=e)


class LLMEvaluationReviewAgent(BaseAgent, _LLMGuidedMixin):
    """Team meeting to review evaluation results.

    Runs **after** EvaluationAgent + ComparisonAgent.  The evaluation
    team (including Liam for quality assessment) interprets the metrics
    and produces a recommendation.
    """

    name = "llm_evaluation_review"
    description = "LLM team meeting to interpret evaluation and comparison"

    def __init__(
        self,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        num_rounds: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.team_lead = team_lead or DEFAULT_TEAM_LEAD
        self.team_members = team_members or EVALUATION_TEAM_MEMBERS
        self.num_rounds = num_rounds

    def run(self, context: WorkflowContext) -> AgentResult:
        comp = context.comparison_result
        if comp is None:
            return AgentResult.ok(context, "No comparison results to review")
        try:
            context.with_job_dir()
            lines = []
            for name, score in comp.ranking:
                lines.append(f"  - {name}: score {score:.3f}")
            rank_text = "\n".join(lines) if lines else "  (none)"

            eval_lines = []
            for name, ev in context.evaluation_results.items():
                parts = []
                if ev.lddt is not None:
                    parts.append(f"lDDT={ev.lddt:.3f}")
                if ev.tm_score is not None:
                    parts.append(f"TM={ev.tm_score:.3f}")
                if ev.rmsd is not None:
                    parts.append(f"RMSD={ev.rmsd:.2f}")
                eval_lines.append(f"  - {name}: {', '.join(parts) or 'n/a'}")
            eval_text = "\n".join(eval_lines) if eval_lines else "  (none)"

            agenda = (
                "Review the evaluation and comparison results.\n\n"
                f"Ranking:\n{rank_text}\n\n"
                f"Evaluation details:\n{eval_text}\n\n"
                "Discuss which predictor produced the best structure and why. "
                "Assess model quality using ModFold-style scoring and pLDDT. "
                "Identify weaknesses, per-residue quality concerns, and whether "
                "the structures are suitable for downstream applications."
            )
            questions = (
                "Which predictor produced the best structure and why?",
                "Are there any quality concerns in the top-ranked structure?",
                "What per-residue quality issues exist (low-confidence regions)?",
                "Are the structures suitable for downstream use (docking, design)?",
            )
            prev_summaries = [
                v for v in [
                    context.extra.get("plan"),
                    context.extra.get("prediction_review"),
                ] if v
            ]
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                summaries=tuple(prev_summaries),
                save_name="evaluation_review",
            )
            context.extra["evaluation_review"] = summary or ""
            return AgentResult.ok(context, "Evaluation review completed")
        except Exception as e:
            return AgentResult.fail(f"Evaluation review failed: {e}", error=e)


class LLMRefinementReviewAgent(BaseAgent, _LLMGuidedMixin):
    """Team meeting for refinement strategy.

    Runs **after** LLMEvaluationReviewAgent.  Digital Recep (refinement
    expert), Liam (quality assessment), Structural Biologist, and the
    Scientific Critic discuss whether and how to refine the structures.
    """

    name = "llm_refinement_review"
    description = "LLM team meeting for structure refinement strategy"

    def __init__(
        self,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        num_rounds: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.team_lead = team_lead or DEFAULT_TEAM_LEAD
        self.team_members = team_members or REFINEMENT_TEAM_MEMBERS
        self.num_rounds = num_rounds

    def run(self, context: WorkflowContext) -> AgentResult:
        # Only run if we have evaluation results to discuss
        if not context.evaluation_results and not context.prediction_results:
            return AgentResult.ok(context, "No structures to discuss for refinement")
        try:
            context.with_job_dir()

            # Build structure summary for refinement discussion
            struct_lines = []
            for name, pr in context.prediction_results.items():
                if not pr.success:
                    continue
                parts = [f"{len(pr.structure_paths)} structures"]
                if pr.scores:
                    plddts = [s.plddt for s in pr.scores if s.plddt]
                    if plddts:
                        parts.append(f"pLDDT {min(plddts):.0f}-{max(plddts):.0f}")
                ev = context.evaluation_results.get(name)
                if ev:
                    if ev.lddt is not None:
                        parts.append(f"lDDT={ev.lddt:.3f}")
                    if ev.rmsd is not None:
                        parts.append(f"RMSD={ev.rmsd:.2f}")
                struct_lines.append(f"  - {name}: {', '.join(parts)}")
            struct_text = "\n".join(struct_lines) if struct_lines else "  (no successful predictions)"

            # Include previous meeting context
            prev_eval = context.extra.get("evaluation_review", "")
            eval_snippet = prev_eval[:500] if prev_eval else "(no evaluation review available)"

            agenda = (
                "Discuss structure refinement strategy for the predicted structures.\n\n"
                f"Structures available:\n{struct_text}\n\n"
                f"Evaluation review summary:\n{eval_snippet}\n\n"
                "Decide which structures need refinement, what refinement protocol "
                "to use (AMBER relaxation for quick stereochemical cleanup, "
                "GalaxyRefine for side-chain repacking and mild backbone perturbation, "
                "ModRefiner for full atomic-level refinement, or Rosetta FastRelax "
                "for energy-driven relaxation), and what quality metrics to track "
                "before/after refinement (MolProbity score, clash score, "
                "Ramachandran favored %, rotamer outliers %, RMSD to pre-refinement)."
            )
            questions = (
                "Which structures need refinement and why? (e.g. clash score > 20, "
                "Ramachandran favored < 95%, or strained rotamers)",
                "What refinement protocol is most appropriate for each structure? "
                "Consider the balance between aggressive refinement (risk of fold "
                "distortion) and minimal cleanup (may leave quality issues).",
                "What quality metrics should be tracked before/after refinement? "
                "At minimum: clash score, Ramachandran stats, RMSD to input.",
                "Are there any risks of refinement distorting the fold? "
                "(e.g. domain rearrangement, loop remodeling artifacts)",
                "What restraint strategies should be applied? "
                "(e.g. harmonic restraints on well-predicted regions, flexible loops)",
            )
            prev_summaries = [
                v for v in [
                    context.extra.get("plan"),
                    context.extra.get("prediction_review"),
                    context.extra.get("evaluation_review"),
                ] if v
            ]
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                summaries=tuple(prev_summaries),
                save_name="refinement_review",
            )
            context.extra["refinement_review"] = summary or ""
            return AgentResult.ok(context, "Refinement review completed")
        except Exception as e:
            return AgentResult.fail(f"Refinement review failed: {e}", error=e)


class LLMMutagenesisPlanningAgent(BaseAgent, _LLMGuidedMixin):
    """Team meeting to plan mutagenesis and sequence design strategy.

    The Protein Engineer, ML Specialist, Biophysicist, and Scientific Critic
    discuss mutation strategies, hotspot identification, library design,
    and ProteinMPNN parameters.
    """

    name = "llm_mutagenesis_planning"
    description = "LLM team meeting for mutagenesis and sequence design strategy"

    def __init__(
        self,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        num_rounds: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.team_lead = team_lead or DEFAULT_TEAM_LEAD
        self.team_members = team_members or MUTAGENESIS_TEAM_MEMBERS
        self.num_rounds = num_rounds

    def run(self, context: WorkflowContext) -> AgentResult:
        try:
            context.with_job_dir()

            # Build context about available structures
            struct_info = ""
            if context.prediction_results:
                lines = []
                for name, pr in context.prediction_results.items():
                    if pr.success and pr.scores:
                        plddts = [s.plddt for s in pr.scores if s.plddt]
                        if plddts:
                            lines.append(
                                f"  - {name}: best pLDDT={max(plddts):.1f}, "
                                f"{len(pr.structure_paths)} structures"
                            )
                if lines:
                    struct_info = "\n\nAvailable structures:\n" + "\n".join(lines)

            agenda = (
                "Plan the mutagenesis and sequence design strategy for protein "
                "engineering.\n"
                f"{struct_info}\n\n"
                "Discuss: which residues to target for mutation (based on "
                "evolutionary conservation, structural role, and predicted "
                "stability effects), whether to use saturation mutagenesis "
                "or focused libraries, ProteinMPNN parameters (temperature, "
                "fixed positions, chain masking), and experimental validation "
                "strategy."
            )
            questions = (
                "Which positions should be targeted for mutation? Consider: "
                "active site residues (conserved, risky), surface residues "
                "(safer for stability, good for binding), buried residues "
                "(crucial for folding, high-risk).",
                "What mutation strategy is best? Saturation mutagenesis at "
                "key positions vs. ProteinMPNN-guided design vs. consensus "
                "sequence mutations vs. phylogenetic analysis?",
                "For ProteinMPNN: what temperature (0.1 conservative, 0.3 moderate, "
                "0.5 diverse)? Which positions to fix and which to redesign?",
                "How should we validate designs computationally before experiments? "
                "(Self-consistency TM-score, ddG prediction, pLDDT of redesigned "
                "structure, solubility prediction)",
                "What library size is experimentally tractable? How to prioritize "
                "variants for screening?",
            )
            prev_summaries = [
                v for v in [
                    context.extra.get("plan"),
                    context.extra.get("prediction_review"),
                    context.extra.get("evaluation_review"),
                    context.extra.get("refinement_review"),
                ] if v
            ]
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                summaries=tuple(prev_summaries),
                save_name="mutagenesis_planning",
            )
            context.extra["mutagenesis_plan"] = summary or ""
            return AgentResult.ok(context, "Mutagenesis planning completed")
        except Exception as e:
            return AgentResult.fail(f"Mutagenesis planning failed: {e}", error=e)
