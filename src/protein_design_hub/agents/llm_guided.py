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

import json
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger(__name__)

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
    FULL_PIPELINE_TEAM_MEMBERS,
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
        verdict_step: str | None = None,
    ) -> str | None:
        """Run a meeting and return its summary."""
        merged_rules = list(rules)
        if verdict_step:
            merged_rules.append(_verdict_contract_rule(verdict_step))
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
            agenda_rules=tuple(merged_rules),
            summaries=summaries,
            contexts=contexts,
            num_rounds=num_rounds,
            return_summary=True,
        )


# ── helpers ────────────────────────────────────────────────────────

def _verdict_contract_rule(step_name: str) -> str:
    """Output contract for all meeting summaries."""
    return (
        "End your final summary with exactly one single-line JSON object prefixed by "
        "`VERDICT_JSON:` using this schema: "
        f'{{"step":"{step_name}","status":"PASS|WARN|FAIL","key_findings":["..."],'
        '"thresholds":{"metric":"value"},"actions":["..."]}}. '
        "Do not include markdown fences around this JSON."
    )


def _parse_mutation_plan_from_summary(
    summary: str, sequence: str,
) -> dict | None:
    """Extract a structured mutation plan from meeting summary.

    Searches for ``MUTATION_PLAN_JSON:`` prefix and validates each
    position against the actual sequence.  Invalid positions are
    silently skipped with a warning logged.

    Returns ``None`` if no valid plan is found.
    """
    if not summary:
        return None

    payload = None
    for line in reversed(summary.splitlines()):
        if "MUTATION_PLAN_JSON:" in line:
            payload = line.split("MUTATION_PLAN_JSON:", 1)[1].strip()
            break
    if not payload:
        return None

    try:
        parsed = json.loads(payload)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    positions = parsed.get("positions")
    if not isinstance(positions, list) or not positions:
        return None

    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
    validated: list[dict] = []
    for entry in positions:
        if not isinstance(entry, dict):
            continue
        residue = entry.get("residue")
        if not isinstance(residue, int) or residue < 1:
            continue
        if residue > len(sequence):
            continue
        wt_aa = str(entry.get("wt_aa", "")).upper()
        actual_aa = sequence[residue - 1]
        if wt_aa and wt_aa != actual_aa:
            # Mismatch — auto-correct to actual
            logger.warning(
                "Mutation plan position %d: LLM said wt_aa=%s but actual=%s; correcting.",
                residue, wt_aa, actual_aa,
            )
            wt_aa = actual_aa
        if not wt_aa:
            wt_aa = actual_aa

        targets = entry.get("targets", ["*"])
        if not isinstance(targets, list):
            targets = ["*"]
        targets = [
            t.upper() for t in targets
            if isinstance(t, str) and (t.upper() in valid_aas or t == "*")
        ]
        # Remove WT from targets
        targets = [t for t in targets if t != wt_aa]
        if not targets:
            targets = ["*"]

        validated.append({
            "residue": residue,
            "wt_aa": wt_aa,
            "targets": targets,
            "rationale": str(entry.get("rationale", "")),
        })

    if not validated:
        return None

    return {
        "positions": validated,
        "strategy": str(parsed.get("strategy", "targeted")),
        "rationale": str(parsed.get("rationale", "")),
    }


def _parse_verdict_from_summary(summary: str, step_name: str) -> dict:
    """Extract verdict from the explicit VERDICT_JSON contract."""
    default_verdict = {
        "step": step_name,
        "status": "WARN",
        "key_findings": ["Structured verdict missing; please review summary manually."],
        "thresholds": {},
        "actions": ["Re-run meeting or inspect transcript for final recommendation."],
        "source": "fallback",
    }
    if not summary:
        return default_verdict

    payload = None
    for line in reversed(summary.splitlines()):
        if "VERDICT_JSON:" in line:
            payload = line.split("VERDICT_JSON:", 1)[1].strip()
            break
    if not payload:
        return default_verdict

    try:
        parsed = json.loads(payload)
    except Exception:
        return default_verdict

    if not isinstance(parsed, dict):
        return default_verdict

    status = str(parsed.get("status", "WARN")).upper()
    if status not in {"PASS", "WARN", "FAIL"}:
        status = "WARN"

    key_findings = parsed.get("key_findings")
    if not isinstance(key_findings, list):
        key_findings = []
    key_findings = [str(x) for x in key_findings if str(x).strip()]
    if not key_findings:
        key_findings = ["No key findings supplied in structured verdict."]

    thresholds = parsed.get("thresholds", {})
    if not isinstance(thresholds, dict):
        thresholds = {}

    actions = parsed.get("actions", [])
    if not isinstance(actions, list):
        actions = []
    actions = [str(x) for x in actions if str(x).strip()]

    return {
        "step": str(parsed.get("step", step_name)),
        "status": status,
        "key_findings": key_findings,
        "thresholds": thresholds,
        "actions": actions,
        "source": "verdict_json",
    }


def _prediction_detail_text(context: WorkflowContext) -> str:
    """Build comprehensive per-predictor summary for LLM review."""
    lines: list[str] = []
    for name, res in context.prediction_results.items():
        if not res.success:
            lines.append(f"  - {name}: FAILED ({res.error_message})")
            continue
        parts = [f"{len(res.structure_paths)} structures"]
        parts.append(f"runtime={res.runtime_seconds:.0f}s")
        if res.scores:
            plddts = [s.plddt for s in res.scores if s.plddt]
            ptms = [s.ptm for s in res.scores if s.ptm]
            iptms = [s.iptm for s in res.scores if s.iptm]
            if plddts:
                parts.append(
                    f"pLDDT min={min(plddts):.1f} max={max(plddts):.1f} "
                    f"mean={sum(plddts)/len(plddts):.1f}"
                )
                # Count low-confidence models
                low = sum(1 for p in plddts if p < 50)
                if low:
                    parts.append(f"{low}/{len(plddts)} models pLDDT<50")
            if ptms:
                parts.append(f"pTM best={max(ptms):.3f}")
            if iptms:
                parts.append(f"ipTM best={max(iptms):.3f}")
            # Per-residue confidence stats for best model
            best_score = max(res.scores, key=lambda s: s.plddt or 0)
            if best_score.plddt_per_residue:
                pr = best_score.plddt_per_residue
                low_res = sum(1 for v in pr if v < 50)
                med_res = sum(1 for v in pr if 50 <= v < 70)
                high_res = sum(1 for v in pr if v >= 70)
                parts.append(
                    f"per-residue: {high_res} high(>=70), "
                    f"{med_res} medium(50-70), {low_res} low(<50)"
                )
        lines.append(f"  - {name}: {', '.join(parts)}")
    return "\n".join(lines)


def _evaluation_detail_text(context: WorkflowContext) -> str:
    """Build comprehensive per-predictor evaluation summary for LLM review."""
    lines: list[str] = []
    for name, ev in context.evaluation_results.items():
        parts: list[str] = []
        # Core metrics
        if ev.lddt is not None:
            parts.append(f"lDDT={ev.lddt:.3f}")
        if ev.tm_score is not None:
            parts.append(f"TM={ev.tm_score:.3f}")
        if ev.rmsd is not None:
            parts.append(f"RMSD={ev.rmsd:.2f}Å")
        if ev.gdt_ts is not None:
            parts.append(f"GDT-TS={ev.gdt_ts:.1f}")
        if ev.gdt_ha is not None:
            parts.append(f"GDT-HA={ev.gdt_ha:.1f}")
        if ev.qs_score is not None:
            parts.append(f"QS-score={ev.qs_score:.3f}")
        # Structural quality
        if ev.clash_score is not None:
            parts.append(f"clash={ev.clash_score:.1f}")
        if ev.clash_count is not None:
            parts.append(f"clashes={ev.clash_count}")
        if ev.contact_energy is not None:
            parts.append(f"contact_E={ev.contact_energy:.1f}")
        # Energy scores
        if ev.rosetta_total_score is not None:
            parts.append(f"Rosetta={ev.rosetta_total_score:.1f}")
        if ev.openmm_gbsa_energy_kj_mol is not None:
            parts.append(f"GBSA={ev.openmm_gbsa_energy_kj_mol:.0f}kJ/mol")
        if ev.foldx_ddg_kcal_mol is not None:
            parts.append(f"FoldX_ddG={ev.foldx_ddg_kcal_mol:.2f}kcal/mol")
        # MQA scores
        if ev.voromqa_score is not None:
            parts.append(f"VoroMQA={ev.voromqa_score:.3f}")
        if ev.cad_score is not None:
            parts.append(f"CAD={ev.cad_score:.3f}")
        # Surface & interface
        if ev.sasa_total is not None:
            parts.append(f"SASA={ev.sasa_total:.0f}Å²")
        if ev.interface_bsa_total is not None:
            parts.append(f"BSA={ev.interface_bsa_total:.0f}Å²")
        if ev.salt_bridge_count is not None:
            parts.append(f"salt_bridges={ev.salt_bridge_count}")
        # Disorder
        if ev.disorder_fraction is not None:
            parts.append(f"disorder={ev.disorder_fraction:.1%}")
        # Shape complementarity
        if ev.shape_complementarity is not None:
            parts.append(f"Sc={ev.shape_complementarity:.3f}")
        # Sequence recovery
        if ev.sequence_recovery is not None:
            parts.append(f"seq_recovery={ev.sequence_recovery:.1%}")

        lines.append(f"  - {name}: {', '.join(parts) or 'n/a'}")
    return "\n".join(lines)


# ── LLM-guided step agents ─────────────────────────────────────────

class LLMInputReviewAgent(BaseAgent, _LLMGuidedMixin):
    """LLM team reviews the input sequences after parsing.

    Validates protein sequences, identifies unusual features, predicts
    complexity, and suggests special considerations before prediction.
    The verdict is stored in ``context.step_verdicts["input_review"]``.
    """

    name = "llm_input_review"
    description = "LLM team reviews input sequences for quality and characteristics"

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
        if not context.sequences:
            return AgentResult.ok(context, "No sequences to review")
        try:
            context.with_job_dir()
            # Build detailed sequence info
            seq_details: list[str] = []
            for s in context.sequences:
                residue_counts: dict[str, int] = {}
                for aa in s.sequence:
                    residue_counts[aa] = residue_counts.get(aa, 0) + 1
                unusual = {k: v for k, v in residue_counts.items()
                           if k not in "ACDEFGHIKLMNPQRSTVWY"}
                detail = f"  - {s.id}: {len(s.sequence)} residues"
                if unusual:
                    detail += f", unusual residues: {unusual}"
                # Show composition hints
                cys_count = residue_counts.get("C", 0)
                pro_count = residue_counts.get("P", 0)
                gly_count = residue_counts.get("G", 0)
                if cys_count >= 4:
                    detail += f", {cys_count} Cys (possible disulfide bonds)"
                if pro_count / max(len(s.sequence), 1) > 0.1:
                    detail += f", Pro-rich ({pro_count}/{len(s.sequence)})"
                if gly_count / max(len(s.sequence), 1) > 0.15:
                    detail += f", Gly-rich (possible IDR/flexible)"
                seq_details.append(detail)

            is_multimer = (context.prediction_input and
                           context.prediction_input.is_multimer)
            total_len = sum(len(s.sequence) for s in context.sequences)

            agenda = (
                "Review the input protein sequences before prediction.\n\n"
                f"Number of sequences: {len(context.sequences)}\n"
                f"Total residues: {total_len}\n"
                f"Multimer: {is_multimer}\n\n"
                f"Sequence details:\n"
                + "\n".join(seq_details) + "\n\n"
                "Assess the input quality and identify any special considerations "
                "that should inform predictor selection and evaluation strategy."
            )
            questions = (
                "Are there any sequence quality issues (non-standard residues, "
                "very short/long sequences, missing regions)?",
                "What protein type does this appear to be (globular, membrane, "
                "IDP, antibody/nanobody, enzyme, multi-domain)?",
                "Are there structural features to watch for (disulfide bonds, "
                "metal binding, signal peptides, transmembrane helices)?",
                "Verdict: PASS (proceed normally), WARN (proceed with caveats), "
                "or FAIL (input needs correction)?",
            )
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                save_name="input_review",
                verdict_step="input_review",
            )
            context.extra["input_review"] = summary or ""
            context.step_verdicts["input_review"] = _parse_verdict_from_summary(
                summary or "", "input_review",
            )
            return AgentResult.ok(context, "Input review completed")
        except Exception as e:
            return AgentResult.fail(f"Input review failed: {e}", error=e)


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
            prev_summaries = [
                v for v in [context.extra.get("input_review")] if v
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
                save_name="planning_meeting",
                verdict_step="planning",
            )
            context.extra["plan"] = summary or ""
            context.step_verdicts["planning"] = _parse_verdict_from_summary(
                summary or "", "planning",
            )
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
            pred_summary = _prediction_detail_text(context)

            # Count successes/failures
            n_success = sum(1 for r in context.prediction_results.values() if r.success)
            n_fail = sum(1 for r in context.prediction_results.values() if not r.success)

            agenda = (
                "Review the prediction results and assess quality.\n\n"
                f"Predictors run: {len(context.prediction_results)} "
                f"({n_success} succeeded, {n_fail} failed)\n\n"
                f"Detailed prediction results:\n{pred_summary}\n\n"
                "Assess model quality using pLDDT distributions, pTM scores, "
                "and per-residue confidence. Identify low-confidence regions "
                "(pLDDT < 50), compare across predictors, flag any predictor "
                "whose output is suspect, and recommend whether structures "
                "need refinement before evaluation. Consider whether consensus "
                "across predictors increases confidence."
            )
            questions = (
                "Which predictions show the highest quality and should be prioritised? "
                "Cite specific pLDDT, pTM, and ipTM values.",
                "Are there quality red flags? List specific: low pLDDT regions, "
                "failed predictors, inconsistent predictions across methods.",
                "Should any structures be sent for refinement before evaluation? "
                "Specify which refinement method (AMBER, GalaxyRefine, ReFOLD).",
                "What quality assessment metrics should be applied during evaluation? "
                "Recommend specific thresholds.",
                "Verdict: PASS (predictions are reliable), WARN (proceed with "
                "caveats), or FAIL (re-prediction needed)?",
            )
            prev_summaries = [
                v for v in [
                    context.extra.get("input_review"),
                    context.extra.get("plan"),
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
                save_name="prediction_review",
                verdict_step="prediction_review",
            )
            context.extra["prediction_review"] = summary or ""
            context.step_verdicts["prediction_review"] = _parse_verdict_from_summary(
                summary or "", "prediction_review",
            )
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
            # Composite ranking
            lines = []
            for name, score in comp.ranking:
                lines.append(f"  - {name}: composite score {score:.3f}")
            rank_text = "\n".join(lines) if lines else "  (none)"

            # Comprehensive evaluation details
            eval_text = _evaluation_detail_text(context)

            # Prediction confidence recap
            pred_lines = []
            for name, pr in context.prediction_results.items():
                if pr.success and pr.scores:
                    plddts = [s.plddt for s in pr.scores if s.plddt]
                    ptms = [s.ptm for s in pr.scores if s.ptm]
                    parts = []
                    if plddts:
                        parts.append(f"best pLDDT={max(plddts):.1f}")
                    if ptms:
                        parts.append(f"best pTM={max(ptms):.3f}")
                    pred_lines.append(f"  - {name}: {', '.join(parts)}")
            pred_recap = "\n".join(pred_lines) if pred_lines else "  (none)"

            agenda = (
                "Review the comprehensive evaluation and comparison results.\n\n"
                f"Composite ranking (higher is better):\n{rank_text}\n\n"
                f"Full evaluation metrics:\n{eval_text}\n\n"
                f"Prediction confidence recap:\n{pred_recap}\n\n"
                "Interpret ALL metrics holistically. Assess structure quality "
                "using clash scores (< 10 excellent, > 40 severe), energy scores "
                "(Rosetta < -2 REU/res is well-folded), VoroMQA (> 0.4 good), "
                "and disorder fraction. Identify weaknesses and whether the "
                "structures are suitable for downstream applications."
            )
            questions = (
                "Which predictor produced the best structure? Cite specific "
                "metrics (lDDT, TM-score, clash score, energy) to justify.",
                "Are there quality concerns? Evaluate: clash score (< 10?), "
                "energy scores, VoroMQA/CAD, disorder fraction.",
                "What per-residue quality issues exist? Identify specific regions "
                "that are unreliable based on available per-residue data.",
                "Are the structures suitable for downstream use? Consider each: "
                "docking (need low clash + good interface), design (need reliable "
                "backbone), experimental interpretation (need good Ramachandran).",
                "Verdict: PASS (structures are reliable for downstream use), "
                "WARN (usable with caveats), or FAIL (refinement/re-prediction needed)?",
            )
            prev_summaries = [
                v for v in [
                    context.extra.get("input_review"),
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
                verdict_step="evaluation_review",
            )
            context.extra["evaluation_review"] = summary or ""
            context.step_verdicts["evaluation_review"] = _parse_verdict_from_summary(
                summary or "", "evaluation_review",
            )
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

            # Build comprehensive structure summary for refinement discussion
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
                        parts.append(f"RMSD={ev.rmsd:.2f}Å")
                    if ev.clash_score is not None:
                        parts.append(f"clash={ev.clash_score:.1f}")
                    if ev.voromqa_score is not None:
                        parts.append(f"VoroMQA={ev.voromqa_score:.3f}")
                    if ev.rosetta_total_score is not None:
                        parts.append(f"Rosetta={ev.rosetta_total_score:.1f}")
                struct_lines.append(f"  - {name}: {', '.join(parts)}")
            struct_text = "\n".join(struct_lines) if struct_lines else "  (no successful predictions)"

            # Include previous meeting context
            prev_eval = context.extra.get("evaluation_review", "")
            eval_snippet = prev_eval[:500] if prev_eval else "(no evaluation review available)"

            # Include evaluation verdict
            eval_verdict = context.step_verdicts.get("evaluation_review", {})
            eval_status = eval_verdict.get("status", "unknown")

            agenda = (
                "Discuss structure refinement strategy for the predicted structures.\n\n"
                f"Structures available:\n{struct_text}\n\n"
                f"Evaluation verdict: {eval_status}\n"
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
                verdict_step="refinement_review",
            )
            context.extra["refinement_review"] = summary or ""
            context.step_verdicts["refinement_review"] = _parse_verdict_from_summary(
                summary or "", "refinement_review",
            )
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
                verdict_step="mutagenesis_planning",
            )
            context.extra["mutagenesis_plan"] = summary or ""
            context.step_verdicts["mutagenesis_planning"] = _parse_verdict_from_summary(
                summary or "", "mutagenesis_planning",
            )
            return AgentResult.ok(context, "Mutagenesis planning completed")
        except Exception as e:
            return AgentResult.fail(f"Mutagenesis planning failed: {e}", error=e)


class LLMReportNarrativeAgent(BaseAgent, _LLMGuidedMixin):
    """Synthesise all meeting outcomes into an executive summary.

    Runs **before** ReportAgent.  The full-pipeline team reviews all
    step verdicts and meeting summaries to produce a single coherent
    narrative for the final report.
    """

    name = "llm_report_narrative"
    description = "LLM team synthesises all results into executive summary"

    def __init__(
        self,
        team_lead: LLMAgent | None = None,
        team_members: Sequence[LLMAgent] | None = None,
        num_rounds: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.team_lead = team_lead or DEFAULT_TEAM_LEAD
        self.team_members = team_members or FULL_PIPELINE_TEAM_MEMBERS
        self.num_rounds = num_rounds

    def run(self, context: WorkflowContext) -> AgentResult:
        # Only run if we have any meeting summaries to synthesise
        meeting_keys = [
            "input_review", "plan", "prediction_review",
            "evaluation_review", "refinement_review", "mutagenesis_plan",
        ]
        available = {k: context.extra[k] for k in meeting_keys
                     if context.extra.get(k)}
        if not available:
            return AgentResult.ok(context, "No meeting data to synthesise")

        try:
            context.with_job_dir()

            # Compile verdict summary
            verdict_lines = []
            for step, verdict in context.step_verdicts.items():
                status = verdict.get("status", "?")
                n_findings = len(verdict.get("key_findings", []))
                verdict_lines.append(
                    f"  - {step}: {status} ({n_findings} findings)"
                )
            verdict_text = "\n".join(verdict_lines) if verdict_lines else "  (no verdicts)"

            # Compile meeting summary snippets (truncated for context)
            meeting_text_parts = []
            for key, text in available.items():
                snippet = text[:400] + "..." if len(text) > 400 else text
                meeting_text_parts.append(f"  [{key}]: {snippet}")
            meetings_text = "\n".join(meeting_text_parts)

            # Ranking summary
            rank_text = "  (no ranking)"
            if context.comparison_result and context.comparison_result.ranking:
                rank_lines = [
                    f"  - {name}: {score:.3f}"
                    for name, score in context.comparison_result.ranking
                ]
                rank_text = "\n".join(rank_lines)

            agenda = (
                "Synthesise all pipeline results into a final executive summary "
                "for the report.\n\n"
                f"Step verdicts:\n{verdict_text}\n\n"
                f"Final ranking:\n{rank_text}\n\n"
                f"Meeting summaries:\n{meetings_text}\n\n"
                "Produce a concise, actionable executive summary that covers: "
                "overall pipeline outcome, key findings from each step, "
                "the recommended best structure and its suitability, "
                "remaining risks and caveats, and concrete next steps."
            )
            questions = (
                "What is the overall pipeline outcome? Summarise in one sentence.",
                "What are the top 3 most important findings across all steps?",
                "Is the best-ranked structure ready for downstream use? "
                "If not, what additional steps are needed?",
                "What are the key risks and limitations the user should know?",
                "Final verdict: PASS (results are reliable and actionable), "
                "WARN (usable with documented caveats), or FAIL (needs re-work)?",
            )
            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                num_rounds=self.num_rounds,
                summaries=tuple(available.values()),
                save_name="executive_summary",
                verdict_step="executive_summary",
            )
            context.extra["executive_summary"] = summary or ""
            context.step_verdicts["executive_summary"] = _parse_verdict_from_summary(
                summary or "", "executive_summary",
            )
            return AgentResult.ok(context, "Executive summary completed")
        except Exception as e:
            return AgentResult.fail(f"Executive summary failed: {e}", error=e)


# ── Mutagenesis pipeline LLM agents ──────────────────────────────

class LLMBaselineReviewAgent(BaseAgent, _LLMGuidedMixin):
    """LLM team reviews the wild-type baseline structure in detail.

    Identifies low-confidence regions, flags critical residues,
    and assesses suitability for mutagenesis.  Stores per-residue
    analysis in ``context.extra["baseline_review"]`` and low-confidence
    positions in ``context.extra["baseline_low_confidence_positions"]``.
    """

    name = "llm_baseline_review"
    description = "LLM team reviews wild-type baseline for mutagenesis suitability"

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
        if not context.prediction_results and not context.sequences:
            return AgentResult.ok(context, "No baseline data to review")
        try:
            context.with_job_dir()

            # ── Build per-residue pLDDT breakdown ────────────────
            per_residue_text = ""
            low_conf_positions: list[int] = []
            plddt_distribution = ""
            for name, pr in context.prediction_results.items():
                if not pr.success or not pr.scores:
                    continue
                best_score = max(pr.scores, key=lambda s: s.plddt or 0)
                if best_score.plddt_per_residue:
                    pr_vals = best_score.plddt_per_residue
                    low = [
                        (i + 1, v) for i, v in enumerate(pr_vals) if v < 70
                    ]
                    low_conf_positions = [pos for pos, _ in sorted(low, key=lambda x: x[1])]

                    # Detailed per-residue breakdown for LLM
                    very_high = sum(1 for v in pr_vals if v >= 90)
                    high = sum(1 for v in pr_vals if 70 <= v < 90)
                    medium = sum(1 for v in pr_vals if 50 <= v < 70)
                    low_count = sum(1 for v in pr_vals if v < 50)
                    mean_plddt = sum(pr_vals) / len(pr_vals)

                    plddt_distribution = (
                        f"\n  pLDDT distribution ({name}): "
                        f"mean={mean_plddt:.1f}, "
                        f"very_high(>=90)={very_high}, "
                        f"confident(70-90)={high}, "
                        f"low(50-70)={medium}, "
                        f"very_low(<50)={low_count}"
                    )

                    if low:
                        low_text = ", ".join(
                            f"{context.sequences[0].sequence[pos-1] if context.sequences and pos <= len(context.sequences[0].sequence) else '?'}{pos} (pLDDT={val:.1f})"
                            for pos, val in sorted(low, key=lambda x: x[1])[:20]
                        )
                        per_residue_text += f"\n  Low-confidence residues ({name}): {low_text}"
                    per_residue_text += (
                        f"\n  {name}: {very_high}/{len(pr_vals)} residues with pLDDT>=90"
                    )

            pred_summary = _prediction_detail_text(context)
            eval_summary = _evaluation_detail_text(context) if context.evaluation_results else ""

            # ── Build sequence info with residue numbering ───────
            seq_info = ""
            if context.sequences:
                seq = context.sequences[0]
                seq_info = (
                    f"\nSequence: {seq.id}, {len(seq.sequence)} residues"
                    f"\nFull sequence: {seq.sequence}"
                )

            agenda = (
                "Review the wild-type baseline structure in preparation for mutagenesis.\n\n"
                f"{seq_info}\n"
                f"\nPrediction results:\n{pred_summary}\n"
                f"{plddt_distribution}\n"
                f"{per_residue_text}\n"
            )
            if eval_summary:
                agenda += f"\nEvaluation metrics:\n{eval_summary}\n"

            agenda += (
                "\nYour task: analyse the per-residue pLDDT profile to identify regions "
                "suitable for stabilising mutations. Identify low-confidence regions "
                "(pLDDT < 70) that could benefit from mutations, flag functional residues "
                "that must NOT be mutated (Cys involved in disulfides, catalytic residues, "
                "conserved Gly/Pro in turns), and assess overall mutagenesis suitability."
            )

            questions = (
                "Analyse the per-residue pLDDT profile: which specific positions have "
                "low confidence and why? List each position with its pLDDT value and "
                "the amino acid at that position.",
                "Which residues are critical and must NOT be mutated? Look for "
                "Cys pairs (disulfide bonds), Pro in turns, Gly in tight loops, "
                "and any conserved functional motifs in the sequence.",
                "Based on the 3D model quality, which regions are well-folded "
                "(pLDDT > 90) and which are disordered/flexible? How does this "
                "inform mutation strategy?",
                "Verdict: is this structure suitable for computational mutagenesis? "
                "PASS (good quality, proceed), WARN (proceed with caveats), or "
                "FAIL (too unreliable for mutagenesis)?",
            )

            prev_summaries = [
                v for v in [
                    context.extra.get("input_review"),
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
                save_name="baseline_review",
                verdict_step="baseline_review",
            )
            context.extra["baseline_review"] = summary or ""
            context.extra["baseline_low_confidence_positions"] = low_conf_positions[:20]
            context.step_verdicts["baseline_review"] = _parse_verdict_from_summary(
                summary or "", "baseline_review",
            )
            return AgentResult.ok(context, "Baseline review completed")
        except Exception as e:
            return AgentResult.fail(f"Baseline review failed: {e}", error=e)


class LLMMutationSuggestionAgent(BaseAgent, _LLMGuidedMixin):
    """LLM team suggests specific mutations based on baseline review.

    Produces a structured mutation plan with positions, target AAs,
    and rationale.  The plan is parsed from ``MUTATION_PLAN_JSON:``
    in the summary.  Falls back to saturation at low-confidence
    positions if parsing fails.
    """

    name = "llm_mutation_suggestion"
    description = "LLM team suggests specific mutations with rationale"

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
        if not context.sequences:
            return AgentResult.ok(context, "No sequences for mutation suggestion")
        try:
            context.with_job_dir()
            sequence = context.sequences[0].sequence

            # ── Per-residue pLDDT from prediction ─────────────────
            low_conf = context.extra.get("baseline_low_confidence_positions", [])
            per_residue_text = ""
            plddt_distribution = ""
            for pred_name, pr in context.prediction_results.items():
                if not pr.success or not pr.scores:
                    continue
                best_score = max(pr.scores, key=lambda s: s.plddt or 0)
                if best_score.plddt_per_residue:
                    pr_vals = best_score.plddt_per_residue
                    mean_plddt = sum(pr_vals) / len(pr_vals)
                    very_high = sum(1 for v in pr_vals if v >= 90)
                    high = sum(1 for v in pr_vals if 70 <= v < 90)
                    medium = sum(1 for v in pr_vals if 50 <= v < 70)
                    low_count = sum(1 for v in pr_vals if v < 50)

                    plddt_distribution = (
                        f"\npLDDT distribution ({pred_name}): "
                        f"mean={mean_plddt:.1f}, "
                        f"very_high(>=90)={very_high}, "
                        f"confident(70-90)={high}, "
                        f"low(50-70)={medium}, "
                        f"very_low(<50)={low_count}"
                    )

                    # Show low-confidence positions with amino acid labels
                    low_residues = [
                        (i + 1, v) for i, v in enumerate(pr_vals) if v < 70
                    ]
                    if low_residues:
                        low_text = ", ".join(
                            f"{sequence[pos-1] if pos <= len(sequence) else '?'}{pos} (pLDDT={val:.1f})"
                            for pos, val in sorted(low_residues, key=lambda x: x[1])[:20]
                        )
                        per_residue_text = f"\nLow-confidence residues: {low_text}"
                break  # Use first successful predictor

            pred_summary = _prediction_detail_text(context)
            eval_summary = _evaluation_detail_text(context) if context.evaluation_results else ""

            agenda = (
                "Based on the baseline structure review, suggest 3-8 specific "
                "residue positions for mutation with target amino acids.\n\n"
                f"Protein: {context.sequences[0].id}, {len(sequence)} residues\n"
                f"Full sequence: {sequence}\n"
                f"\nPrediction results:\n{pred_summary}\n"
                f"{plddt_distribution}\n"
                f"{per_residue_text}\n"
            )
            if eval_summary:
                agenda += f"\nEvaluation metrics:\n{eval_summary}\n"

            agenda += (
                "\nConsider: stabilising mutations at low-confidence regions, "
                "surface mutations for improved solubility, conservative substitutions "
                "at semi-conserved positions, and avoid critical functional residues.\n\n"
                "You MUST provide your mutation plan in a specific format."
            )

            mutation_plan_rule = (
                'End your final summary with exactly one single-line JSON object prefixed by '
                '`MUTATION_PLAN_JSON:` using this schema: '
                '{"positions": [{"residue": <int>, "wt_aa": "<AA>", '
                '"targets": ["<AA>", ...], "rationale": "<why>"}], '
                '"strategy": "targeted|saturation|conservative", '
                '"rationale": "<overall strategy rationale>"}. '
                'Do not include markdown fences around this JSON.'
            )

            questions = (
                "Which 3-8 positions should be mutated? For each, cite the exact "
                "pLDDT value at that position, the amino acid identity, structural "
                "context (buried/surface, loop/helix), and propose 2-4 target AAs "
                "with rationale for each substitution.",
                "What is the overall mutation strategy? (targeted stabilisation, "
                "surface engineering, saturation at weak spots, conservative consensus)",
                "How many total variants will this produce? Is this experimentally tractable?",
                "What are the risks? Which mutations might destabilise the fold? "
                "Which positions should be avoided (e.g., Cys in disulfides, "
                "catalytic residues, conserved Gly/Pro)?",
                "Experimental validation strategy for the top candidates?",
            )

            prev_summaries = [
                v for v in [
                    context.extra.get("input_review"),
                    context.extra.get("plan"),
                    context.extra.get("prediction_review"),
                    context.extra.get("evaluation_review"),
                    context.extra.get("baseline_review"),
                ] if v
            ]

            summary = self._run_meeting_if_enabled(
                meeting_type="team",
                agenda=agenda,
                context=context,
                team_lead=self.team_lead,
                team_members=self.team_members,
                questions=questions,
                rules=(mutation_plan_rule,),
                num_rounds=self.num_rounds,
                summaries=tuple(prev_summaries),
                save_name="mutation_suggestion",
                verdict_step="mutation_suggestion",
            )

            context.extra["mutation_suggestion_raw"] = summary or ""
            context.step_verdicts["mutation_suggestion"] = _parse_verdict_from_summary(
                summary or "", "mutation_suggestion",
            )

            # Parse the structured mutation plan
            plan = _parse_mutation_plan_from_summary(summary or "", sequence)
            if plan:
                context.extra["mutation_suggestions"] = plan
                context.extra["mutation_suggestion_source"] = "llm"
                n_positions = len(plan["positions"])
                return AgentResult.ok(
                    context,
                    f"Mutation suggestion completed: {n_positions} positions proposed.",
                )
            else:
                # Fallback: saturation at top-5 low-confidence positions
                fallback_positions = low_conf[:5] if low_conf else []
                if fallback_positions:
                    n_fallback = len(fallback_positions)
                    warning_msg = (
                        f"LLM plan unparseable — falling back to saturation at {n_fallback} position(s). "
                        f"The LLM did not produce a valid MUTATION_PLAN_JSON footer."
                    )
                    logger.warning(warning_msg)
                    context.extra["mutation_suggestion_warning"] = warning_msg
                    fallback_plan = {
                        "positions": [
                            {
                                "residue": p,
                                "wt_aa": sequence[p - 1] if p <= len(sequence) else "X",
                                "targets": ["*"],
                                "rationale": "Fallback: low-confidence region",
                            }
                            for p in fallback_positions
                        ],
                        "strategy": "saturation",
                        "rationale": "Automated fallback — LLM plan could not be parsed.",
                    }
                    context.extra["mutation_suggestions"] = fallback_plan
                    context.extra["mutation_suggestion_source"] = "fallback"
                    return AgentResult.ok(
                        context,
                        f"Mutation suggestion fallback: saturation at {n_fallback} positions.",
                    )
                else:
                    context.extra["mutation_suggestions"] = None
                    context.extra["mutation_suggestion_source"] = "none"
                    return AgentResult.ok(
                        context,
                        "No mutation suggestions available (plan parse failed, no fallback positions).",
                    )

        except Exception as e:
            return AgentResult.fail(f"Mutation suggestion failed: {e}", error=e)


class LLMMutationResultsAgent(BaseAgent, _LLMGuidedMixin):
    """LLM team interprets mutation execution results.

    Reviews which mutations improved the structure, which failed,
    identifies top candidates, and suggests experimental validation.
    """

    name = "llm_mutation_results"
    description = "LLM team interprets mutation results and recommends candidates"

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
        comparison = context.extra.get("mutation_comparison", {})
        if not comparison:
            return AgentResult.ok(context, "No mutation comparison data to interpret")
        try:
            context.with_job_dir()

            # ── Sequence info ─────────────────────────────────────
            seq_info = ""
            sequence = ""
            if context.sequences:
                seq = context.sequences[0]
                sequence = seq.sequence
                seq_info = f"Protein: {seq.id}, {len(sequence)} residues\nFull sequence: {sequence}\n"

            # ── WT baseline metrics ───────────────────────────────
            wt_metrics = comparison.get("wt_metrics", {})
            wt_text = ""
            if wt_metrics:
                wt_parts = [f"mean pLDDT={wt_metrics.get('mean_plddt', 0):.1f}"]
                if wt_metrics.get("clash_score") is not None:
                    wt_parts.append(f"clash={wt_metrics['clash_score']:.1f}")
                if wt_metrics.get("sasa_total") is not None:
                    wt_parts.append(f"SASA={wt_metrics['sasa_total']:.0f}")
                wt_text = f"WT baseline: {', '.join(wt_parts)}\n"

            # ── Per-residue pLDDT distribution ────────────────────
            per_res = comparison.get("wt_per_residue_analysis", {})
            per_res_text = ""
            if per_res:
                dist = per_res.get("plddt_distribution", {})
                per_res_text = (
                    f"\nWT per-residue pLDDT: mean={per_res.get('mean_plddt', 0):.1f}, "
                    f"min={per_res.get('min_plddt', 0):.1f}, max={per_res.get('max_plddt', 0):.1f}\n"
                    f"  Distribution: very_high(>=90)={dist.get('very_high_gte_90', 0)}, "
                    f"confident(70-90)={dist.get('confident_70_90', 0)}, "
                    f"low(50-70)={dist.get('low_50_70', 0)}, "
                    f"very_low(<50)={dist.get('very_low_lt_50', 0)}\n"
                )
                low_positions = per_res.get("low_confidence_positions", [])
                if low_positions:
                    low_text = ", ".join(
                        f"{p['aa']}{p['pos']} (pLDDT={p['plddt']:.1f})"
                        for p in low_positions[:15]
                    )
                    per_res_text += f"  Low-confidence residues: {low_text}\n"

            # ── Format ranked mutations with full detail ──────────
            ranked = comparison.get("ranked_mutations", [])
            result_lines = []
            for i, r in enumerate(ranked[:20], 1):
                parts = [f"{r.get('mutation_code', '?')}"]
                parts.append(f"score={r.get('improvement_score', 0):.3f}")
                parts.append(f"delta_mean_pLDDT={r.get('delta_mean_plddt', 0):+.2f}")
                parts.append(f"delta_local_pLDDT={r.get('delta_local_plddt', 0):+.2f}")
                parts.append(f"mutant_mean_pLDDT={r.get('mean_plddt', 0):.1f}")
                if r.get("rmsd_to_base") is not None:
                    parts.append(f"RMSD={r['rmsd_to_base']:.2f}Å")
                if r.get("ost_lddt") is not None:
                    parts.append(f"OST_lDDT={r['ost_lddt']:.3f}")
                if r.get("ost_rmsd_ca") is not None:
                    parts.append(f"OST_RMSD_CA={r['ost_rmsd_ca']:.2f}Å")
                if r.get("clash_score") is not None:
                    parts.append(f"clash={r['clash_score']:.1f}")
                if r.get("sasa_total") is not None:
                    parts.append(f"SASA={r['sasa_total']:.0f}")
                result_lines.append(f"  {i}. {', '.join(parts)}")

            results_text = "\n".join(result_lines) if result_lines else "  (no results)"

            # ── OST metrics for best mutation ─────────────────────
            best_ost = comparison.get("best_ost_metrics", {})
            ost_text = ""
            if best_ost:
                ost_parts = []
                for k, v in best_ost.items():
                    ost_parts.append(f"{k}={v:.3f}")
                ost_text = f"\nBest mutation OST metrics vs WT: {', '.join(ost_parts)}\n"

            # ── Per-position summary ──────────────────────────────
            by_pos = comparison.get("by_position", {})
            pos_text = ""
            if by_pos:
                pos_lines = []
                for pos_key, pos_data in sorted(by_pos.items(), key=lambda x: int(x[0])):
                    best = pos_data.get("best", {})
                    n_total = len(pos_data.get("all", []))
                    n_beneficial = sum(
                        1 for m in pos_data.get("all", [])
                        if m.get("improvement_score", 0) > 0
                    )
                    pos_lines.append(
                        f"  Position {pos_key} ({best.get('original_aa', '?')}): "
                        f"{n_beneficial}/{n_total} beneficial, "
                        f"best={best.get('mutation_code', '?')} "
                        f"(score={best.get('improvement_score', 0):.3f})"
                    )
                pos_text = "\nPer-position summary:\n" + "\n".join(pos_lines[:15]) + "\n"

            agenda = (
                "Interpret the mutation scanning results and recommend candidates "
                "for experimental validation.\n\n"
                f"{seq_info}"
                f"{wt_text}"
                f"{per_res_text}"
                f"\nMutation summary:\n"
                f"  Total mutations tested: {comparison.get('total_mutations', 0)}\n"
                f"  Successful: {comparison.get('successful_count', 0)}\n"
                f"  Beneficial (score > 0): {comparison.get('beneficial_count', 0)}\n"
                f"  Detrimental (score < -0.5): {comparison.get('detrimental_count', 0)}\n"
                f"\nRanked mutations (top 20):\n{results_text}\n"
                f"{ost_text}"
                f"{pos_text}"
                "\nYour task: provide a thorough scientific interpretation. "
                "Assess which mutations are genuinely beneficial based on BOTH "
                "pLDDT improvement AND structural quality (RMSD, OST lDDT, clash). "
                "A mutation with high pLDDT but poor RMSD may have altered the fold. "
                "Identify the top 3-5 candidates for experimental validation and "
                "suggest a validation strategy."
            )

            questions = (
                "Which mutations are genuinely beneficial and why? Analyse each top "
                "candidate considering: delta pLDDT, RMSD to WT, OST lDDT, clash score, "
                "and the structural context of the mutated position.",
                "Why did certain mutations fail or show negative scores? "
                "Are there position-specific patterns? Which amino acid types "
                "consistently improve or worsen the structure?",
                "Top 3-5 candidates for experimental validation: rank and justify "
                "based on the full set of structural metrics, not just pLDDT.",
                "Any unexpected results that warrant further investigation? "
                "For example, positions where mutations improved pLDDT but "
                "increased RMSD, or vice versa.",
                "Recommended experimental validation strategy: "
                "expression, purification, stability assays (DSF, CD), "
                "activity assays, crystallography or cryo-EM?",
            )

            prev_summaries = [
                v for v in [
                    context.extra.get("input_review"),
                    context.extra.get("baseline_review"),
                    context.extra.get("mutation_suggestion_raw"),
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
                save_name="mutation_results_review",
                verdict_step="mutation_results_review",
            )
            context.extra["mutation_interpretation"] = summary or ""
            context.step_verdicts["mutation_results_review"] = _parse_verdict_from_summary(
                summary or "", "mutation_results_review",
            )
            return AgentResult.ok(context, "Mutation results interpretation completed")
        except Exception as e:
            return AgentResult.fail(f"Mutation results review failed: {e}", error=e)
