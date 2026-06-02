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
from protein_design_hub.analysis.protein_utils import (
    format_mutation_three_letter,
    format_residue_three_letter,
    three_letter,
)
from protein_design_hub.analysis.mutation_scoring import position_risk_summary
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
                    f"per-residue pLDDT: {high_res} high(>=70), "
                    f"{med_res} medium(50-70), {low_res} low(<50)"
                )
                # Show worst positions with amino acid labels
                _seq = (context.sequences[0].sequence
                        if context.sequences else "")
                low_positions = sorted(
                    [(i, v) for i, v in enumerate(pr) if v < 70],
                    key=lambda x: x[1],
                )[:8]
                if low_positions:
                    worst = ", ".join(
                        f"{format_residue_three_letter(_seq[i] if i < len(_seq) else '?', i+1)}({v:.0f})"
                        for i, v in low_positions
                    )
                    parts.append(f"low-conf residues: {worst}")
        lines.append(f"  - {name}: {', '.join(parts)}")
    return "\n".join(lines)


def _per_residue_summary(
    values: list[float],
    label: str,
    bad_threshold: float,
    bad_direction: str = "below",  # "below" or "above"
    top_n_worst: int = 5,
    sequence: str = "",
) -> str:
    """Summarise per-residue scores: mean/min/max + count of bad residues + worst positions."""
    if not values:
        return ""
    mean_v = sum(values) / len(values)
    min_v = min(values)
    max_v = max(values)
    if bad_direction == "below":
        bad_idx = [(i, v) for i, v in enumerate(values) if v < bad_threshold]
    else:
        bad_idx = [(i, v) for i, v in enumerate(values) if v > bad_threshold]
    n_bad = len(bad_idx)
    bad_idx_sorted = sorted(bad_idx, key=lambda x: x[1] if bad_direction == "below" else -x[1])
    worst_str = ""
    if bad_idx_sorted:
        worst_parts = []
        for i, v in bad_idx_sorted[:top_n_worst]:
            aa = sequence[i] if i < len(sequence) else "?"
            worst_parts.append(f"{aa}{i+1}({v:.2f})")
        worst_str = f"; worst: {', '.join(worst_parts)}"
    return (
        f"{label}: mean={mean_v:.3f}, min={min_v:.3f}, max={max_v:.3f}, "
        f"{n_bad}/{len(values)} {'<' if bad_direction=='below' else '>'}{bad_threshold}"
        f"{worst_str}"
    )


def _evaluation_detail_text(context: WorkflowContext, sequence: str = "") -> str:
    """Build full per-predictor evaluation text for LLM review.

    Exposes ALL available metrics grouped by category:
      - Reference-based accuracy (lDDT, TM, RMSD, GDT)
      - Structural geometry (clash, contact energy, MolProbity from metadata)
      - Energy scores (Rosetta, FoldX, OpenMM GBSA)
      - Model quality assessment (VoroMQA, CAD-score)
      - Surface & interface (SASA, BSA, salt bridges, shape complementarity)
      - Biophysical properties (disorder, sequence recovery)
      - Per-residue breakdowns (lDDT, VoroMQA, CAD, disorder)
      - Any extra fields stored in metadata by the tools
    """
    lines: list[str] = []
    for name, ev in context.evaluation_results.items():
        block: list[str] = [f"  [{name}]"]

        # ── Reference-based accuracy ─────────────────────────────
        ref_parts: list[str] = []
        if ev.lddt is not None:
            ref_parts.append(f"lDDT={ev.lddt:.3f}")
        if ev.tm_score is not None:
            ref_parts.append(f"TM-score={ev.tm_score:.3f}")
        if ev.rmsd is not None:
            ref_parts.append(f"RMSD={ev.rmsd:.2f}Å")
        if ev.gdt_ts is not None:
            ref_parts.append(f"GDT-TS={ev.gdt_ts:.1f}")
        if ev.gdt_ha is not None:
            ref_parts.append(f"GDT-HA={ev.gdt_ha:.1f}")
        if ev.qs_score is not None:
            ref_parts.append(f"QS-score={ev.qs_score:.3f}")
        if ref_parts:
            block.append(f"    Accuracy: {', '.join(ref_parts)}")

        # ── Structural geometry ──────────────────────────────────
        geom_parts: list[str] = []
        if ev.clash_score is not None:
            qual = "excellent" if ev.clash_score < 10 else ("good" if ev.clash_score < 20 else ("acceptable" if ev.clash_score < 40 else "POOR"))
            geom_parts.append(f"clash_score={ev.clash_score:.1f} ({qual})")
        if ev.clash_count is not None:
            geom_parts.append(f"clash_pairs={ev.clash_count}")
        if ev.contact_energy is not None:
            geom_parts.append(f"contact_energy={ev.contact_energy:.1f}")
        if ev.contact_energy_per_residue is not None:
            geom_parts.append(f"contact_E/res={ev.contact_energy_per_residue:.3f}")
        # MolProbity data from metadata (if OST/phenix ran)
        mp = ev.metadata.get("clash_score", {}) if ev.metadata else {}
        for mp_key in ("molprobity_score", "molprobity_clashscore",
                       "molprobity_rama_favored_pct", "molprobity_rama_outliers_pct",
                       "molprobity_rotamer_outliers_pct", "molprobity_rms_bonds",
                       "molprobity_rms_angles"):
            val = mp.get(mp_key)
            if val is not None:
                geom_parts.append(f"{mp_key.replace('molprobity_','MP_')}={val:.2f}")
        # Check other metadata keys for MolProbity
        for meta_key in ("molprobity", "observed_scoring"):
            mp2 = ev.metadata.get(meta_key, {}) if ev.metadata else {}
            for sub_key in ("molprobity_score", "molprobity_rama_favored_pct",
                            "molprobity_rama_outliers_pct", "molprobity_rotamer_outliers_pct"):
                val = mp2.get(sub_key)
                if val is not None:
                    geom_parts.append(f"MP_{sub_key.replace('molprobity_','')}={val:.2f}")
        if geom_parts:
            block.append(f"    Geometry: {', '.join(geom_parts)}")

        # ── Energy scores ────────────────────────────────────────
        energy_parts: list[str] = []
        if ev.rosetta_total_score is not None:
            per_res = ""
            n_res = len(sequence) if sequence else 0
            if n_res > 0:
                per_res = f" ({ev.rosetta_total_score/n_res:.2f} REU/res)"
            energy_parts.append(f"Rosetta={ev.rosetta_total_score:.1f} REU{per_res}")
        if ev.rosetta_score_jd2_total_score is not None and ev.rosetta_score_jd2_total_score != ev.rosetta_total_score:
            energy_parts.append(f"Rosetta_JD2={ev.rosetta_score_jd2_total_score:.1f}")
        if ev.rosetta_cartesian_ddg is not None:
            energy_parts.append(f"Rosetta_cart_ddG={ev.rosetta_cartesian_ddg:.2f} REU")
        if ev.openmm_potential_energy_kj_mol is not None:
            energy_parts.append(f"OpenMM_PE={ev.openmm_potential_energy_kj_mol:.0f} kJ/mol")
        if ev.openmm_gbsa_energy_kj_mol is not None:
            energy_parts.append(f"OpenMM_GBSA={ev.openmm_gbsa_energy_kj_mol:.0f} kJ/mol")
        if ev.foldx_ddg_kcal_mol is not None:
            stability = "stabilising" if ev.foldx_ddg_kcal_mol < -1 else ("destabilising" if ev.foldx_ddg_kcal_mol > 1 else "neutral")
            energy_parts.append(f"FoldX_ddG={ev.foldx_ddg_kcal_mol:.2f} kcal/mol ({stability})")
        if energy_parts:
            block.append(f"    Energy: {', '.join(energy_parts)}")

        # ── Model quality assessment ─────────────────────────────
        mqa_parts: list[str] = []
        if ev.voromqa_score is not None:
            qual = "good" if ev.voromqa_score > 0.4 else ("borderline" if ev.voromqa_score > 0.3 else "POOR")
            mqa_parts.append(f"VoroMQA={ev.voromqa_score:.3f} ({qual})")
        if ev.voromqa_residue_count is not None:
            mqa_parts.append(f"VoroMQA_residues={ev.voromqa_residue_count}")
        if ev.cad_score is not None:
            mqa_parts.append(f"CAD-score={ev.cad_score:.3f}")
        if mqa_parts:
            block.append(f"    MQA: {', '.join(mqa_parts)}")

        # ── Surface & interface ──────────────────────────────────
        surf_parts: list[str] = []
        if ev.sasa_total is not None:
            surf_parts.append(f"SASA={ev.sasa_total:.0f}Å²")
        if ev.interface_bsa_total is not None:
            quality_bsa = "high-affinity" if ev.interface_bsa_total > 1200 else ("moderate" if ev.interface_bsa_total > 600 else "low")
            surf_parts.append(f"interface_BSA={ev.interface_bsa_total:.0f}Å² ({quality_bsa})")
        if ev.salt_bridge_count is not None:
            surf_parts.append(f"salt_bridges={ev.salt_bridge_count}")
        if ev.salt_bridge_count_interchain is not None:
            surf_parts.append(f"salt_bridges_interchain={ev.salt_bridge_count_interchain}")
        if ev.shape_complementarity is not None:
            surf_parts.append(f"shape_complementarity={ev.shape_complementarity:.3f}")
        if ev.interface_residues_a is not None:
            surf_parts.append(f"interface_res_A={ev.interface_residues_a}")
        if ev.interface_residues_b is not None:
            surf_parts.append(f"interface_res_B={ev.interface_residues_b}")
        if surf_parts:
            block.append(f"    Surface/Interface: {', '.join(surf_parts)}")

        # ── Biophysical properties ───────────────────────────────
        biophys_parts: list[str] = []
        if ev.disorder_fraction is not None:
            biophys_parts.append(f"disorder_fraction={ev.disorder_fraction:.1%}")
        if ev.disorder_regions:
            n_regions = len(ev.disorder_regions)
            lengths = [r.get("length", r.get("end", 0) - r.get("start", 0)) for r in ev.disorder_regions]
            total_disordered = sum(lengths)
            biophys_parts.append(f"disorder_regions={n_regions} (total {total_disordered} residues)")
        if ev.sequence_recovery is not None:
            biophys_parts.append(f"seq_recovery={ev.sequence_recovery:.1%}")
        if biophys_parts:
            block.append(f"    Biophysics: {', '.join(biophys_parts)}")

        # ── Per-residue breakdowns ───────────────────────────────
        # These give agents spatial context on where problems are
        if ev.lddt_per_residue:
            s = _per_residue_summary(ev.lddt_per_residue, "lDDT/res", 0.5, "below", sequence=sequence)
            if s:
                block.append(f"    {s}")
        if ev.voromqa_per_residue:
            s = _per_residue_summary(ev.voromqa_per_residue, "VoroMQA/res", 0.4, "below", sequence=sequence)
            if s:
                block.append(f"    {s}")
        if ev.cad_score_per_residue:
            s = _per_residue_summary(ev.cad_score_per_residue, "CAD/res", 0.5, "below", sequence=sequence)
            if s:
                block.append(f"    {s}")
        if ev.disorder_per_residue:
            s = _per_residue_summary(ev.disorder_per_residue, "disorder/res", 0.5, "above", sequence=sequence)
            if s:
                block.append(f"    {s}")

        if len(block) == 1:
            block.append("    (no metrics computed)")
        lines.append("\n".join(block))
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

            # Comprehensive evaluation details (pass sequence for per-residue position labels)
            _seq = context.sequences[0].sequence if context.sequences else ""
            eval_text = _evaluation_detail_text(context, sequence=_seq)

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
                f"Full evaluation metrics (grouped by category):\n{eval_text}\n\n"
                f"Prediction confidence recap:\n{pred_recap}\n\n"
                "The evaluation data above contains ALL available metrics, grouped as: "
                "Accuracy (lDDT/TM/RMSD), Geometry (clash/MolProbity/contact energy), "
                "Energy (Rosetta/FoldX/OpenMM GBSA), MQA (VoroMQA/CAD), "
                "Surface/Interface (SASA/BSA/salt bridges/shape complementarity), "
                "Biophysics (disorder/sequence recovery), and per-residue breakdowns "
                "(shows mean/min/max + worst positions for lDDT, VoroMQA, CAD, disorder).\n\n"
                "Each team member should interpret the metrics within their domain. "
                "Cross-domain conflicts are especially valuable: e.g. if Rosetta energy "
                "is poor but VoroMQA is good, or if clash score is low but lDDT is poor."
            )
            questions = (
                "Which predictor produced the best structure? Justify using AT LEAST "
                "3 independent metrics from different categories (e.g. lDDT + VoroMQA + "
                "clash score). Do NOT rely on pLDDT alone.",
                "Are there specific geometric problems? Report: clash score (threshold<10), "
                "contact energy, and any MolProbity data (Ramachandran favored %, "
                "rotamer outliers %, cbeta deviations) if available.",
                "What do the energy scores say? Interpret Rosetta (REU/res: well-folded "
                "if <-2.5), OpenMM GBSA (more negative = better solvation), and FoldX ddG "
                "(if available). Do energy and structural quality metrics agree?",
                "What do the per-residue breakdowns reveal? Identify specific positions "
                "where lDDT/VoroMQA/CAD scores are poor (below threshold) — these are "
                "the regions agents should flag for refinement or mutagenesis.",
                "Are the structures suitable for downstream use? Assess separately for: "
                "docking (need clash<10, BSA>800Å², shape_Sc>0.6), mutagenesis (need "
                "reliable backbone = lDDT>0.8), experimental validation (need good geometry).",
                "Verdict: PASS, WARN, or FAIL — with specific metric thresholds that "
                "informed the decision.",
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
            _bl_seq = context.sequences[0].sequence if context.sequences else ""
            eval_summary = _evaluation_detail_text(context, sequence=_bl_seq) if context.evaluation_results else ""

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
                "\nYour task: perform a comprehensive structural analysis of the wild-type "
                "baseline to inform mutagenesis strategy. Go beyond pLDDT — reason about "
                "the structural context of each low-confidence region: is it a flexible "
                "loop, a buried core residue, a surface patch, an interface? Identify "
                "residues that are UNSAFE to mutate (disulfide-forming Cys, catalytic "
                "residues, conserved Gly/Pro in tight turns) and regions where mutations "
                "are LIKELY to improve stability (high B-factor equivalents = low pLDDT "
                "in loops, exposed charged residues, buried polar residues).\n"
                "Important: the Structural Biologist, Biophysicist, and ML Specialist "
                "should each bring their domain-specific lens. Disagreements about which "
                "positions to target are scientifically valuable — resolve them with "
                "specific evidence."
            )

            questions = (
                "Analyse the per-residue pLDDT profile in structural context: for each "
                "low-confidence position (<70), state: (a) amino acid identity, (b) "
                "pLDDT value, (c) structural element (loop/helix/strand/turn), and "
                "(d) likely reason for low confidence (flexibility, disorder, crystal "
                "packing artifact, or prediction limitation). Give specific positions.",
                "Which residues are critical and must NOT be mutated — and why? "
                "Look beyond Cys pairs: include Pro in turns (φ≈-60°), Gly in tight "
                "loops (needs smallest residue), charged residues forming salt bridges "
                "(Asp-Arg pairs within 4Å), and any NxS/NxT glycosylation sites. "
                "For each, state the structural rationale.",
                "From a biophysical perspective: which regions suggest thermodynamic "
                "weakness? Look for: exposed hydrophobic patches, buried polar residues "
                "(fa_sol penalty), strained backbone geometry, and unsatisfied H-bond "
                "donors/acceptors. These are priority targets for stabilising mutations.",
                "Do the ML confidence scores (pLDDT) and the energy metrics agree on "
                "which regions are problematic? If they disagree — high pLDDT but poor "
                "energy, or vice versa — explain what that means for mutagenesis.",
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
                            f"{format_residue_three_letter(sequence[pos-1] if pos <= len(sequence) else '?', pos)} (pLDDT={val:.1f})"
                            for pos, val in sorted(low_residues, key=lambda x: x[1])[:20]
                        )
                        per_residue_text = f"\nLow-confidence residues: {low_text}"
                break  # Use first successful predictor

            pred_summary = _prediction_detail_text(context)
            _ms_seq = sequence if sequence else ""
            eval_summary = _evaluation_detail_text(context, sequence=_ms_seq) if context.evaluation_results else ""

            # Data-backed mutability risk for the candidate (low-confidence)
            # positions, so the "avoid critical residues" guidance below is
            # grounded in an actual signal rather than a single-sequence guess.
            risk_text = ""
            cand_positions = [p for p in low_conf if isinstance(p, int)][:20]
            if sequence and cand_positions:
                risks = position_risk_summary(sequence, cand_positions)
                flagged = [
                    f"{format_residue_three_letter(d['wt_aa'], p)} (risk {d['risk']}: {'; '.join(d['notes'])})"
                    for p, d in sorted(risks.items()) if d.get("notes")
                ]
                if flagged:
                    risk_text = (
                        "\nPosition mutability risk (higher = more conserved/structurally "
                        "critical WT residue — prefer NOT to mutate these): "
                        + ", ".join(flagged) + "\n"
                    )

                # ESM-2 zero-shot conservation: positions where few substitutions
                # are tolerated are evolutionarily constrained → avoid mutating.
                import os as _os
                if _os.environ.get("PDHUB_ESM2", "1") != "0":
                    try:
                        from protein_design_hub.analysis.esm2_zero_shot import (
                            ESM2VariantScorer, get_esm2_scorer,
                        )
                        if ESM2VariantScorer.is_available():
                            sc = get_esm2_scorer()
                            cons = []
                            for p in cand_positions:
                                if not (1 <= p <= len(sequence)):
                                    continue
                                deltas = sc.score_position(sequence, p)
                                wt = sequence[p - 1].upper()
                                tolerated = sum(1 for a, d in deltas.items() if a != wt and d >= -2.0)
                                tag = "CONSERVED" if tolerated <= 3 else ("plastic" if tolerated >= 10 else "moderate")
                                cons.append(f"{format_residue_three_letter(wt, p)}={tolerated} tolerated subs [{tag}]")
                            if cons:
                                risk_text += (
                                    "\nESM-2 zero-shot tolerance per position (fewer tolerated "
                                    "substitutions = more conserved, higher mutation risk): "
                                    + ", ".join(cons) + "\n"
                                )
                    except Exception as _esm_err:  # noqa: BLE001
                        logger.warning("ESM-2 suggestion annotation skipped: %s", _esm_err)

            agenda = (
                "Based on the baseline structure review, suggest 3-8 specific "
                "residue positions for mutation with target amino acids.\n\n"
                f"Protein: {context.sequences[0].id}, {len(sequence)} residues\n"
                f"Full sequence: {sequence}\n"
                f"\nPrediction results:\n{pred_summary}\n"
                f"{plddt_distribution}\n"
                f"{per_residue_text}\n"
                f"{risk_text}"
            )
            if eval_summary:
                agenda += f"\nEvaluation metrics:\n{eval_summary}\n"

            agenda += (
                "\nConsider: stabilising mutations at low-confidence regions, "
                "surface mutations for improved solubility, conservative substitutions "
                "at semi-conserved positions, and avoid critical functional residues.\n\n"
                "Key debate to resolve: the Protein Engineer should propose mutations "
                "from a structural/stability perspective; the ML Specialist should "
                "evaluate the likelihood each proposed mutation is natural (i.e., "
                "appears in homologous sequences); the Biophysicist should predict "
                "the thermodynamic effect; the Scientific Critic should challenge "
                "any proposal lacking quantitative support. The team MUST reconcile "
                "disagreements before the final mutation plan is produced.\n\n"
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
                "Which 3-8 positions should be mutated? For each, cite: (a) exact "
                "pLDDT value, (b) amino acid identity, (c) structural context "
                "(buried/surface, secondary structure element), (d) 2-4 target AAs "
                "with rationale for each, and (e) predicted effect on stability "
                "(stabilising/destabilising/neutral with reasoning).",
                "For each proposed mutation, what is the evolutionary evidence? "
                "Is the target AA found in homologous sequences at that position? "
                "High conservation at the WT position = high-risk mutation.",
                "What is the overall mutation strategy and why? Specifically: "
                "does the team agree on the strategy? If the Protein Engineer and "
                "ML Specialist disagree, state the disagreement and which view prevailed.",
                "How many total variants will this produce? Is this experimentally tractable? "
                "Consider that screening >200 variants requires deep mutational scanning; "
                "<20 variants can be validated individually by DSF and SEC.",
                "Risk assessment: which of the proposed mutations carry the highest "
                "risk of fold disruption? For each high-risk mutation, what is the "
                "safety check (TM-score cutoff, pLDDT threshold, ddG limit)?",
                "Experimental validation strategy for the top 3 candidates?",
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
                        f"{format_residue_three_letter(p['aa'], p['pos'])} (pLDDT={p['plddt']:.1f})"
                        for p in low_positions[:15]
                    )
                    per_res_text += f"  Low-confidence residues: {low_text}\n"

            # ── Format ranked mutations with full detail ──────────
            ranked = comparison.get("ranked_mutations", [])
            result_lines = []
            for i, r in enumerate(ranked[:20], 1):
                parts = [f"{format_mutation_three_letter(r.get('mutation_code', '?'))}"]
                parts.append(f"score={r.get('improvement_score', 0):.3f}")
                _comp = r.get("score_components", {})
                if _comp.get("ddg_kcal") is not None:
                    parts.append(f"ddG={_comp['ddg_kcal']:+.2f}kcal/mol")
                if r.get("esm2_delta_ll") is not None:
                    parts.append(f"ESM2_dLL={r['esm2_delta_ll']:+.2f}")
                if r.get("am_score") is not None:
                    parts.append(f"AlphaMissense={r['am_score']:.2f}")
                if r.get("flags"):
                    parts.append(f"flags=[{'; '.join(r['flags'])}]")
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
                        f"  Position {pos_key} ({three_letter(best.get('original_aa', '?'))}): "
                        f"{n_beneficial}/{n_total} beneficial, "
                        f"best={format_mutation_three_letter(best.get('mutation_code', '?'))} "
                        f"(score={best.get('improvement_score', 0):.3f})"
                    )
                pos_text = "\nPer-position summary:\n" + "\n".join(pos_lines[:15]) + "\n"

            # Compute overall beneficial rate for context
            total = comparison.get("total_mutations", 0)
            beneficial = comparison.get("beneficial_count", 0)
            beneficial_rate = (beneficial / total * 100) if total > 0 else 0

            # Combinatorial candidate (additive epistasis estimate), if any.
            combo = comparison.get("combinatorial_candidate")
            combo_text = ""
            if combo:
                _codes = ", ".join(
                    format_mutation_three_letter(c) for c in combo.get("mutations", [])
                )
                combo_text = (
                    f"\nCombinatorial candidate (ADDITIVE estimate, epistasis untested): "
                    f"{_codes} | additive ΔΔG={combo.get('additive_ddg_kcal', 0):+.2f} kcal/mol. "
                    f"{combo.get('note', '')}\n"
                )

            agenda = (
                "Interpret the mutation scanning results and recommend candidates "
                "for experimental validation.\n\n"
                "Scoring note: the improvement score is a COMPOSITE PHYSICS score "
                "(stability ΔΔG + fold agreement from OST lDDT/RMSD + clash/PTM "
                "penalties). pLDDT is only a confidence gate, NOT the objective. "
                "Each ranked entry shows its ddG and any risk flags.\n"
                f"{seq_info}"
                f"{wt_text}"
                f"{per_res_text}"
                f"\nMutation summary:\n"
                f"  Total mutations tested: {total}\n"
                f"  Successful: {comparison.get('successful_count', 0)}\n"
                f"  Beneficial (composite score > 0.25): {beneficial} ({beneficial_rate:.0f}%)\n"
                f"  Detrimental (composite score < -0.25): {comparison.get('detrimental_count', 0)}\n"
                f"\nRanked mutations (top 20):\n{results_text}\n"
                f"{combo_text}"
                f"{ost_text}"
                f"{pos_text}"
                "\nYour task: provide a rigorous multi-perspective interpretation. "
                "This is NOT just 'pick the highest-scoring mutations'. Each team "
                "member must bring their domain lens:\n"
                "  Protein Engineer: structural rationale for top mutations — what "
                "    chemistry explains improvement at each position?\n"
                "  ML Specialist: are the pLDDT improvements real or an artifact "
                "    of the predictor being biased toward certain AAs?\n"
                "  Biophysicist: what experimental assay would definitively confirm "
                "    stability improvement for each top candidate?\n"
                "  Scientific Critic: which of the 'beneficial' mutations might "
                "    be false positives? What's the risk of each top candidate?\n"
                "Reconcile any conflicts — a mutation is only recommended if at "
                "least 3 of 4 domain perspectives support it."
            )

            questions = (
                "Which mutations are genuinely beneficial — not just highest-scoring? "
                "For each top candidate: (a) delta_mean_pLDDT, (b) delta_local_pLDDT, "
                "(c) RMSD to WT (high RMSD = fold change, not just local improvement), "
                "(d) OST lDDT (independent validation), (e) clash score, and "
                "(f) structural rationale for *why* this substitution works.",
                "Flag any suspiciously high-scoring mutations. A mutation that dramatically "
                "improves pLDDT but also increases RMSD>1.5Å likely reflects a fold change "
                "rather than true stabilisation. Are any top candidates in this category?",
                "Which positions show the most consistent improvement across multiple "
                "substitutions? Position-consistent improvement is stronger evidence "
                "than a single outlier mutation.",
                "Top 3-5 candidates for experimental validation — ranked by the team's "
                "combined assessment (structural + ML + biophysics). For each, state: "
                "(a) the recommended substitution, (b) the primary evidence, "
                "(c) the main risk, and (d) the specific assay to validate it.",
                "Recommended experimental validation order and strategy: which assay "
                "first (thermal shift / DSF for Tm, SEC for oligomeric state, "
                "CD for secondary structure, activity assay for function)? "
                "What result would confirm success vs. failure for each candidate?",
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
