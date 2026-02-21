"""Computational agents for the agent-governed mutagenesis pipeline.

These agents handle mutation execution, comparison, and reporting.
They are used in the ``mutagenesis_post`` orchestrator mode, which
runs after the user has approved the mutation suggestions from Phase 1.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext

logger = logging.getLogger(__name__)

# Valid single-letter amino acids
_VALID_AAS = set("ACDEFGHIKLMNPQRSTVWY")


def _build_scanner(output_dir: Optional[Path] = None):
    """Create a MutationScanner with OpenStructure and evaluation metrics enabled."""
    from protein_design_hub.analysis.mutation_scanner import MutationScanner

    kwargs: dict[str, Any] = {
        "predictor": "esmfold_api",
        "evaluation_metrics": ["openmm_gbsa", "cad_score", "voromqa"],
        "run_openstructure_comprehensive": True,
    }
    if output_dir:
        kwargs["output_dir"] = output_dir

    try:
        return MutationScanner(**kwargs)
    except TypeError:
        # Older version without run_openstructure_comprehensive
        kwargs.pop("run_openstructure_comprehensive", None)
        scanner = MutationScanner(**kwargs)
        setattr(scanner, "run_openstructure_comprehensive", True)
        return scanner


def _extract_ost_metrics(extra_metrics: dict) -> dict:
    """Pull key OpenStructure metrics from extra_metrics for summary."""
    ost = extra_metrics.get("ost_comprehensive")
    if not isinstance(ost, dict):
        return {}
    g = ost.get("global", {})
    out: dict[str, Any] = {}
    for key in ("lddt", "rmsd_ca", "rmsd_backbone", "qs_score", "tm_score", "gdt_ts"):
        val = g.get(key)
        if isinstance(val, (int, float)):
            out[f"ost_{key}"] = float(val)
    cats = g.get("lddt_quality_categories", {})
    if cats:
        out["ost_lddt_categories"] = cats
    return out


class MutationExecutionAgent(BaseAgent):
    """Execute approved mutations via MutationScanner.

    Uses ``scan_position`` for saturation (targets=["*"]) and individual
    predictions for targeted mutations.  OpenStructure comprehensive
    comparison is enabled so each mutant is compared against the WT
    reference structure.

    Reads ``context.extra["approved_mutations"]`` — a list of dicts::

        [{"residue": int, "wt_aa": str, "targets": [str]}, ...]

    Results are stored in ``context.extra["mutation_results"]``.
    """

    name = "mutation_execution"
    description = "Execute approved mutations using MutationScanner"

    def run(self, context: WorkflowContext) -> AgentResult:
        approved = context.extra.get("approved_mutations", [])
        if not approved:
            low_conf = context.extra.get("baseline_low_confidence_positions", [])
            if low_conf:
                seq = context.sequences[0].sequence if context.sequences else ""
                approved = [
                    {
                        "residue": pos,
                        "wt_aa": seq[pos - 1] if pos <= len(seq) else "X",
                        "targets": ["*"],
                    }
                    for pos in low_conf[:5]
                ]
                logger.info(
                    "No approved mutations; falling back to saturation at "
                    "top-%d low-confidence positions", len(approved),
                )
            else:
                return AgentResult.fail(
                    "No approved mutations and no low-confidence positions available."
                )

        if not context.sequences:
            return AgentResult.fail("No sequences in context for mutation execution.")

        sequence = context.sequences[0].sequence

        try:
            job_dir = context.with_job_dir()
            mut_dir = job_dir / "mutagenesis" / "structures"
            mut_dir.mkdir(parents=True, exist_ok=True)

            scanner = _build_scanner(output_dir=mut_dir)

            all_results: List[Dict[str, Any]] = []
            wt_plddt: Optional[float] = None
            wt_path: Optional[Path] = None
            wt_plddt_per_residue: Optional[List[float]] = None

            # ── Predict WT once up front ─────────────────────────
            logger.info("Predicting wild-type baseline structure...")
            try:
                _wt_pdb, wt_vals, wt_out = scanner.predict_single(sequence, "wildtype")
                wt_plddt = sum(wt_vals) / len(wt_vals) if wt_vals else 0.0
                wt_path = wt_out
                wt_plddt_per_residue = wt_vals

                # WT baseline metrics (with OST self-comparison skipped
                # since reference=None gives us absolute metrics)
                wt_metrics = scanner.calculate_biophysical_metrics(wt_out)
                context.extra["mutation_wt_metrics"] = {
                    "mean_plddt": wt_plddt,
                    "plddt_per_residue": wt_vals,
                    "clash_score": wt_metrics.get("clash_score"),
                    "sasa_total": wt_metrics.get("sasa_total"),
                    "structure_path": str(wt_out),
                    "extra_metrics": wt_metrics.get("extra_metrics", {}),
                }
            except Exception as e:
                return AgentResult.fail(
                    f"Wild-type structure prediction failed: {e}", error=e,
                )

            # ── Execute mutations ────────────────────────────────
            total = sum(
                19 if m.get("targets") == ["*"] else len(m.get("targets", []))
                for m in approved
            )
            done = 0

            for entry in approved:
                position = entry["residue"]
                targets = entry.get("targets", ["*"])
                wt_aa = entry.get("wt_aa", sequence[position - 1] if position <= len(sequence) else "?")

                if targets == ["*"]:
                    # ── Full saturation via scan_position ────────
                    try:
                        def _sat_progress(current, total_sat, msg):
                            self._report_progress(
                                "mutation", msg, done + current, total,
                            )

                        sat_result = scanner.scan_position(
                            sequence, position, progress_callback=_sat_progress,
                        )
                        for m in sat_result.mutations:
                            d = m.to_dict()
                            # Enrich with OST summary
                            ost_summary = _extract_ost_metrics(
                                getattr(m, "extra_metrics", None) or {},
                            )
                            d.update(ost_summary)
                            all_results.append(d)
                        done += 19
                    except Exception as e:
                        logger.warning("Saturation at %d failed: %s", position, e)
                        all_results.append({
                            "position": position,
                            "original_aa": wt_aa,
                            "mutant_aa": "*",
                            "mutation_code": f"{wt_aa}{position}*",
                            "success": False,
                            "error_message": str(e),
                        })
                else:
                    # ── Targeted mutations ───────────────────────
                    for target_aa in targets:
                        if target_aa not in _VALID_AAS or target_aa == wt_aa:
                            continue
                        done += 1
                        mutation_code = f"{wt_aa}{position}{target_aa}"
                        try:
                            mutant_seq = (
                                sequence[: position - 1]
                                + target_aa
                                + sequence[position:]
                            )
                            _pdb, mut_vals, mut_out = scanner.predict_single(
                                mutant_seq, f"mut_{mutation_code}",
                            )

                            mut_mean = sum(mut_vals) / len(mut_vals) if mut_vals else 0.0
                            mut_local = (
                                mut_vals[position - 1]
                                if position <= len(mut_vals) else 0.0
                            )
                            wt_local = (
                                wt_plddt_per_residue[position - 1]
                                if wt_plddt_per_residue and position <= len(wt_plddt_per_residue)
                                else 0.0
                            )

                            # Biophysical metrics with WT as reference
                            biophys = scanner.calculate_biophysical_metrics(
                                mut_out, wt_path,
                            )

                            ost_summary = _extract_ost_metrics(
                                biophys.get("extra_metrics", {}),
                            )

                            d = {
                                "position": position,
                                "original_aa": wt_aa,
                                "mutant_aa": target_aa,
                                "mutation_code": mutation_code,
                                "mean_plddt": mut_mean,
                                "plddt_per_residue": mut_vals,
                                "local_plddt": mut_local,
                                "delta_mean_plddt": mut_mean - wt_plddt,
                                "delta_local_plddt": mut_local - wt_local,
                                "rmsd_to_base": biophys.get("rmsd"),
                                "tm_score_to_base": biophys.get("tm_score"),
                                "clash_score": biophys.get("clash_score"),
                                "sasa_total": biophys.get("sasa_total"),
                                "extra_metrics": biophys.get("extra_metrics", {}),
                                "structure_path": str(mut_out) if mut_out else None,
                                "success": True,
                            }
                            d.update(ost_summary)
                            all_results.append(d)

                            self._report_progress(
                                "mutation", mutation_code, done, total,
                            )
                        except Exception as e:
                            logger.warning("Mutation %s failed: %s", mutation_code, e)
                            all_results.append({
                                "position": position,
                                "original_aa": wt_aa,
                                "mutant_aa": target_aa,
                                "mutation_code": mutation_code,
                                "success": False,
                                "error_message": str(e),
                            })

            if not any(r.get("success") for r in all_results):
                return AgentResult.fail("All mutations failed during execution.")

            context.extra["mutation_results"] = all_results
            context.extra["mutation_wt_plddt"] = wt_plddt
            context.extra["mutation_wt_path"] = str(wt_path) if wt_path else None
            context.extra["mutation_wt_plddt_per_residue"] = wt_plddt_per_residue

            n_ok = sum(1 for r in all_results if r.get("success"))
            return AgentResult.ok(
                context,
                f"Executed {n_ok}/{len(all_results)} mutations successfully.",
            )
        except Exception as e:
            return AgentResult.fail(f"Mutation execution failed: {e}", error=e)


class MutationComparisonAgent(BaseAgent):
    """Compare mutants vs wild-type and rank by improvement score.

    Reads ``context.extra["mutation_results"]`` and stores a summary
    dict in ``context.extra["mutation_comparison"]``.  Includes OST
    metrics and per-residue pLDDT analysis where available.
    """

    name = "mutation_comparison"
    description = "Compare mutation results vs wild-type and rank"

    def run(self, context: WorkflowContext) -> AgentResult:
        results = context.extra.get("mutation_results", [])
        if not results:
            return AgentResult.fail("No mutation results to compare.")

        try:
            successful = [r for r in results if r.get("success")]
            failed = [r for r in results if not r.get("success")]

            # Compute improvement scores
            for r in successful:
                delta_mean = r.get("delta_mean_plddt", 0.0) or 0.0
                delta_local = r.get("delta_local_plddt", 0.0) or 0.0
                r["improvement_score"] = 0.6 * delta_mean + 0.4 * delta_local

            # Rank by improvement score
            ranked = sorted(
                successful,
                key=lambda x: x.get("improvement_score", 0),
                reverse=True,
            )

            # Classify
            beneficial = [r for r in ranked if r.get("improvement_score", 0) > 0]
            neutral = [
                r for r in ranked
                if -0.5 <= r.get("improvement_score", 0) <= 0.5
            ]
            detrimental = [r for r in ranked if r.get("improvement_score", 0) < -0.5]

            # Group by position
            by_position: Dict[int, Dict[str, Any]] = {}
            for r in ranked:
                pos = r["position"]
                if pos not in by_position:
                    by_position[pos] = {"best": r, "all": []}
                by_position[pos]["all"].append(r)

            # ── Build per-residue pLDDT analysis ─────────────────
            wt_per_res = context.extra.get("mutation_wt_plddt_per_residue", [])
            per_residue_analysis: Dict[str, Any] = {}
            if wt_per_res:
                low_conf = [
                    {"pos": i + 1, "aa": (context.sequences[0].sequence[i] if context.sequences else "?"), "plddt": v}
                    for i, v in enumerate(wt_per_res) if v < 70
                ]
                high_conf = sum(1 for v in wt_per_res if v >= 90)
                per_residue_analysis = {
                    "total_residues": len(wt_per_res),
                    "mean_plddt": sum(wt_per_res) / len(wt_per_res),
                    "min_plddt": min(wt_per_res),
                    "max_plddt": max(wt_per_res),
                    "high_confidence_count": high_conf,
                    "low_confidence_positions": sorted(low_conf, key=lambda x: x["plddt"]),
                    "plddt_distribution": {
                        "very_low_lt_50": sum(1 for v in wt_per_res if v < 50),
                        "low_50_70": sum(1 for v in wt_per_res if 50 <= v < 70),
                        "confident_70_90": sum(1 for v in wt_per_res if 70 <= v < 90),
                        "very_high_gte_90": sum(1 for v in wt_per_res if v >= 90),
                    },
                }

            # ── Collect OST metrics across mutations ─────────────
            ost_summary_best = {}
            if ranked:
                best = ranked[0]
                for k in ("ost_lddt", "ost_rmsd_ca", "ost_tm_score", "ost_qs_score", "ost_gdt_ts"):
                    val = best.get(k)
                    if val is not None:
                        ost_summary_best[k] = val

            comparison = {
                "total_mutations": len(results),
                "successful_count": len(successful),
                "failed_count": len(failed),
                "beneficial_count": len(beneficial),
                "neutral_count": len(neutral),
                "detrimental_count": len(detrimental),
                "best_overall": ranked[0] if ranked else None,
                "by_position": {
                    str(pos): data for pos, data in by_position.items()
                },
                "ranked_mutations": ranked,
                "wt_plddt": context.extra.get("mutation_wt_plddt"),
                "wt_per_residue_analysis": per_residue_analysis,
                "best_ost_metrics": ost_summary_best,
                "wt_metrics": context.extra.get("mutation_wt_metrics", {}),
            }

            context.extra["mutation_comparison"] = comparison
            return AgentResult.ok(
                context,
                f"Compared {len(successful)} mutations: "
                f"{len(beneficial)} beneficial, {len(detrimental)} detrimental.",
            )
        except Exception as e:
            return AgentResult.fail(f"Mutation comparison failed: {e}", error=e)


class MutagenesiReportAgent(BaseAgent):
    """Generate comprehensive mutagenesis report.

    Writes JSON and Markdown reports to ``{job_dir}/mutagenesis/``.
    Includes per-residue pLDDT analysis and OST metrics.
    """

    name = "mutagenesis_report"
    description = "Generate mutagenesis pipeline report"

    def run(self, context: WorkflowContext) -> AgentResult:
        try:
            job_dir = context.with_job_dir()
            report_dir = job_dir / "mutagenesis"
            report_dir.mkdir(parents=True, exist_ok=True)

            comparison = context.extra.get("mutation_comparison", {})
            interpretation = context.extra.get("mutation_interpretation", "")
            baseline_review = context.extra.get("baseline_review", "")

            # Write full JSON report
            report_data = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "job_id": context.job_id,
                "sequence": (
                    context.sequences[0].sequence if context.sequences else ""
                ),
                "sequence_id": (
                    context.sequences[0].id if context.sequences else ""
                ),
                "approved_mutations": context.extra.get("approved_mutations", []),
                "mutation_suggestions": context.extra.get("mutation_suggestions"),
                "mutation_suggestion_source": context.extra.get(
                    "mutation_suggestion_source", "unknown"
                ),
                "comparison": comparison,
                "wt_metrics": context.extra.get("mutation_wt_metrics", {}),
                "step_verdicts": dict(context.step_verdicts),
            }
            json_path = report_dir / "mutagenesis_report.json"
            json_path.write_text(json.dumps(report_data, indent=2, default=str))

            # Write Markdown summary
            md = self._build_markdown(context, comparison, interpretation, baseline_review)
            md_path = report_dir / "mutagenesis_summary.md"
            md_path.write_text(md)

            # Persist pipeline context for resume
            context_path = report_dir / "pipeline_context.json"
            context_data = {
                "job_id": context.job_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extra_keys": list(context.extra.keys()),
                "step_verdicts": dict(context.step_verdicts),
                "approved_mutations": context.extra.get("approved_mutations", []),
                "mutation_suggestion_source": context.extra.get(
                    "mutation_suggestion_source", "unknown"
                ),
            }
            context_path.write_text(json.dumps(context_data, indent=2, default=str))

            return AgentResult.ok(
                context, f"Mutagenesis report written to {report_dir}",
            )
        except Exception as e:
            return AgentResult.fail(f"Report generation failed: {e}", error=e)

    @staticmethod
    def _build_markdown(
        context: WorkflowContext,
        comparison: dict,
        interpretation: str,
        baseline_review: str,
    ) -> str:
        lines = [
            "# Mutagenesis Pipeline Report",
            f"\n**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Job ID:** {context.job_id}",
        ]

        if context.sequences:
            seq = context.sequences[0]
            lines.append(f"**Sequence:** {seq.id} ({len(seq.sequence)} residues)")

        # ── WT baseline summary ──────────────────────────────
        wt_metrics = context.extra.get("mutation_wt_metrics", {})
        if wt_metrics:
            lines.append("\n## Wild-Type Baseline")
            lines.append(f"- Mean pLDDT: {wt_metrics.get('mean_plddt', 0):.1f}")
            if wt_metrics.get("clash_score") is not None:
                lines.append(f"- Clash score: {wt_metrics['clash_score']:.1f}")
            if wt_metrics.get("sasa_total") is not None:
                lines.append(f"- SASA: {wt_metrics['sasa_total']:.0f}")

        # ── Per-residue pLDDT ────────────────────────────────
        per_res = comparison.get("wt_per_residue_analysis", {})
        if per_res:
            dist = per_res.get("plddt_distribution", {})
            lines.append("\n### Per-Residue pLDDT Distribution")
            lines.append(f"- Very high (>=90): {dist.get('very_high_gte_90', 0)}")
            lines.append(f"- Confident (70-90): {dist.get('confident_70_90', 0)}")
            lines.append(f"- Low (50-70): {dist.get('low_50_70', 0)}")
            lines.append(f"- Very low (<50): {dist.get('very_low_lt_50', 0)}")

            low_positions = per_res.get("low_confidence_positions", [])
            if low_positions:
                lines.append("\n**Low-confidence residues (pLDDT < 70):**")
                for p in low_positions[:15]:
                    lines.append(f"- {p['aa']}{p['pos']}: pLDDT = {p['plddt']:.1f}")

        # ── Mutation summary ─────────────────────────────────
        lines.append("\n## Mutation Results Summary")
        lines.append(f"- Total mutations tested: {comparison.get('total_mutations', 0)}")
        lines.append(f"- Successful: {comparison.get('successful_count', 0)}")
        lines.append(f"- Beneficial: {comparison.get('beneficial_count', 0)}")
        lines.append(f"- Detrimental: {comparison.get('detrimental_count', 0)}")

        best = comparison.get("best_overall")
        if best:
            lines.append(f"\n### Best Mutation")
            lines.append(
                f"- **{best.get('mutation_code', '?')}**: "
                f"score = {best.get('improvement_score', 0):.3f}, "
                f"delta pLDDT = {best.get('delta_mean_plddt', 0):+.2f}"
            )
            if best.get("ost_lddt") is not None:
                lines.append(f"- OST lDDT vs WT: {best['ost_lddt']:.3f}")
            if best.get("ost_rmsd_ca") is not None:
                lines.append(f"- OST RMSD(CA) vs WT: {best['ost_rmsd_ca']:.2f}")

        # ── Top mutations table ──────────────────────────────
        ranked = comparison.get("ranked_mutations", [])
        if ranked:
            lines.append("\n### Top Mutations")
            lines.append(
                "| Rank | Mutation | Score | Delta pLDDT | RMSD | OST lDDT | Clash |"
            )
            lines.append(
                "|------|----------|-------|-------------|------|----------|-------|"
            )
            for i, r in enumerate(ranked[:10], 1):
                rmsd = r.get("rmsd_to_base")
                ost_lddt = r.get("ost_lddt")
                clash = r.get("clash_score")
                rmsd_s = f"{rmsd:.2f}" if rmsd is not None else "-"
                ost_s = f"{ost_lddt:.3f}" if ost_lddt is not None else "-"
                clash_s = f"{clash:.1f}" if clash is not None else "-"
                lines.append(
                    f"| {i} | {r.get('mutation_code', '?')} | "
                    f"{r.get('improvement_score', 0):.3f} | "
                    f"{r.get('delta_mean_plddt', 0):+.2f} | "
                    f"{rmsd_s} | {ost_s} | {clash_s} |"
                )

        # ── Step verdicts ────────────────────────────────────
        if context.step_verdicts:
            lines.append("\n## Step Verdicts")
            for step, verdict in context.step_verdicts.items():
                status = verdict.get("status", "?")
                findings = verdict.get("key_findings", [])
                lines.append(f"\n### {step.replace('_', ' ').title()} [{status}]")
                for f in findings:
                    lines.append(f"- {f}")

        if interpretation:
            lines.append("\n## LLM Interpretation")
            lines.append(interpretation[:3000])

        if baseline_review:
            lines.append("\n## Baseline Review")
            lines.append(baseline_review[:1500])

        return "\n".join(lines)
