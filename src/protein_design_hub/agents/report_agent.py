"""Report generation agent."""

import json

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.io.writers.report_writer import ReportWriter


class ReportAgent(BaseAgent):
    """
    Agent responsible for writing comparison summary, HTML report,
    per-predictor/evaluation artifacts, and LLM meeting summaries.
    """

    name = "report"
    description = "Generate comparison and HTML reports"

    # Keys in context.extra that hold LLM meeting summaries
    _MEETING_KEYS = [
        ("plan", "Planning Meeting"),
        ("prediction_review", "Prediction Review"),
        ("evaluation_review", "Evaluation Review"),
        ("refinement_review", "Refinement Review"),
        ("mutagenesis_plan", "Mutagenesis & Design Planning"),
    ]

    def __init__(self, settings: Settings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings or get_settings()
        self.report_writer = ReportWriter()

    def run(self, context: WorkflowContext) -> AgentResult:
        if context.comparison_result is None:
            return AgentResult.fail("No comparison result; run ComparisonAgent first")
        if context.job_dir is None:
            return AgentResult.fail("job_dir is not set")

        try:
            self._report_progress("reporting", "Generating reports", 5, 5)
            self.report_writer.output_dir = context.job_dir
            comp = context.comparison_result

            (context.job_dir / "evaluation").mkdir(exist_ok=True)
            self.report_writer.write_comparison_report(
                comp,
                context.job_dir / "evaluation" / "comparison_summary.json",
            )

            if self.settings.output.generate_report:
                (context.job_dir / "report").mkdir(exist_ok=True)
                self.report_writer.write_html_report(
                    comp,
                    context.job_dir / "report" / "report.html",
                )

            for predictor_name, pred_result in comp.prediction_results.items():
                if pred_result.success:
                    self.report_writer.write_prediction_report(
                        pred_result,
                        context.job_dir / predictor_name / "scores.json",
                    )
                    self.report_writer.write_status_file(
                        predictor_name,
                        pred_result.success,
                        pred_result.runtime_seconds,
                        pred_result.error_message,
                        context.job_dir,
                    )

            for predictor_name, eval_result in comp.evaluation_results.items():
                self.report_writer.write_evaluation_report(
                    eval_result,
                    context.job_dir / "evaluation" / f"{predictor_name}_metrics.json",
                )

            # Save LLM meeting summaries alongside the report
            self._save_meeting_summaries(context)

            return AgentResult.ok(context, "Reports generated")
        except Exception as e:
            return AgentResult.fail(f"Report step failed: {e}", error=e)

    def _save_meeting_summaries(self, context: WorkflowContext) -> None:
        """Persist LLM agent meeting summaries as a JSON file in the report dir."""
        summaries = {}
        for key, label in self._MEETING_KEYS:
            text = context.extra.get(key)
            if text:
                summaries[key] = {"title": label, "summary": text}

        if not summaries:
            return

        report_dir = context.job_dir / "report"
        report_dir.mkdir(exist_ok=True)
        with open(report_dir / "agent_summaries.json", "w") as f:
            json.dump(summaries, f, indent=2)
