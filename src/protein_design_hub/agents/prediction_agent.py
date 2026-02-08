"""Structure prediction agent."""

from pathlib import Path

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.pipeline.runner import SequentialPipelineRunner


class PredictionAgent(BaseAgent):
    """
    Agent responsible for running structure predictors (ColabFold, Chai-1,
    Boltz-2, etc.) and populating prediction_results in the context.
    """

    name = "prediction"
    description = "Run structure predictors"

    def __init__(
        self,
        settings: Settings | None = None,
        skip_unavailable: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.settings = settings or get_settings()
        self.skip_unavailable = skip_unavailable
        self.runner = SequentialPipelineRunner(self.settings, progress_callback=kwargs.get("progress_callback"))

    def run(self, context: WorkflowContext) -> AgentResult:
        if context.prediction_input is None:
            return AgentResult.fail("prediction_input is missing; run InputAgent first")

        try:
            self._report_progress("prediction", "Running predictors", 2, 5)
            results = self.runner.run_all_predictors(
                context.prediction_input,
                predictors=context.predictors,
                skip_unavailable=self.skip_unavailable,
            )
            context.prediction_results = results
            if context.job_dir:
                self.runner.save_results(results, context.job_dir)
            return AgentResult.ok(context, "Predictions completed")
        except Exception as e:
            return AgentResult.fail(f"Prediction step failed: {e}", error=e)
