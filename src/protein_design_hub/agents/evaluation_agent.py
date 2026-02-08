"""Structure evaluation agent."""

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.evaluation.composite import CompositeEvaluator


class EvaluationAgent(BaseAgent):
    """
    Agent responsible for evaluating predicted structures (lDDT, TM-score,
    RMSD, etc.) and populating evaluation_results in the context.
    """

    name = "evaluation"
    description = "Evaluate predicted structures"

    def __init__(self, settings: Settings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings or get_settings()
        self.evaluator = CompositeEvaluator(settings=self.settings)

    def run(self, context: WorkflowContext) -> AgentResult:
        if not context.prediction_results:
            return AgentResult.ok(context, "No prediction results to evaluate")

        try:
            self._report_progress("evaluation", "Evaluating structures", 3, 5)
            evaluation_results = {}

            for predictor_name, pred_result in context.prediction_results.items():
                if not pred_result.success or not pred_result.structure_paths:
                    continue
                best_structure = pred_result.best_structure
                if best_structure:
                    try:
                        eval_result = self.evaluator.evaluate(
                            best_structure,
                            context.reference_path,
                        )
                        evaluation_results[predictor_name] = eval_result
                    except Exception as e:
                        print(f"Evaluation failed for {predictor_name}: {e}")

            context.evaluation_results = evaluation_results
            return AgentResult.ok(context, "Evaluation completed")
        except Exception as e:
            return AgentResult.fail(f"Evaluation step failed: {e}", error=e)
