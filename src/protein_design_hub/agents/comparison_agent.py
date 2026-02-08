"""Comparison and ranking agent."""

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.core.types import ComparisonResult, EvaluationResult, PredictionResult


class ComparisonAgent(BaseAgent):
    """
    Agent responsible for comparing predictor outputs, ranking by lDDT/pLDDT,
    and building the ComparisonResult.
    """

    name = "comparison"
    description = "Compare and rank prediction results"

    def _compute_composite_score(
        self,
        pred_result: PredictionResult,
        eval_result: EvaluationResult | None,
    ) -> float:
        """Compute a composite ranking score (0-1) using multiple metrics.

        Weighting strategy:
          - lDDT or pLDDT: 40% (local accuracy, most reliable)
          - TM-score: 30% (global fold accuracy)
          - Inverse clash penalty: 15% (structural quality)
          - pTM/confidence: 15% (model confidence)

        Falls back to pLDDT/100 if no evaluation data is available.
        """
        components: list[tuple[float, float]] = []  # (weight, value)

        # Primary: lDDT from evaluation or pLDDT from prediction
        if eval_result and eval_result.lddt is not None:
            components.append((0.40, eval_result.lddt))
        elif pred_result.scores:
            plddts = [s.plddt for s in pred_result.scores if s.plddt]
            if plddts:
                components.append((0.40, max(plddts) / 100.0))

        # TM-score (if reference was provided)
        if eval_result and eval_result.tm_score is not None:
            components.append((0.30, eval_result.tm_score))

        # Clash score penalty (lower is better; normalise to 0-1)
        if eval_result and eval_result.clash_score is not None:
            clash_quality = max(0.0, 1.0 - eval_result.clash_score / 100.0)
            components.append((0.15, clash_quality))

        # pTM / confidence bonus
        if pred_result.scores:
            ptms = [s.ptm for s in pred_result.scores if s.ptm]
            if ptms:
                components.append((0.15, max(ptms)))

        if not components:
            return 0.0

        # Normalise weights to sum to 1
        total_weight = sum(w for w, _ in components)
        if total_weight == 0:
            return 0.0
        return sum(w * v for w, v in components) / total_weight

    def run(self, context: WorkflowContext) -> AgentResult:
        if not context.prediction_results:
            return AgentResult.fail("No prediction results to compare")

        try:
            self._report_progress("comparison", "Comparing results", 4, 5)
            ranking = []

            for predictor_name, pred_result in context.prediction_results.items():
                if not pred_result.success:
                    continue
                eval_result = context.evaluation_results.get(predictor_name)
                score = self._compute_composite_score(pred_result, eval_result)
                ranking.append((predictor_name, score))

            ranking.sort(key=lambda x: x[1], reverse=True)
            best_predictor = ranking[0][0] if ranking else None

            context.comparison_result = ComparisonResult(
                job_id=context.job_id,
                prediction_results=context.prediction_results,
                evaluation_results=context.evaluation_results,
                best_predictor=best_predictor,
                ranking=ranking,
            )
            return AgentResult.ok(context, "Comparison completed")
        except Exception as e:
            return AgentResult.fail(f"Comparison step failed: {e}", error=e)
