"""Full prediction -> evaluation workflow."""

from pathlib import Path
from typing import Dict, List, Optional, Callable
from datetime import datetime
import json

from protein_design_hub.core.types import (
    PredictionInput,
    PredictionResult,
    EvaluationResult,
    ComparisonResult,
    Sequence,
)
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.pipeline.runner import SequentialPipelineRunner
from protein_design_hub.evaluation.composite import CompositeEvaluator
from protein_design_hub.io.parsers.fasta import FastaParser
from protein_design_hub.io.writers.report_writer import ReportWriter


class PredictionWorkflow:
    """
    Complete workflow for protein structure prediction and evaluation.

    Orchestrates the full pipeline:
    1. Parse input sequences
    2. Run predictions with multiple tools
    3. Evaluate predicted structures
    4. Compare and rank results
    5. Generate reports
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        progress_callback: Optional[Callable[[str, str, int, int], None]] = None,
    ):
        """
        Initialize the workflow.

        Args:
            settings: Configuration settings.
            progress_callback: Optional callback(stage, item, current, total).
        """
        self.settings = settings or get_settings()
        self.progress_callback = progress_callback
        self.runner = SequentialPipelineRunner(settings)
        self.evaluator = CompositeEvaluator(settings=settings)
        self.report_writer = ReportWriter()

    def run(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        reference_path: Optional[Path] = None,
        predictors: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> ComparisonResult:
        """
        Run the complete prediction and evaluation workflow.

        Args:
            input_path: Path to input FASTA file.
            output_dir: Output directory for results.
            reference_path: Optional reference structure for evaluation.
            predictors: List of predictors to use. Uses all enabled if not specified.
            job_id: Optional job identifier.

        Returns:
            ComparisonResult with all predictions and evaluations.
        """
        # Setup
        input_path = Path(input_path)
        if output_dir is None:
            output_dir = self.settings.output.base_dir
        output_dir = Path(output_dir)

        if job_id is None:
            job_id = self._generate_job_id(input_path)

        job_dir = output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        self._report_progress("setup", "Parsing input", 1, 5)

        # Parse input sequences
        sequences = self._parse_input(input_path)

        # Create prediction input
        prediction_input = PredictionInput(
            job_id=job_id,
            sequences=sequences,
            output_dir=job_dir,
            num_models=self.settings.predictors.colabfold.num_models,
            num_recycles=self.settings.predictors.colabfold.num_recycles,
        )

        # Save input sequences
        self._save_input(prediction_input, job_dir)

        # Run predictions
        self._report_progress("prediction", "Running predictors", 2, 5)
        prediction_results = self.runner.run_all_predictors(
            prediction_input,
            predictors=predictors,
        )
        
        # Save prediction summary at root for UI detection
        self.runner.save_results(prediction_results, job_dir)

        # Evaluate structures
        self._report_progress("evaluation", "Evaluating structures", 3, 5)
        evaluation_results = self._evaluate_all(
            prediction_results,
            reference_path,
        )

        # Compare and rank
        self._report_progress("comparison", "Comparing results", 4, 5)
        comparison_result = self._compare_results(
            job_id,
            prediction_results,
            evaluation_results,
        )

        # Generate reports
        self._report_progress("reporting", "Generating reports", 5, 5)
        self._generate_reports(comparison_result, job_dir)

        return comparison_result

    def run_prediction_only(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        predictors: Optional[List[str]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, PredictionResult]:
        """
        Run predictions without evaluation.

        Args:
            input_path: Path to input FASTA file.
            output_dir: Output directory.
            predictors: List of predictors to use.
            job_id: Optional job identifier.

        Returns:
            Dictionary of prediction results.
        """
        input_path = Path(input_path)
        if output_dir is None:
            output_dir = self.settings.output.base_dir
        output_dir = Path(output_dir)

        if job_id is None:
            job_id = self._generate_job_id(input_path)

        job_dir = output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        sequences = self._parse_input(input_path)

        prediction_input = PredictionInput(
            job_id=job_id,
            sequences=sequences,
            output_dir=job_dir,
            num_models=self.settings.predictors.colabfold.num_models,
            num_recycles=self.settings.predictors.colabfold.num_recycles,
        )

        self._save_input(prediction_input, job_dir)

        results = self.runner.run_all_predictors(prediction_input, predictors=predictors)
        self.runner.save_results(results, job_dir)
        return results

    def run_evaluation_only(
        self,
        structure_paths: List[Path],
        reference_path: Optional[Path] = None,
    ) -> List[EvaluationResult]:
        """
        Run evaluation on existing structures.

        Args:
            structure_paths: List of structure file paths.
            reference_path: Optional reference structure.

        Returns:
            List of evaluation results.
        """
        return self.evaluator.evaluate_batch(structure_paths, reference_path)

    def _parse_input(self, input_path: Path) -> List[Sequence]:
        """Parse input file to extract sequences."""
        parser = FastaParser()

        # Try as FASTA first
        try:
            return parser.parse(input_path)
        except Exception:
            pass

        # Try as multimer FASTA with chain separator
        try:
            return parser.parse_multimer(input_path, chain_separator=":")
        except Exception:
            pass

        raise ValueError(f"Could not parse input file: {input_path}")

    def _save_input(self, input_data: PredictionInput, job_dir: Path) -> None:
        """Save input data for reference."""
        input_dir = job_dir / "input"
        input_dir.mkdir(exist_ok=True)

        # Save sequences
        parser = FastaParser()
        parser.write(input_data.sequences, input_dir / "sequences.fasta")

        # Save metadata
        metadata = {
            "job_id": input_data.job_id,
            "num_sequences": len(input_data.sequences),
            "total_length": input_data.total_length,
            "is_multimer": input_data.is_multimer,
            "num_models": input_data.num_models,
            "num_recycles": input_data.num_recycles,
            "timestamp": datetime.now().isoformat(),
        }
        with open(job_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _evaluate_all(
        self,
        prediction_results: Dict[str, PredictionResult],
        reference_path: Optional[Path],
    ) -> Dict[str, EvaluationResult]:
        """Evaluate all predicted structures."""
        evaluation_results = {}

        for predictor_name, pred_result in prediction_results.items():
            if not pred_result.success or not pred_result.structure_paths:
                continue

            # Evaluate the best structure from each predictor
            best_structure = pred_result.best_structure
            if best_structure:
                try:
                    eval_result = self.evaluator.evaluate(best_structure, reference_path)
                    evaluation_results[predictor_name] = eval_result
                except Exception as e:
                    print(f"Evaluation failed for {predictor_name}: {e}")

        return evaluation_results

    def _compare_results(
        self,
        job_id: str,
        prediction_results: Dict[str, PredictionResult],
        evaluation_results: Dict[str, EvaluationResult],
    ) -> ComparisonResult:
        """Compare results and determine ranking."""
        # Rank by pLDDT if no reference, otherwise by lDDT
        ranking = []

        for predictor_name, pred_result in prediction_results.items():
            if not pred_result.success:
                continue

            score = 0.0

            # Use evaluation metrics if available
            if predictor_name in evaluation_results:
                eval_result = evaluation_results[predictor_name]
                if eval_result.lddt is not None:
                    score = eval_result.lddt
                elif eval_result.tm_score is not None:
                    score = eval_result.tm_score

            # Fall back to pLDDT
            if score == 0.0 and pred_result.scores:
                max_plddt = max(
                    (s.plddt for s in pred_result.scores if s.plddt),
                    default=None
                )
                if max_plddt:
                    score = max_plddt / 100.0  # Normalize to 0-1

            ranking.append((predictor_name, score))

        # Sort by score descending
        ranking.sort(key=lambda x: x[1], reverse=True)

        best_predictor = ranking[0][0] if ranking else None

        return ComparisonResult(
            job_id=job_id,
            prediction_results=prediction_results,
            evaluation_results=evaluation_results,
            best_predictor=best_predictor,
            ranking=ranking,
        )

    def _generate_reports(
        self,
        comparison_result: ComparisonResult,
        job_dir: Path,
    ) -> None:
        """Generate output reports."""
        self.report_writer.output_dir = job_dir

        # Write comparison summary
        self.report_writer.write_comparison_report(
            comparison_result,
            job_dir / "evaluation" / "comparison_summary.json",
        )

        # Write HTML report
        if self.settings.output.generate_report:
            self.report_writer.write_html_report(
                comparison_result,
                job_dir / "report" / "report.html",
            )

        # Write individual predictor reports
        for predictor_name, pred_result in comparison_result.prediction_results.items():
            if pred_result.success:
                self.report_writer.write_prediction_report(
                    pred_result,
                    job_dir / predictor_name / "scores.json",
                )

                self.report_writer.write_status_file(
                    predictor_name,
                    pred_result.success,
                    pred_result.runtime_seconds,
                    pred_result.error_message,
                    job_dir,
                )

        # Write evaluation reports
        for predictor_name, eval_result in comparison_result.evaluation_results.items():
            self.report_writer.write_evaluation_report(
                eval_result,
                job_dir / "evaluation" / f"{predictor_name}_metrics.json",
            )

    def _generate_job_id(self, input_path: Path) -> str:
        """Generate a job ID from input path and timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = input_path.stem
        return f"{name}_{timestamp}"

    def _report_progress(self, stage: str, item: str, current: int, total: int) -> None:
        """Report progress via callback."""
        if self.progress_callback:
            self.progress_callback(stage, item, current, total)
