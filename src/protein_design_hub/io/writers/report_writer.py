"""Report generation for prediction and evaluation results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from protein_design_hub.core.types import (
    PredictionResult,
    EvaluationResult,
    ComparisonResult,
)


class ReportWriter:
    """Writer for generating reports in various formats."""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report writer.

        Args:
            output_dir: Default output directory for reports.
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./outputs")

    def write_prediction_report(
        self,
        result: PredictionResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Write a JSON report for prediction results.

        Args:
            result: PredictionResult to report on.
            output_path: Output path for the report.

        Returns:
            Path to the generated report.
        """
        if output_path is None:
            output_path = self.output_dir / result.predictor.value / "scores.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "job_id": result.job_id,
            "predictor": result.predictor.value,
            "timestamp": result.timestamp.isoformat(),
            "runtime_seconds": result.runtime_seconds,
            "success": result.success,
            "error_message": result.error_message,
            "num_structures": len(result.structure_paths),
            "structures": [
                {
                    "path": str(path),
                    "scores": {
                        "plddt": score.plddt,
                        "ptm": score.ptm,
                        "iptm": score.iptm,
                        "confidence": score.confidence,
                    }
                    if score else None
                }
                for path, score in zip(
                    result.structure_paths,
                    result.scores + [None] * (len(result.structure_paths) - len(result.scores))
                )
            ],
            "metadata": result.metadata,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return output_path

    def write_evaluation_report(
        self,
        result: EvaluationResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Write a JSON report for evaluation results.

        Args:
            result: EvaluationResult to report on.
            output_path: Output path for the report.

        Returns:
            Path to the generated report.
        """
        if output_path is None:
            output_path = self.output_dir / "evaluation" / f"{result.structure_path.stem}_metrics.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = result.to_dict()
        report["timestamp"] = datetime.now().isoformat()

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return output_path

    def write_comparison_report(
        self,
        result: ComparisonResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Write a JSON report for comparison results.

        Args:
            result: ComparisonResult to report on.
            output_path: Output path for the report.

        Returns:
            Path to the generated report.
        """
        if output_path is None:
            output_path = self.output_dir / "evaluation" / "comparison_summary.json"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "job_id": result.job_id,
            "timestamp": result.timestamp.isoformat(),
            "best_predictor": result.best_predictor,
            "ranking": result.ranking,
            "predictors": {},
        }

        for predictor_name, pred_result in result.prediction_results.items():
            report["predictors"][predictor_name] = {
                "success": pred_result.success,
                "runtime_seconds": pred_result.runtime_seconds,
                "num_structures": len(pred_result.structure_paths),
                "best_plddt": max(
                    (s.plddt for s in pred_result.scores if s.plddt),
                    default=None
                ),
            }

            if predictor_name in result.evaluation_results:
                eval_result = result.evaluation_results[predictor_name]
                report["predictors"][predictor_name]["evaluation"] = {
                    "lddt": eval_result.lddt,
                    "tm_score": eval_result.tm_score,
                    "qs_score": eval_result.qs_score,
                    "rmsd": eval_result.rmsd,
                }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return output_path

    def write_html_report(
        self,
        result: ComparisonResult,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Generate an HTML report with visualizations.

        Args:
            result: ComparisonResult to report on.
            output_path: Output path for the report.

        Returns:
            Path to the generated HTML report.
        """
        if output_path is None:
            output_path = self.output_dir / "report" / "report.html"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate HTML content
        html = self._generate_html_report(result)

        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _generate_html_report(self, result: ComparisonResult) -> str:
        """Generate HTML content for the report."""
        # Build metrics table rows
        rows = []
        for predictor_name, pred_result in result.prediction_results.items():
            runtime = f"{pred_result.runtime_seconds:.1f}s"
            plddt = "-"
            if pred_result.scores:
                max_plddt = max((s.plddt for s in pred_result.scores if s.plddt), default=None)
                if max_plddt:
                    plddt = f"{max_plddt:.1f}"

            lddt = tm_score = qs_score = rmsd = "-"
            if predictor_name in result.evaluation_results:
                eval_r = result.evaluation_results[predictor_name]
                lddt = f"{eval_r.lddt:.3f}" if eval_r.lddt else "-"
                tm_score = f"{eval_r.tm_score:.3f}" if eval_r.tm_score else "-"
                qs_score = f"{eval_r.qs_score:.3f}" if eval_r.qs_score else "-"
                rmsd = f"{eval_r.rmsd:.2f}" if eval_r.rmsd else "-"

            is_best = predictor_name == result.best_predictor
            row_class = "best-row" if is_best else ""
            badge = " <span class='badge'>BEST</span>" if is_best else ""

            rows.append(f"""
            <tr class="{row_class}">
                <td>{predictor_name}{badge}</td>
                <td>{runtime}</td>
                <td>{plddt}</td>
                <td>{lddt}</td>
                <td>{tm_score}</td>
                <td>{qs_score}</td>
                <td>{rmsd}</td>
            </tr>
            """)

        table_rows = "\n".join(rows)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Protein Design Hub - Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            background: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .best-row {{
            background: #e8f5e9;
        }}
        .badge {{
            background: #4CAF50;
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            margin-left: 8px;
        }}
        .metadata {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
        }}
        .metric-card {{
            display: inline-block;
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px 25px;
            margin: 10px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #4CAF50;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Protein Design Hub - Prediction Report</h1>

        <div class="metadata">
            <strong>Job ID:</strong> {result.job_id}<br>
            <strong>Generated:</strong> {result.timestamp.strftime("%Y-%m-%d %H:%M:%S")}<br>
            <strong>Best Predictor:</strong> {result.best_predictor or "N/A"}
        </div>

        <h2>Summary</h2>
        <div class="metric-cards">
            <div class="metric-card">
                <div class="metric-value">{len(result.prediction_results)}</div>
                <div class="metric-label">Predictors</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{sum(len(r.structure_paths) for r in result.prediction_results.values())}</div>
                <div class="metric-label">Structures</div>
            </div>
        </div>

        <h2>Results Comparison</h2>
        <table>
            <thead>
                <tr>
                    <th>Predictor</th>
                    <th>Runtime</th>
                    <th>pLDDT</th>
                    <th>lDDT</th>
                    <th>TM-score</th>
                    <th>QS-score</th>
                    <th>RMSD</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <div class="metadata">
            <p><em>Report generated by Protein Design Hub v0.1.0</em></p>
        </div>
    </div>
</body>
</html>
"""
        return html

    def write_status_file(
        self,
        predictor_name: str,
        success: bool,
        runtime: float,
        error_message: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Write a status file for a predictor run.

        Args:
            predictor_name: Name of the predictor.
            success: Whether the prediction succeeded.
            runtime: Runtime in seconds.
            error_message: Error message if failed.
            output_dir: Output directory.

        Returns:
            Path to the status file.
        """
        if output_dir is None:
            output_dir = self.output_dir

        output_path = Path(output_dir) / predictor_name / "status.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        status = {
            "predictor": predictor_name,
            "success": success,
            "runtime_seconds": runtime,
            "timestamp": datetime.now().isoformat(),
        }

        if error_message:
            status["error_message"] = error_message

        with open(output_path, "w") as f:
            json.dump(status, f, indent=2)

        return output_path
