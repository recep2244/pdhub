"""Input parsing and setup agent."""

import json
from datetime import datetime
from pathlib import Path

from protein_design_hub.agents.base import AgentResult, BaseAgent
from protein_design_hub.agents.context import WorkflowContext
from protein_design_hub.core.config import Settings, get_settings
from protein_design_hub.core.types import PredictionInput, Sequence
from protein_design_hub.io.parsers.fasta import FastaParser


class InputAgent(BaseAgent):
    """
    Agent responsible for parsing input (FASTA), creating PredictionInput,
    and saving input artifacts to the job directory.
    """

    name = "input"
    description = "Parse input sequences and prepare prediction input"

    def __init__(self, settings: Settings | None = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings or get_settings()

    def run(self, context: WorkflowContext) -> AgentResult:
        if not context.input_path or not context.input_path.exists():
            return AgentResult.fail("input_path is missing or does not exist")

        try:
            if not context.job_id:
                context.job_id = self._generate_job_id(context.input_path)
            context.with_job_dir()
            self._report_progress("input", "Parsing input", 1, 5)

            sequences = self._parse_input(context.input_path)
            context.sequences = sequences

            context.prediction_input = PredictionInput(
                job_id=context.job_id,
                sequences=sequences,
                output_dir=context.job_dir,
                num_models=self.settings.predictors.colabfold.num_models,
                num_recycles=self.settings.predictors.colabfold.num_recycles,
            )

            self._save_input(context)
            return AgentResult.ok(context, "Input parsed and saved")
        except Exception as e:
            return AgentResult.fail(f"Input step failed: {e}", error=e)

    def _parse_input(self, input_path: Path) -> list[Sequence]:
        parser = FastaParser()
        try:
            return parser.parse(input_path)
        except Exception:
            pass
        try:
            return parser.parse_multimer(input_path, chain_separator=":")
        except Exception:
            pass
        raise ValueError(f"Could not parse input file: {input_path}")

    def _save_input(self, context: WorkflowContext) -> None:
        input_dir = context.job_dir / "input"
        input_dir.mkdir(exist_ok=True)
        parser = FastaParser()
        parser.write(context.prediction_input.sequences, input_dir / "sequences.fasta")
        metadata = {
            "job_id": context.prediction_input.job_id,
            "num_sequences": len(context.prediction_input.sequences),
            "total_length": context.prediction_input.total_length,
            "is_multimer": context.prediction_input.is_multimer,
            "num_models": context.prediction_input.num_models,
            "num_recycles": context.prediction_input.num_recycles,
            "timestamp": datetime.now().isoformat(),
        }
        with open(context.job_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_job_id(self, input_path: Path) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{input_path.stem}_{timestamp}"
