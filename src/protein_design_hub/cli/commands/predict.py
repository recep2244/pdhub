"""Predict command for structure prediction."""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional, List

app = typer.Typer(help="Run structure predictions")
console = Console()


@app.command("run")
def predict_run(
    input_file: Path = typer.Argument(..., help="Input FASTA file"),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory"
    ),
    predictor: Optional[str] = typer.Option(
        "all",
        "--predictor", "-p",
        help="Predictor to use (colabfold, chai1, boltz2, or 'all')"
    ),
    num_models: int = typer.Option(
        5,
        "--num-models", "-n",
        help="Number of models to generate"
    ),
    num_recycles: int = typer.Option(
        3,
        "--num-recycles", "-r",
        help="Number of recycling iterations"
    ),
    job_id: Optional[str] = typer.Option(
        None,
        "--job-id", "-j",
        help="Custom job identifier"
    ),
    skip_unavailable: bool = typer.Option(
        True,
        "--skip-unavailable",
        help="Skip predictors that aren't installed"
    ),
):
    """Run structure prediction on input sequences."""
    from protein_design_hub.pipeline.runner import SequentialPipelineRunner
    from protein_design_hub.core.types import PredictionInput
    from protein_design_hub.core.config import get_settings
    from protein_design_hub.io.parsers.fasta import FastaParser
    from datetime import datetime

    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    settings = get_settings()

    # Override settings
    settings.predictors.colabfold.num_models = num_models
    settings.predictors.colabfold.num_recycles = num_recycles

    # Parse input
    console.print(f"\n[bold]Parsing input file: {input_file}[/bold]")
    parser = FastaParser()
    try:
        sequences = parser.parse(input_file)
    except Exception as e:
        console.print(f"[red]Error parsing input: {e}[/red]")
        raise typer.Exit(1)

    console.print(f"  Found {len(sequences)} sequence(s), total length: {sum(len(s) for s in sequences)}")

    # Setup output
    if output is None:
        output = settings.output.base_dir

    if job_id is None:
        job_id = f"{input_file.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    job_dir = Path(output) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Create prediction input
    prediction_input = PredictionInput(
        job_id=job_id,
        sequences=sequences,
        output_dir=job_dir,
        num_models=num_models,
        num_recycles=num_recycles,
    )

    # Determine predictors to run
    if predictor.lower() == "all":
        predictors = ["colabfold", "chai1", "boltz2"]
    else:
        predictors = [p.strip() for p in predictor.split(",")]

    console.print(f"\n[bold]Running predictions with: {', '.join(predictors)}[/bold]\n")

    # Run predictions
    runner = SequentialPipelineRunner(settings)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running predictions...", total=len(predictors))

        results = {}
        for pred_name in predictors:
            progress.update(task, description=f"Running {pred_name}...")

            try:
                result = runner.run_single_predictor(
                    pred_name,
                    prediction_input,
                    auto_install=False,
                )
                results[pred_name] = result

                if result.success:
                    console.print(f"  [green]✓[/green] {pred_name}: {len(result.structure_paths)} structures ({result.runtime_seconds:.1f}s)")
                else:
                    console.print(f"  [red]✗[/red] {pred_name}: {result.error_message}")

            except Exception as e:
                if skip_unavailable:
                    console.print(f"  [yellow]⊘[/yellow] {pred_name}: {e}")
                else:
                    console.print(f"  [red]✗[/red] {pred_name}: {e}")

            progress.advance(task)

    # Save summary
    summary_path = runner.save_results(results, job_dir)

    console.print(f"\n[bold green]Predictions complete![/bold green]")
    console.print(f"  Output directory: {job_dir}")
    console.print(f"  Summary: {summary_path}\n")


@app.command("single")
def predict_single(
    input_file: Path = typer.Argument(..., help="Input FASTA file"),
    predictor: str = typer.Argument(..., help="Predictor to use (colabfold, chai1, boltz2)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Run a single predictor."""
    predict_run(
        input_file=input_file,
        output=output,
        predictor=predictor,
        num_models=5,
        num_recycles=3,
        job_id=None,
        skip_unavailable=False,
    )


@app.callback()
def callback():
    """Structure prediction commands."""
    pass
