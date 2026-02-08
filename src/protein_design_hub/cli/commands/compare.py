"""Compare command for running full comparison pipeline."""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional

app = typer.Typer(help="Compare predictions from multiple tools")
console = Console()


@app.command("run")
def compare_run(
    input_file: Path = typer.Argument(..., help="Input FASTA file"),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output directory"
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference", "-r",
        help="Reference structure for evaluation"
    ),
    predictors: Optional[str] = typer.Option(
        None,
        "--predictors", "-p",
        help="Comma-separated list of predictors (default: all enabled)"
    ),
    job_id: Optional[str] = typer.Option(
        None,
        "--job-id", "-j",
        help="Custom job identifier"
    ),
    use_agents: bool = typer.Option(
        False,
        "--agents", "-a",
        help="Use multi-agent step pipeline (one agent per step)"
    ),
    use_llm_agents: bool = typer.Option(
        False,
        "--llm-agents",
        help="Use LLM-guided multi-agent pipeline (Virtual-Lab style: team meetings + step agents)"
    ),
):
    """Run full comparison pipeline: predict with all tools, then evaluate."""
    from protein_design_hub.core.config import get_settings

    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    if reference and not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    settings = get_settings()

    # Parse predictors
    predictor_list = None
    if predictors:
        predictor_list = [p.strip() for p in predictors.split(",")]

    # Determine mode
    if use_llm_agents:
        mode_label = "LLM-guided multi-agent pipeline"
        agent_mode = "llm"
    elif use_agents:
        mode_label = "multi-agent step pipeline"
        agent_mode = "step"
    else:
        mode_label = None
        agent_mode = None

    console.print(f"\n[bold blue]Protein Design Hub - Comparison Pipeline[/bold blue]\n")
    if mode_label:
        console.print(f"  [dim]Mode: {mode_label}[/dim]")
    console.print(f"  Input: {input_file}")
    if reference:
        console.print(f"  Reference: {reference}")
    console.print()

    # Progress tracking
    def progress_callback(stage: str, item: str, current: int, total: int):
        pass  # Could be used for more detailed progress

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running comparison pipeline...", total=None)

        try:
            if agent_mode is not None:
                from protein_design_hub.agents import AgentOrchestrator

                orchestrator = AgentOrchestrator(
                    mode=agent_mode,
                    progress_callback=progress_callback,
                )
                agent_result = orchestrator.run(
                    input_path=input_file,
                    output_dir=output,
                    reference_path=reference,
                    predictors=predictor_list,
                    job_id=job_id,
                )
                if not agent_result.success:
                    console.print(f"[red]Error: {agent_result.message}[/red]")
                    if agent_result.error:
                        console.print(f"[red]{agent_result.error}[/red]")
                    raise typer.Exit(1)
                result = agent_result.context.comparison_result
            else:
                from protein_design_hub.pipeline.workflow import PredictionWorkflow

                workflow = PredictionWorkflow(settings, progress_callback)
                result = workflow.run(
                    input_path=input_file,
                    output_dir=output,
                    reference_path=reference,
                    predictors=predictor_list,
                    job_id=job_id,
                )
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Display results
    console.print("\n[bold green]Comparison Complete![/bold green]\n")

    # Prediction summary
    pred_table = Table(title="Prediction Results")
    pred_table.add_column("Predictor", style="cyan")
    pred_table.add_column("Status")
    pred_table.add_column("Structures")
    pred_table.add_column("Runtime")
    pred_table.add_column("Best pLDDT")

    for predictor_name, pred_result in result.prediction_results.items():
        if pred_result.success:
            status = "[green]SUCCESS[/green]"
            num_structures = str(len(pred_result.structure_paths))
            runtime = f"{pred_result.runtime_seconds:.1f}s"
            best_plddt = "-"
            if pred_result.scores:
                max_plddt = max((s.plddt for s in pred_result.scores if s.plddt), default=None)
                if max_plddt:
                    best_plddt = f"{max_plddt:.1f}"
        else:
            status = "[red]FAILED[/red]"
            num_structures = "0"
            runtime = "-"
            best_plddt = "-"

        pred_table.add_row(predictor_name, status, num_structures, runtime, best_plddt)

    console.print(pred_table)

    # Evaluation summary (if reference provided)
    if result.evaluation_results:
        console.print()
        eval_table = Table(title="Evaluation Results")
        eval_table.add_column("Predictor", style="cyan")
        eval_table.add_column("lDDT")
        eval_table.add_column("TM-score")
        eval_table.add_column("RMSD")
        eval_table.add_column("QS-score")

        for predictor_name, eval_result in result.evaluation_results.items():
            eval_table.add_row(
                predictor_name,
                f"{eval_result.lddt:.3f}" if eval_result.lddt else "-",
                f"{eval_result.tm_score:.3f}" if eval_result.tm_score else "-",
                f"{eval_result.rmsd:.2f} Å" if eval_result.rmsd else "-",
                f"{eval_result.qs_score:.3f}" if eval_result.qs_score else "-",
            )

        console.print(eval_table)

    # Ranking
    if result.ranking:
        console.print()
        rank_table = Table(title="Ranking")
        rank_table.add_column("Rank")
        rank_table.add_column("Predictor", style="cyan")
        rank_table.add_column("Score")

        for i, (predictor_name, score) in enumerate(result.ranking, 1):
            rank = f"[bold green]{i}[/bold green]" if i == 1 else str(i)
            rank_table.add_row(rank, predictor_name, f"{score:.3f}")

        console.print(rank_table)

    # Output location
    if output:
        output_dir = Path(output) / result.job_id
    else:
        output_dir = settings.output.base_dir / result.job_id

    console.print(f"\n[bold]Results saved to:[/bold] {output_dir}")
    console.print(f"  Report: {output_dir / 'report' / 'report.html'}\n")


@app.command("results")
def show_results(
    job_dir: Path = typer.Argument(..., help="Job directory containing results"),
):
    """Display results from a previous comparison run."""
    import json

    if not job_dir.exists():
        console.print(f"[red]Error: Directory not found: {job_dir}[/red]")
        raise typer.Exit(1)

    summary_path = job_dir / "evaluation" / "comparison_summary.json"
    if not summary_path.exists():
        console.print(f"[red]Error: Summary file not found: {summary_path}[/red]")
        raise typer.Exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    console.print(f"\n[bold blue]Comparison Results: {summary['job_id']}[/bold blue]")
    console.print(f"  Generated: {summary['timestamp']}\n")

    if summary.get("best_predictor"):
        console.print(f"  [bold green]Best Predictor: {summary['best_predictor']}[/bold green]\n")

    # Display predictor results
    table = Table(title="Results")
    table.add_column("Predictor", style="cyan")
    table.add_column("Status")
    table.add_column("Runtime")
    table.add_column("pLDDT")
    table.add_column("lDDT")

    for name, data in summary.get("predictors", {}).items():
        status = "[green]OK[/green]" if data.get("success") else "[red]FAIL[/red]"
        runtime = f"{data.get('runtime_seconds', 0):.1f}s"
        plddt = f"{data.get('best_plddt', 0):.1f}" if data.get("best_plddt") else "-"

        lddt = "-"
        if "evaluation" in data:
            lddt = f"{data['evaluation'].get('lddt', 0):.3f}" if data['evaluation'].get('lddt') else "-"

        table.add_row(name, status, runtime, plddt, lddt)

    console.print(table)
    console.print()


@app.callback()
def callback():
    """Comparison commands."""
    pass
