"""Evaluate command for structure evaluation."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional, List

app = typer.Typer(help="Evaluate predicted structures")
console = Console()


@app.command("run")
def evaluate_run(
    model: Path = typer.Argument(..., help="Model structure file (PDB or CIF)"),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference", "-r",
        help="Reference structure for comparison"
    ),
    metrics: Optional[str] = typer.Option(
        "lddt,tm_score,rmsd",
        "--metrics", "-m",
        help="Comma-separated list of metrics to compute"
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file for results (JSON)"
    ),
):
    """Evaluate a structure using various quality metrics."""
    from protein_design_hub.evaluation.composite import CompositeEvaluator
    from protein_design_hub.core.config import get_settings
    import json

    if not model.exists():
        console.print(f"[red]Error: Model file not found: {model}[/red]")
        raise typer.Exit(1)

    if reference and not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    settings = get_settings()

    # Parse metrics
    metric_list = [m.strip().lower() for m in metrics.split(",")]

    console.print(f"\n[bold]Evaluating: {model.name}[/bold]")
    if reference:
        console.print(f"  Reference: {reference.name}")
    console.print(f"  Metrics: {', '.join(metric_list)}\n")

    # Run evaluation
    evaluator = CompositeEvaluator(metrics=metric_list, settings=settings)

    # Check available metrics
    available = evaluator.get_available_metrics()
    unavailable = [m for m, v in available.items() if not v]
    if unavailable:
        requirements = evaluator.get_metric_requirements()
        for m in unavailable:
            console.print(f"[yellow]Warning: {m} not available - {requirements.get(m, 'Unknown')}[/yellow]")

    try:
        result = evaluator.evaluate(model, reference)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    if result.lddt is not None:
        table.add_row("lDDT", f"{result.lddt:.4f}")

    if result.tm_score is not None:
        table.add_row("TM-score", f"{result.tm_score:.4f}")

    if result.qs_score is not None:
        table.add_row("QS-score", f"{result.qs_score:.4f}")

    if result.rmsd is not None:
        table.add_row("RMSD", f"{result.rmsd:.4f} Å")

    if result.gdt_ts is not None:
        table.add_row("GDT-TS", f"{result.gdt_ts:.4f}")

    if result.gdt_ha is not None:
        table.add_row("GDT-HA", f"{result.gdt_ha:.4f}")

    console.print(table)

    # Save results if output specified
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        result_dict = result.to_dict()
        with open(output, "w") as f:
            json.dump(result_dict, f, indent=2)

        console.print(f"\nResults saved to: {output}")

    console.print()


@app.command("batch")
def evaluate_batch(
    input_dir: Path = typer.Argument(..., help="Directory containing structure files"),
    reference: Optional[Path] = typer.Option(None, "--reference", "-r", help="Reference structure"),
    pattern: str = typer.Option("*.pdb,*.cif", "--pattern", "-p", help="File patterns to match"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file"),
):
    """Evaluate multiple structures in a directory."""
    from protein_design_hub.evaluation.composite import CompositeEvaluator
    from protein_design_hub.core.config import get_settings
    import csv

    if not input_dir.exists():
        console.print(f"[red]Error: Directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    evaluator = CompositeEvaluator(settings=settings)

    # Find files
    patterns = [p.strip() for p in pattern.split(",")]
    files = []
    for pat in patterns:
        files.extend(input_dir.glob(pat))

    if not files:
        console.print(f"[yellow]No files found matching: {pattern}[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Evaluating {len(files)} structures[/bold]\n")

    results = []
    for f in files:
        try:
            result = evaluator.evaluate(f, reference)
            results.append({
                "file": f.name,
                "lddt": result.lddt,
                "tm_score": result.tm_score,
                "rmsd": result.rmsd,
                "qs_score": result.qs_score,
            })
            console.print(f"  [green]✓[/green] {f.name}")
        except Exception as e:
            console.print(f"  [red]✗[/red] {f.name}: {e}")
            results.append({
                "file": f.name,
                "error": str(e),
            })

    # Display summary
    table = Table(title="Evaluation Summary")
    table.add_column("File", style="cyan")
    table.add_column("lDDT")
    table.add_column("TM-score")
    table.add_column("RMSD")

    for r in results:
        if "error" in r:
            table.add_row(r["file"], "[red]ERROR[/red]", "", "")
        else:
            table.add_row(
                r["file"],
                f"{r['lddt']:.3f}" if r.get("lddt") else "-",
                f"{r['tm_score']:.3f}" if r.get("tm_score") else "-",
                f"{r['rmsd']:.2f}" if r.get("rmsd") else "-",
            )

    console.print(table)

    # Save to CSV if output specified
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "lddt", "tm_score", "rmsd", "qs_score", "error"])
            writer.writeheader()
            writer.writerows(results)

        console.print(f"\nResults saved to: {output}")

    console.print()


@app.command("metrics")
def list_metrics():
    """List available evaluation metrics."""
    from protein_design_hub.evaluation.composite import CompositeEvaluator

    console.print("\n[bold]Available Metrics[/bold]\n")

    metrics_info = CompositeEvaluator.list_all_metrics()

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("Description")
    table.add_column("Reference Required")
    table.add_column("Available")

    for metric in metrics_info:
        available = "[green]YES[/green]" if metric["available"] else f"[red]NO[/red]"
        ref_required = "Yes" if metric["requires_reference"] else "No"

        table.add_row(
            metric["name"],
            metric["description"],
            ref_required,
            available,
        )

    console.print(table)

    # Show requirements for unavailable metrics
    unavailable = [m for m in metrics_info if not m["available"]]
    if unavailable:
        console.print("\n[bold]Installation Requirements:[/bold]")
        for m in unavailable:
            if m["requirements"]:
                console.print(f"  {m['name']}: {m['requirements']}")

    console.print()


@app.callback()
def callback():
    """Structure evaluation commands."""
    pass
