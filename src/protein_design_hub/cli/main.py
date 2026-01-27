"""Main CLI entry point for Protein Design Hub."""

import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional

from protein_design_hub import __version__
from protein_design_hub.cli.commands import (
    predict,
    evaluate,
    compare,
    install,
    design,
    backbone,
    energy,
)

app = typer.Typer(
    name="pdhub",
    help="Protein Design Hub - Unified protein structure prediction and evaluation",
    add_completion=False,
)

console = Console()

# Register sub-commands
app.add_typer(predict.app, name="predict", help="Run structure predictions")
app.add_typer(evaluate.app, name="evaluate", help="Evaluate predicted structures")
app.add_typer(compare.app, name="compare", help="Compare predictions from multiple tools")
app.add_typer(install.app, name="install", help="Install and manage prediction tools")
app.add_typer(design.app, name="design", help="Sequence design and analysis")
app.add_typer(backbone.app, name="backbone", help="Backbone generation tools")
app.add_typer(energy.app, name="energy", help="Energy and scoring tools")


@app.command()
def status():
    """Show installation status of all prediction tools."""
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold blue]Protein Design Hub Status[/bold blue]\n")

    # Predictor status table
    table = Table(title="Predictors")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Version")
    table.add_column("Latest")
    table.add_column("GPU")

    for name in PredictorRegistry.list_available():
        try:
            predictor = PredictorRegistry.get(name, settings)
            status_info = predictor.get_status()

            if status_info["installed"]:
                status_str = "[green]INSTALLED[/green]"
            else:
                status_str = "[red]NOT INSTALLED[/red]"

            version = status_info.get("version") or "-"
            latest = status_info.get("latest_version") or "-"

            if status_info.get("needs_update"):
                latest = f"[yellow]{latest}[/yellow]"

            gpu = "[green]YES[/green]" if status_info.get("gpu_available") else "[dim]NO[/dim]"

            table.add_row(name, status_str, version, latest, gpu)
        except Exception as e:
            table.add_row(name, f"[red]ERROR[/red]", str(e), "-", "-")

    console.print(table)

    # GPU status
    console.print("\n[bold]GPU Status:[/bold]")
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"  [green]AVAILABLE[/green]: {device_name} ({memory:.1f} GB)")
        else:
            console.print("  [yellow]NOT AVAILABLE[/yellow]")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")

    console.print()


@app.command()
def verify(
    predictor: Optional[str] = typer.Option(
        None, "--predictor", "-p", help="Specific predictor to verify (or 'all')"
    ),
):
    """Run verification tests for installed predictors."""
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    predictors_to_verify = []
    if predictor is None or predictor.lower() == "all":
        predictors_to_verify = PredictorRegistry.list_available()
    else:
        predictors_to_verify = [predictor]

    console.print("\n[bold blue]Verifying Predictors[/bold blue]\n")

    for name in predictors_to_verify:
        try:
            pred = PredictorRegistry.get(name, settings)

            if not pred.installer.is_installed():
                console.print(f"  {name}: [yellow]SKIPPED[/yellow] (not installed)")
                continue

            console.print(f"  Verifying {name}...", end=" ")

            success, message = pred.verify_installation()

            if success:
                console.print(f"[green]OK[/green] - {message}")
            else:
                console.print(f"[red]FAILED[/red] - {message}")

        except Exception as e:
            console.print(f"  {name}: [red]ERROR[/red] - {e}")

    console.print()


@app.command()
def web(
    host: str = typer.Option("localhost", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8501, "--port", "-p", help="Port to run on"),
):
    """Launch the Streamlit web interface."""
    import subprocess
    import sys

    web_app_path = Path(__file__).parent.parent / "web" / "app.py"

    console.print(f"\n[bold blue]Starting Protein Design Hub Web UI[/bold blue]")
    console.print(f"  URL: http://{host}:{port}\n")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(web_app_path),
                "--server.address",
                host,
                "--server.port",
                str(port),
            ],
            check=True,
        )
    except FileNotFoundError:
        console.print("[red]Streamlit not installed. Install with: pip install streamlit[/red]")
    except KeyboardInterrupt:
        console.print("\n[dim]Server stopped[/dim]")


@app.command()
def version():
    """Show version information."""
    console.print(f"Protein Design Hub v{__version__}")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """Protein Design Hub - Unified protein structure prediction and evaluation."""
    pass


if __name__ == "__main__":
    app()
