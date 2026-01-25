"""Install command for managing prediction tools."""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional

app = typer.Typer(help="Install and manage prediction tools")
console = Console()


@app.command("all")
def install_all(
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Install all prediction tools."""
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold blue]Installing all prediction tools[/bold blue]\n")

    for name in PredictorRegistry.list_available():
        predictor = PredictorRegistry.get(name, settings)
        installer = predictor.installer

        if installer.is_installed() and not force:
            console.print(f"  [green]✓[/green] {name}: already installed")
            continue

        console.print(f"  Installing {name}...")
        try:
            success = installer.install()
            if success:
                console.print(f"  [green]✓[/green] {name}: installed successfully")
            else:
                console.print(f"  [red]✗[/red] {name}: installation failed")
        except Exception as e:
            console.print(f"  [red]✗[/red] {name}: {e}")

    console.print()


@app.command("predictor")
def install_predictor(
    name: str = typer.Argument(..., help="Predictor name (colabfold, chai1, boltz2)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Install a specific prediction tool."""
    from protein_design_hub.predictors.registry import PredictorRegistry, get_predictor
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print(f"\n[bold]Installing {name}[/bold]\n")

    try:
        predictor = get_predictor(name, settings)
    except Exception as e:
        console.print(f"[red]Error: Unknown predictor '{name}'[/red]")
        console.print(f"Available: {', '.join(PredictorRegistry.list_available())}")
        raise typer.Exit(1)

    installer = predictor.installer

    if installer.is_installed() and not force:
        version = installer.get_installed_version()
        console.print(f"  {name} is already installed (v{version})")
        console.print("  Use --force to reinstall")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Installing {name}...", total=None)

        try:
            success = installer.install()
            if success:
                version = installer.get_installed_version()
                console.print(f"\n[green]✓[/green] {name} installed successfully (v{version})")
            else:
                console.print(f"\n[red]✗[/red] Installation failed")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]✗[/red] Installation error: {e}")
            raise typer.Exit(1)

    console.print()


@app.command("update")
def update_tools(
    name: Optional[str] = typer.Argument(None, help="Predictor name (or omit for all)"),
):
    """Update prediction tools to latest versions."""
    from protein_design_hub.predictors.registry import PredictorRegistry, get_predictor
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    if name:
        predictors = [name]
    else:
        predictors = PredictorRegistry.list_available()

    console.print("\n[bold blue]Checking for updates[/bold blue]\n")

    for pred_name in predictors:
        try:
            predictor = get_predictor(pred_name, settings)
            installer = predictor.installer

            if not installer.is_installed():
                console.print(f"  {pred_name}: [dim]not installed[/dim]")
                continue

            current = installer.get_installed_version()
            latest = installer.get_latest_version()

            if current and latest and current != latest:
                console.print(f"  {pred_name}: {current} -> {latest}")
                console.print(f"    Updating...", end=" ")

                if installer.update():
                    console.print("[green]done[/green]")
                else:
                    console.print("[red]failed[/red]")
            else:
                console.print(f"  {pred_name}: [green]up to date[/green] (v{current or 'unknown'})")

        except Exception as e:
            console.print(f"  {pred_name}: [red]error[/red] - {e}")

    console.print()


@app.command("status")
def show_status():
    """Show installation status of all tools."""
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold blue]Installation Status[/bold blue]\n")

    table = Table()
    table.add_column("Tool", style="cyan")
    table.add_column("Installed")
    table.add_column("Version")
    table.add_column("Latest")
    table.add_column("Path")

    for name in PredictorRegistry.list_available():
        try:
            predictor = PredictorRegistry.get(name, settings)
            status = predictor.installer.get_status()

            if status.installed:
                installed = "[green]YES[/green]"
            else:
                installed = "[red]NO[/red]"

            version = status.version or "-"
            latest = status.latest_version or "-"

            if status.needs_update:
                latest = f"[yellow]{latest}[/yellow]"

            path = str(status.path) if status.path else "-"
            if len(path) > 40:
                path = "..." + path[-37:]

            table.add_row(name, installed, version, latest, path)

        except Exception as e:
            table.add_row(name, "[red]ERROR[/red]", str(e), "-", "-")

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
            console.print("  [yellow]NOT AVAILABLE[/yellow] - Predictions will be slow")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")

    console.print()


@app.command("uninstall")
def uninstall_tool(
    name: str = typer.Argument(..., help="Tool to uninstall"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Uninstall a prediction tool."""
    from protein_design_hub.predictors.registry import get_predictor
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    try:
        predictor = get_predictor(name, settings)
    except Exception:
        console.print(f"[red]Unknown tool: {name}[/red]")
        raise typer.Exit(1)

    if not predictor.installer.is_installed():
        console.print(f"{name} is not installed")
        return

    if not confirm:
        confirm = typer.confirm(f"Are you sure you want to uninstall {name}?")
        if not confirm:
            console.print("Cancelled")
            return

    console.print(f"Uninstalling {name}...")
    try:
        predictor.installer.uninstall()
        console.print(f"[green]✓[/green] {name} uninstalled")
    except NotImplementedError:
        console.print(f"[yellow]Uninstall not implemented for {name}[/yellow]")
        console.print("  Manual removal may be required")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.callback()
def callback():
    """Installation management commands."""
    pass
