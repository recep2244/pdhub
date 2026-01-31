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
    skip_eval: bool = typer.Option(
        False,
        "--skip-eval",
        help="Skip installing evaluation toolchains (e.g. OpenStructure via micromamba)",
    ),
):
    """Install all prediction and design tools."""
    from protein_design_hub.predictors.registry import PredictorRegistry
    from protein_design_hub.design.registry import DesignerRegistry
    from protein_design_hub.design.generators.registry import GeneratorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold blue]Installing all prediction + design tools[/bold blue]\n")

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

    console.print("\n[bold blue]Installing design tools[/bold blue]\n")

    for name in DesignerRegistry.list_available():
        designer = DesignerRegistry.get(name, settings)
        installer = designer.installer

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

    console.print("\n[bold blue]Installing backbone generators[/bold blue]\n")
    for name in GeneratorRegistry.list_available():
        generator = GeneratorRegistry.get(name, settings)
        installer = generator.installer

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

    if not skip_eval:
        console.print("\n[bold blue]Installing evaluation toolchains[/bold blue]\n")
        try:
            from protein_design_hub.evaluation.ost_runner import get_ost_runner

            runner = get_ost_runner()
            if runner.is_available():
                console.print(
                    "  [green]✓[/green] OpenStructure: already available (micromamba env 'ost')"
                )
            elif runner.micromamba_path:
                console.print("  Installing OpenStructure (micromamba env 'ost')...")
                import subprocess

                result = subprocess.run(
                    [
                        str(runner.micromamba_path),
                        "create",
                        "-n",
                        "ost",
                        "-c",
                        "conda-forge",
                        "-c",
                        "bioconda",
                        "openstructure",
                        "-y",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if result.returncode == 0:
                    console.print("  [green]✓[/green] OpenStructure: installed")
                else:
                    console.print("  [red]✗[/red] OpenStructure: install failed")
                    console.print(result.stderr.strip() or result.stdout.strip())
            else:
                console.print(
                    "  [yellow]⚠[/yellow] micromamba not found; skipping OpenStructure install"
                )
        except Exception as e:
            console.print(f"  [yellow]⚠[/yellow] Evaluation install skipped: {e}")

    console.print()


@app.command("predictor")
def install_predictor(
    name: str = typer.Argument(
        ..., help="Predictor name (colabfold, chai1, boltz2, esmfold, esmfold_api, esm3)"
    ),
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


@app.command("designer")
def install_designer(
    name: str = typer.Argument(..., help="Designer name (proteinmpnn)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Install a specific design tool."""
    from protein_design_hub.design.registry import DesignerRegistry, get_designer
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    console.print(f"\n[bold]Installing designer: {name}[/bold]\n")

    try:
        designer = get_designer(name, settings)
    except Exception:
        console.print(f"[red]Error: Unknown designer '{name}'[/red]")
        console.print(f"Available: {', '.join(DesignerRegistry.list_available())}")
        raise typer.Exit(1)

    installer = designer.installer
    if installer.is_installed() and not force:
        version = installer.get_installed_version()
        console.print(f"  {name} is already installed (v{version or 'unknown'})")
        console.print("  Use --force to reinstall")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Installing {name}...", total=None)
        try:
            success = installer.install()
            if success:
                version = installer.get_installed_version()
                console.print(
                    f"\n[green]✓[/green] {name} installed successfully (v{version or 'unknown'})"
                )
            else:
                console.print("\n[red]✗[/red] Installation failed")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]✗[/red] Installation error: {e}")
            raise typer.Exit(1)

    console.print()


@app.command("generator")
def install_generator(
    name: str = typer.Argument(..., help="Backbone generator name (rfdiffusion)"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reinstall"),
):
    """Install a specific backbone generator tool."""
    from protein_design_hub.design.generators.registry import GeneratorRegistry, get_generator
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    console.print(f"\n[bold]Installing generator: {name}[/bold]\n")

    try:
        generator = get_generator(name, settings)
    except Exception:
        console.print(f"[red]Error: Unknown generator '{name}'[/red]")
        console.print(f"Available: {', '.join(GeneratorRegistry.list_available())}")
        raise typer.Exit(1)

    installer = generator.installer
    if installer.is_installed() and not force:
        version = installer.get_installed_version()
        console.print(f"  {name} is already installed (v{version or 'unknown'})")
        console.print("  Use --force to reinstall")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Installing {name}...", total=None)
        try:
            success = installer.install()
            if success:
                version = installer.get_installed_version()
                console.print(
                    f"\n[green]✓[/green] {name} installed successfully (v{version or 'unknown'})"
                )
            else:
                console.print("\n[red]✗[/red] Installation failed")
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
    from protein_design_hub.design.registry import DesignerRegistry
    from protein_design_hub.design.generators.registry import GeneratorRegistry
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold blue]Installation Status[/bold blue]\n")

    table = Table(title="Predictors")
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

    # Design tools status
    console.print("\n[bold blue]Design Tools[/bold blue]\n")
    dtable = Table(title="Designers")
    dtable.add_column("Tool", style="cyan")
    dtable.add_column("Installed")
    dtable.add_column("Version")

    for name in DesignerRegistry.list_available():
        try:
            designer = DesignerRegistry.get(name, settings)
            status = designer.installer.get_status()
            installed = "[green]YES[/green]" if status.installed else "[red]NO[/red]"
            version = status.version or "-"
            dtable.add_row(name, installed, version)
        except Exception as e:
            dtable.add_row(name, "[red]ERROR[/red]", str(e))
    console.print(dtable)

    gtable = Table(title="Backbone Generators")
    gtable.add_column("Tool", style="cyan")
    gtable.add_column("Installed")
    gtable.add_column("Version")

    for name in GeneratorRegistry.list_available():
        try:
            gen = GeneratorRegistry.get(name, settings)
            status = gen.installer.get_status()
            installed = "[green]YES[/green]" if status.installed else "[red]NO[/red]"
            version = status.version or "-"
            gtable.add_row(name, installed, version)
        except Exception as e:
            gtable.add_row(name, "[red]ERROR[/red]", str(e))
    console.print(gtable)

    # Evaluation tools status
    console.print("\n[bold]Evaluation Tools:[/bold]")

    eval_table = Table()
    eval_table.add_column("Tool", style="cyan")
    eval_table.add_column("Installed")
    eval_table.add_column("Version")
    eval_table.add_column("Notes")

    # OpenStructure via micromamba runner
    try:
        from protein_design_hub.evaluation.ost_runner import get_ost_runner

        runner = get_ost_runner()
        if runner.is_available():
            version = runner.get_version() or "unknown"
            eval_table.add_row(
                "OpenStructure", "[green]YES[/green]", version, "via micromamba (ost env)"
            )
        else:
            eval_table.add_row(
                "OpenStructure",
                "[red]NO[/red]",
                "-",
                "micromamba create -n ost -c conda-forge -c bioconda openstructure",
            )
    except Exception as e:
        eval_table.add_row("OpenStructure", "[red]ERROR[/red]", str(e), "-")

    # TMalign
    try:
        import shutil

        tmalign_path = shutil.which("TMalign")
        if tmalign_path:
            eval_table.add_row("TMalign", "[green]YES[/green]", "-", tmalign_path)
        else:
            eval_table.add_row(
                "TMalign", "[yellow]NO[/yellow]", "-", "TM-score uses BioPython fallback"
            )
    except Exception as e:
        eval_table.add_row("TMalign", "[red]ERROR[/red]", str(e), "-")

    console.print(eval_table)

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
