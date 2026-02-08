"""CLI commands for LLM agent meetings and diagnostics."""

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional

app = typer.Typer(help="LLM agent meetings and diagnostics")
console = Console()


@app.command("status")
def agents_status():
    """Show LLM backend configuration and connectivity."""
    from protein_design_hub.core.config import get_settings, LLM_PROVIDER_PRESETS

    settings = get_settings()
    cfg = settings.llm.resolve()

    console.print("\n[bold blue]LLM Agent Configuration[/bold blue]\n")

    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value")

    table.add_row("Provider", cfg.provider)
    table.add_row("Base URL", cfg.base_url)
    table.add_row("Model", cfg.model)
    key_display = cfg.api_key[:8] + "..." if len(cfg.api_key) > 8 else cfg.api_key
    table.add_row("API Key", key_display)
    table.add_row("Temperature", str(cfg.temperature))
    table.add_row("Max Tokens", str(cfg.max_tokens))
    table.add_row("Rounds/Meeting", str(cfg.num_rounds))
    console.print(table)

    # Test connectivity
    console.print("\n[bold]Connectivity test:[/bold]")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        models = client.models.list()
        model_names = [m.id for m in models.data]
        if cfg.model in model_names:
            console.print(f"  [green]OK[/green] – model '{cfg.model}' available")
        else:
            console.print(f"  [yellow]WARNING[/yellow] – model '{cfg.model}' not in listed models (may still work)")
            if model_names:
                console.print(f"  Available: {', '.join(model_names[:10])}")
    except Exception as e:
        console.print(f"  [red]FAILED[/red] – {e}")
        if cfg.provider == "ollama":
            console.print("  [dim]Make sure Ollama is running: ollama serve[/dim]")
        elif cfg.provider in ("deepseek", "openai", "gemini", "kimi"):
            console.print(f"  [dim]Check your API key for {cfg.provider}[/dim]")

    # Show available providers
    console.print("\n[bold]Available provider presets:[/bold]")
    preset_table = Table()
    preset_table.add_column("Provider", style="cyan")
    preset_table.add_column("Type")
    preset_table.add_column("Default Model")
    preset_table.add_column("Base URL")

    for name, (url, model, _) in LLM_PROVIDER_PRESETS.items():
        ptype = "[green]local[/green]" if name in ("ollama", "lmstudio", "vllm", "llamacpp") else "[yellow]cloud[/yellow]"
        current = " [bold green]← active[/bold green]" if name == cfg.provider else ""
        preset_table.add_row(f"{name}{current}", ptype, model, url)

    console.print(preset_table)
    console.print()


@app.command("list")
def list_agents():
    """List all available scientist agents."""
    from protein_design_hub.agents import scientists as S

    console.print("\n[bold blue]Available Scientist Agents[/bold blue]\n")

    table = Table()
    table.add_column("Agent", style="cyan")
    table.add_column("Expertise")

    for agent in [
        S.PRINCIPAL_INVESTIGATOR,
        S.SCIENTIFIC_CRITIC,
        S.STRUCTURAL_BIOLOGIST,
        S.COMPUTATIONAL_BIOLOGIST,
        S.MACHINE_LEARNING_SPECIALIST,
        S.IMMUNOLOGIST,
        S.PROTEIN_ENGINEER,
        S.BIOPHYSICIST,
        S.DIGITAL_RECEP,
        S.LIAM,
    ]:
        console.print(f"  [bold cyan]{agent.title}[/bold cyan]")
        console.print(f"    Expertise: {agent.expertise}")
        console.print(f"    Goal: {agent.goal}")
        console.print(f"    Model: {agent.resolved_model}")
        console.print()


@app.command("run")
def agents_run(
    input_file: Path = typer.Argument(..., help="Input FASTA file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    reference: Optional[Path] = typer.Option(
        None, "--reference", "-r", help="Reference PDB/CIF"
    ),
    predictors: Optional[str] = typer.Option(
        None, "--predictors", "-p",
        help="Comma-separated predictors"
    ),
    job_id: Optional[str] = typer.Option(
        None, "--job-id", "-j", help="Custom job identifier"
    ),
    rounds: Optional[int] = typer.Option(
        None, "--rounds", help="Discussion rounds per meeting"
    ),
):
    """Run the full LLM-guided pipeline (shortcut for 'pdhub pipeline run --llm').

    This runs input → planning meeting → predict → review → evaluate →
    compare → evaluation review → report, with LLM agents guiding every
    decision point.
    """
    from protein_design_hub.cli.commands.pipeline import pipeline_run

    pipeline_run(
        input_file=input_file,
        output=output,
        reference=reference,
        predictors=predictors,
        job_id=job_id,
        llm=True,
        provider=None,
        model=None,
        rounds=rounds,
    )


@app.command("meet")
def run_meeting_cmd(
    agenda: str = typer.Argument(..., help="Meeting agenda / question"),
    meeting_type: str = typer.Option(
        "team", "--type", "-t",
        help="Meeting type: 'team' or 'individual'"
    ),
    team: str = typer.Option(
        "default", "--team",
        help="Team preset: default, design, nanobody, evaluation, refinement"
    ),
    num_rounds: Optional[int] = typer.Option(
        None, "--rounds", "-r",
        help="Number of discussion rounds (default from config)"
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/meetings"), "--output", "-o",
        help="Directory to save meeting transcripts"
    ),
    save_name: str = typer.Option(
        "discussion", "--name", "-n",
        help="Base filename for the transcript"
    ),
):
    """Run an LLM agent meeting interactively."""
    from protein_design_hub.agents.meeting import run_meeting
    from protein_design_hub.agents import scientists as S
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    rounds = num_rounds if num_rounds is not None else settings.llm.num_rounds

    # Select team composition
    team_presets = {
        "default": (S.DEFAULT_TEAM_LEAD, S.DEFAULT_TEAM_MEMBERS),
        "design": (S.DEFAULT_TEAM_LEAD, S.DESIGN_TEAM_MEMBERS),
        "nanobody": (S.DEFAULT_TEAM_LEAD, S.NANOBODY_TEAM_MEMBERS),
        "evaluation": (S.DEFAULT_TEAM_LEAD, S.EVALUATION_TEAM_MEMBERS),
        "refinement": (S.DEFAULT_TEAM_LEAD, S.REFINEMENT_TEAM_MEMBERS),
    }

    console.print(f"\n[bold blue]Agent Meeting[/bold blue]\n")
    resolved = settings.llm.resolve()
    console.print(f"  Type: {meeting_type}")
    console.print(f"  Rounds: {rounds}")
    console.print(f"  Model: {resolved.model} @ {resolved.base_url}")
    console.print(f"  Agenda: {agenda}")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Running meeting...", total=None)

        try:
            if meeting_type == "team":
                lead, members = team_presets.get(team, team_presets["default"])
                summary = run_meeting(
                    meeting_type="team",
                    agenda=agenda,
                    save_dir=output_dir,
                    save_name=save_name,
                    team_lead=lead,
                    team_members=members,
                    num_rounds=rounds,
                    return_summary=True,
                )
            elif meeting_type == "individual":
                # Use first non-critic member from the team
                _, members = team_presets.get(team, team_presets["default"])
                member = next(m for m in members if m != S.SCIENTIFIC_CRITIC)
                summary = run_meeting(
                    meeting_type="individual",
                    agenda=agenda,
                    save_dir=output_dir,
                    save_name=save_name,
                    team_member=member,
                    num_rounds=rounds,
                    return_summary=True,
                )
            else:
                console.print(f"[red]Unknown meeting type: {meeting_type}[/red]")
                raise typer.Exit(1)
        except Exception as e:
            console.print(f"\n[red]Meeting failed: {e}[/red]")
            if "Connection" in str(e) or "refused" in str(e):
                console.print("[dim]Is your LLM backend running? (e.g. ollama serve)[/dim]")
            raise typer.Exit(1)

    console.print(f"\n[bold green]Meeting Complete![/bold green]\n")
    console.print("[bold]Summary:[/bold]\n")
    console.print(summary or "(no summary)")
    console.print(f"\n[dim]Transcript saved to: {output_dir}/{save_name}.md[/dim]\n")


@app.callback()
def callback():
    """LLM agent commands."""
    pass
