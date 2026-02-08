"""Unified pipeline CLI – run everything from FASTA to report in one command.

Usage examples
--------------
# Step-only (fast, no LLM):
  pdhub pipeline run input.fasta

# LLM-guided (agents plan, review, interpret):
  pdhub pipeline run input.fasta --llm

# With reference structure for evaluation:
  pdhub pipeline run input.fasta --llm -r reference.pdb

# Pick specific predictors:
  pdhub pipeline run input.fasta --llm -p colabfold,chai1

# Use a different LLM provider:
  pdhub pipeline run input.fasta --llm --provider deepseek

# Dry-run: show what would happen without running:
  pdhub pipeline plan input.fasta --llm
"""

import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

app = typer.Typer(help="Run the full prediction → evaluation → report pipeline")
console = Console()


def _make_progress_callback(console_obj: Console, start_time: float):
    """Return a callback that prints rich-formatted progress updates."""
    _seen: set = set()

    def _cb(stage: str, item: str, current: int, total: int):
        key = f"{stage}:{item}"
        if key in _seen:
            return
        _seen.add(key)

        from protein_design_hub.agents.orchestrator import _AGENT_LABELS

        label = _AGENT_LABELS.get(item, item)
        elapsed = time.time() - start_time
        is_llm = item.startswith("llm_")
        icon = "🧠" if is_llm else "⚙️"
        style = "bold magenta" if is_llm else "bold cyan"
        console_obj.print(
            f"  {icon} [{style}][{current}/{total}][/{style}] {label}  "
            f"[dim]({elapsed:.0f}s elapsed)[/dim]"
        )

    return _cb


@app.command("run")
def pipeline_run(
    input_file: Path = typer.Argument(..., help="Input FASTA file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory (default from config)"
    ),
    reference: Optional[Path] = typer.Option(
        None, "--reference", "-r", help="Reference PDB/CIF for evaluation"
    ),
    predictors: Optional[str] = typer.Option(
        None, "--predictors", "-p",
        help="Comma-separated predictors (default: all enabled)"
    ),
    job_id: Optional[str] = typer.Option(
        None, "--job-id", "-j", help="Custom job identifier"
    ),
    llm: bool = typer.Option(
        False, "--llm", help="Enable LLM-guided pipeline (planning + review meetings)"
    ),
    provider: Optional[str] = typer.Option(
        None, "--provider",
        help="Override LLM provider (ollama, deepseek, openai, gemini, kimi, ...)"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="Override LLM model name"
    ),
    rounds: Optional[int] = typer.Option(
        None, "--rounds", help="LLM discussion rounds per meeting"
    ),
):
    """Run the full pipeline: input → predict → evaluate → compare → report.

    By default runs the fast computational-only pipeline.
    Add --llm to enable LLM agent meetings at key decision points.
    """
    # ── Validate inputs ──────────────────────────────────────────────
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    if reference and not reference.exists():
        console.print(f"[red]Error: Reference file not found: {reference}[/red]")
        raise typer.Exit(1)

    # ── Apply provider / model overrides ─────────────────────────────
    if provider or model:
        _apply_llm_overrides(provider, model)

    # ── Build orchestrator ───────────────────────────────────────────
    from protein_design_hub.agents import AgentOrchestrator
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    mode = "llm" if llm else "step"
    start_time = time.time()
    progress_cb = _make_progress_callback(console, start_time)

    orchestrator = AgentOrchestrator(
        mode=mode,
        progress_callback=progress_cb,
        **({"num_rounds": rounds} if rounds else {}),
    )

    predictor_list = [p.strip() for p in predictors.split(",")] if predictors else None

    # ── Print banner ─────────────────────────────────────────────────
    _print_banner(mode, input_file, reference, predictor_list, settings, orchestrator)

    # ── Run ──────────────────────────────────────────────────────────
    console.print()
    result = orchestrator.run(
        input_path=input_file,
        output_dir=output,
        reference_path=reference,
        predictors=predictor_list,
        job_id=job_id,
    )

    elapsed = time.time() - start_time
    console.print()

    # ── Handle result ────────────────────────────────────────────────
    if not result.success:
        console.print(Panel(
            f"[red bold]Pipeline failed[/red bold]\n\n{result.message}",
            title="Error",
            border_style="red",
        ))
        if result.error:
            console.print(f"[dim]{result.error}[/dim]")
        raise typer.Exit(1)

    ctx = result.context
    _print_results(ctx, elapsed, output, settings)


@app.command("plan")
def pipeline_plan(
    input_file: Path = typer.Argument(..., help="Input FASTA file"),
    llm: bool = typer.Option(
        False, "--llm", help="Show LLM-guided pipeline plan"
    ),
):
    """Dry-run: show the agent pipeline without executing it."""
    from protein_design_hub.agents import AgentOrchestrator

    mode = "llm" if llm else "step"
    orchestrator = AgentOrchestrator(mode=mode)

    console.print(f"\n[bold blue]Pipeline Plan ({mode} mode)[/bold blue]\n")

    steps = orchestrator.describe_pipeline()
    tree = Tree("[bold]Pipeline[/bold]")
    for i, s in enumerate(steps, 1):
        icon = "🧠" if s["type"] == "llm" else "⚙️"
        style = "magenta" if s["type"] == "llm" else "cyan"
        tree.add(f"{icon} [{style}]{i}. {s['label']}[/{style}]")

    console.print(tree)
    console.print()

    if mode == "step":
        console.print("[dim]Add --llm to include LLM agent meetings for planning and review.[/dim]\n")
    else:
        _print_llm_config()


@app.command("status")
def pipeline_status():
    """Show full system status: predictors, LLM, and GPU."""
    from protein_design_hub.core.config import get_settings

    settings = get_settings()

    console.print("\n[bold blue]Protein Design Hub – System Status[/bold blue]\n")

    # 1. Predictors
    _print_predictor_status(settings)

    # 2. LLM
    console.print()
    _print_llm_config()

    # 3. GPU
    console.print()
    _print_gpu_status()

    console.print()


# ── Internal helpers ─────────────────────────────────────────────────


def _apply_llm_overrides(provider: Optional[str], model: Optional[str]):
    """Apply CLI overrides to the LLM config (in-memory only)."""
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    if provider:
        settings.llm.provider = provider
        # Reset auto-filled fields so resolve() picks up new provider
        settings.llm.base_url = ""
        settings.llm.model = ""
        settings.llm.api_key = ""
    if model:
        settings.llm.model = model


def _print_banner(mode, input_file, reference, predictor_list, settings, orchestrator):
    """Print a startup banner."""
    mode_label = (
        "[bold magenta]LLM-guided[/bold magenta] (meetings + computation)"
        if mode == "llm"
        else "[bold cyan]Step-only[/bold cyan] (computation)"
    )

    lines = [
        f"  Mode      : {mode_label}",
        f"  Input     : {input_file}",
    ]
    if reference:
        lines.append(f"  Reference : {reference}")
    if predictor_list:
        lines.append(f"  Predictors: {', '.join(predictor_list)}")
    else:
        lines.append(f"  Predictors: all enabled")

    if mode == "llm":
        resolved = settings.llm.resolve()
        lines.append(f"  LLM       : {resolved.model} @ {resolved.base_url}")

    steps = orchestrator.describe_pipeline()
    lines.append(f"  Steps     : {len(steps)}")

    console.print(Panel(
        "\n".join(lines),
        title="[bold blue]Protein Design Hub – Pipeline[/bold blue]",
        border_style="blue",
    ))


def _print_results(ctx, elapsed, output, settings):
    """Print final results."""
    console.print(Panel(
        f"[bold green]Pipeline completed in {elapsed:.1f}s[/bold green]",
        border_style="green",
    ))

    # Prediction summary
    if ctx.prediction_results:
        pred_table = Table(title="Predictions")
        pred_table.add_column("Predictor", style="cyan")
        pred_table.add_column("Status")
        pred_table.add_column("Structures")
        pred_table.add_column("Runtime")
        pred_table.add_column("Best pLDDT")

        for name, pr in ctx.prediction_results.items():
            if pr.success:
                status = "[green]OK[/green]"
                n = str(len(pr.structure_paths))
                rt = f"{pr.runtime_seconds:.1f}s"
                best = "-"
                if pr.scores:
                    plddts = [s.plddt for s in pr.scores if s.plddt]
                    if plddts:
                        best = f"{max(plddts):.1f}"
            else:
                status = f"[red]FAIL[/red]"
                n, rt, best = "0", "-", "-"
            pred_table.add_row(name, status, n, rt, best)
        console.print(pred_table)

    # Evaluation summary
    if ctx.evaluation_results:
        console.print()
        eval_table = Table(title="Evaluation")
        eval_table.add_column("Predictor", style="cyan")
        eval_table.add_column("lDDT")
        eval_table.add_column("TM-score")
        eval_table.add_column("RMSD")

        for name, ev in ctx.evaluation_results.items():
            eval_table.add_row(
                name,
                f"{ev.lddt:.3f}" if ev.lddt else "-",
                f"{ev.tm_score:.3f}" if ev.tm_score else "-",
                f"{ev.rmsd:.2f} A" if ev.rmsd else "-",
            )
        console.print(eval_table)

    # Ranking
    if ctx.comparison_result and ctx.comparison_result.ranking:
        console.print()
        rank_table = Table(title="Ranking")
        rank_table.add_column("Rank")
        rank_table.add_column("Predictor", style="cyan")
        rank_table.add_column("Score")

        for i, (name, score) in enumerate(ctx.comparison_result.ranking, 1):
            r = f"[bold green]{i}[/bold green]" if i == 1 else str(i)
            rank_table.add_row(r, name, f"{score:.3f}")
        console.print(rank_table)

    # LLM meeting summaries
    llm_keys = ["plan", "prediction_review", "evaluation_review", "refinement_review"]
    summaries = {k: ctx.extra.get(k) for k in llm_keys if ctx.extra.get(k)}
    if summaries:
        console.print()
        console.print("[bold]LLM Agent Summaries:[/bold]")
        for key, text in summaries.items():
            label = key.replace("_", " ").title()
            # Truncate very long summaries for display
            short = text[:500] + "..." if len(text) > 500 else text
            console.print(f"\n  [bold magenta]{label}:[/bold magenta]")
            console.print(f"  {short}")

    # Output location
    job_dir = ctx.job_dir
    if job_dir is None:
        from protein_design_hub.core.config import get_settings
        s = get_settings()
        job_dir = Path(s.output.base_dir) / ctx.job_id

    console.print(f"\n[bold]Results saved to:[/bold] {job_dir}")

    report_path = job_dir / "report" / "report.html"
    if report_path.exists():
        console.print(f"  Report : {report_path}")

    meetings_dir = job_dir / "meetings"
    if meetings_dir.exists():
        transcripts = list(meetings_dir.glob("*.md"))
        if transcripts:
            console.print(f"  Meetings: {meetings_dir} ({len(transcripts)} transcript(s))")

    console.print()


def _print_llm_config():
    """Print resolved LLM configuration."""
    from protein_design_hub.core.config import get_settings

    settings = get_settings()
    cfg = settings.llm.resolve()

    table = Table(title="LLM Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    table.add_row("Provider", cfg.provider)
    table.add_row("Model", cfg.model)
    table.add_row("Base URL", cfg.base_url)
    key_display = cfg.api_key[:8] + "..." if len(cfg.api_key) > 8 else cfg.api_key
    table.add_row("API Key", key_display)
    table.add_row("Temperature", str(cfg.temperature))
    table.add_row("Max Tokens", str(cfg.max_tokens))
    console.print(table)

    # Quick connectivity check
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        if cfg.model in model_ids:
            console.print(f"  [green]Connected[/green] – model '{cfg.model}' available")
        else:
            console.print(f"  [yellow]Connected[/yellow] – model '{cfg.model}' not listed (may still work)")
    except Exception as e:
        console.print(f"  [red]Not connected[/red] – {e}")


def _print_predictor_status(settings):
    """Print installed predictors."""
    from protein_design_hub.predictors.registry import PredictorRegistry

    table = Table(title="Predictors")
    table.add_column("Tool", style="cyan")
    table.add_column("Status")
    table.add_column("GPU")

    for name in PredictorRegistry.list_available():
        try:
            pred = PredictorRegistry.get(name, settings)
            info = pred.get_status()
            status = "[green]INSTALLED[/green]" if info["installed"] else "[red]NOT INSTALLED[/red]"
            gpu = "[green]YES[/green]" if info.get("gpu_available") else "[dim]NO[/dim]"
            table.add_row(name, status, gpu)
        except Exception as e:
            table.add_row(name, f"[red]ERR[/red]", "-")

    console.print(table)


def _print_gpu_status():
    """Print GPU info."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            console.print(f"[bold]GPU:[/bold] [green]{name}[/green] ({mem:.1f} GB)")
        else:
            console.print("[bold]GPU:[/bold] [yellow]Not available[/yellow]")
    except ImportError:
        console.print("[bold]GPU:[/bold] [dim]PyTorch not installed[/dim]")


@app.callback()
def callback():
    """Pipeline commands."""
    pass
