"""Backbone generation commands (RFdiffusion)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(help="Backbone generation tools")
console = Console()


@app.command("rfdiffusion")
def rfdiffusion_run(
    contigs: str = typer.Option(
        "[100-200]",
        "--contigs",
        help="RFdiffusion contig string, e.g. '[100-200]' or '[B1-100/0 100-100]'",
    ),
    num_designs: int = typer.Option(1, "--num-designs", help="Number of backbones to generate"),
    input_pdb: Optional[Path] = typer.Option(
        None, "--input-pdb", help="Optional input PDB for conditioning"
    ),
    config_name: Optional[str] = typer.Option(
        None, "--config-name", help="Hydra config-name (e.g. symmetry)"
    ),
    override: list[str] = typer.Option(
        None, "--override", help="Additional hydra overrides (repeatable)"
    ),
    ckpt: Optional[Path] = typer.Option(None, "--ckpt", help="Checkpoint override path"),
    python_exe: Optional[Path] = typer.Option(
        None, "--python", help="Python executable for RFdiffusion env"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    job_id: Optional[str] = typer.Option(None, "--job-id", "-j", help="Job identifier"),
    auto_install: bool = typer.Option(
        False, "--auto-install", help="Auto-install RFdiffusion via git clone"
    ),
):
    """Run RFdiffusion backbone generation."""
    from protein_design_hub.core.config import get_settings
    from protein_design_hub.design.generators.registry import get_generator
    from protein_design_hub.design.generators.types import BackboneInput

    settings = get_settings()
    if output is None:
        output = settings.output.base_dir

    if job_id is None:
        job_id = f"rfdiff_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir = Path(output) / job_id / "backbones" / "rfdiffusion"
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = get_generator("rfdiffusion", settings)
    console.print(f"\n[bold]Running RFdiffusion[/bold]")
    console.print(f"  Output: {out_dir}")

    res = gen.generate(
        BackboneInput(
            job_id=job_id,
            output_dir=out_dir,
            num_designs=num_designs,
            contigs=contigs,
            input_pdb=input_pdb,
            config_name=config_name,
            overrides=override or [],
            ckpt_override_path=ckpt,
            python_executable=python_exe,
        ),
        auto_install=auto_install,
    )

    if not res.success:
        console.print(f"[red]RFdiffusion failed:[/red] {res.error_message}")
        raise typer.Exit(1)

    console.print(f"[green]✓ Generated {len(res.backbone_paths)} backbone files[/green]")
    for p in res.backbone_paths[:20]:
        console.print(str(p))


@app.callback()
def _callback():
    """Backbone generation commands."""
    pass
