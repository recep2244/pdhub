"""Energy/scoring integrations (Rosetta, FoldX, OpenMM)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Energy and scoring tools")
console = Console()


@app.command("openmm-gbsa")
def openmm_gbsa(
    model: Path = typer.Argument(..., help="Model structure file (PDB)"),
    minimize: bool = typer.Option(True, "--minimize/--no-minimize", help="Minimize before scoring"),
    max_iters: int = typer.Option(200, "--max-iters", help="Minimization iterations"),
):
    """Compute OpenMM implicit-solvent energy (OBC2) and GBSA term."""
    from protein_design_hub.evaluation.metrics.openmm_gbsa import OpenMMGBSAMetric

    metric = OpenMMGBSAMetric(minimize=minimize, max_iters=max_iters)
    if not metric.is_available():
        console.print("[red]OpenMM not available.[/red] Install with: `pip install openmm`")
        raise typer.Exit(1)

    res = metric.compute(model)
    table = Table(title="OpenMM GBSA")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Potential (kJ/mol)", f"{res['openmm_potential_energy_kj_mol']:.3f}")
    table.add_row("GBSA term (kJ/mol)", f"{res['openmm_gbsa_energy_kj_mol']:.3f}")
    table.add_row("Minimized", str(res["minimized"]))
    console.print(table)


@app.command("rosetta-score")
def rosetta_score(
    model: Path = typer.Argument(..., help="Model structure file (PDB)"),
    out_dir: Optional[Path] = typer.Option(None, "--out", "-o", help="Output directory"),
):
    """Score a structure with Rosetta score_jd2 (total_score)."""
    from protein_design_hub.energy.rosetta import run_score_jd2

    out_dir = Path(out_dir) if out_dir else Path(".") / ".pdhub_rosetta_score"
    scores = run_score_jd2(model, out_dir)
    table = Table(title="Rosetta score_jd2")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("total_score (REU)", f"{scores['total_score']:.3f}")
    console.print(table)


@app.command("rosetta-relax")
def rosetta_relax(
    model: Path = typer.Argument(..., help="Input PDB"),
    out_dir: Path = typer.Option(Path("./rosetta_relax"), "--out", "-o", help="Output directory"),
    nstruct: int = typer.Option(1, "--nstruct", help="Number of output structures"),
):
    """Run Rosetta relax and write outputs to a directory."""
    from protein_design_hub.energy.rosetta import run_relax

    out_dir.mkdir(parents=True, exist_ok=True)
    best = run_relax(model, out_dir, nstruct=nstruct)
    console.print(f"[green]✓[/green] Best relaxed PDB: {best}")


@app.command("rosetta-cartesian-ddg")
def rosetta_cartesian_ddg(
    model: Path = typer.Argument(..., help="Input PDB"),
    mutfile: Path = typer.Argument(..., help="Rosetta mutfile for cartesian_ddg"),
    out_dir: Path = typer.Option(
        Path("./rosetta_cart_ddg"), "--out", "-o", help="Output directory"
    ),
):
    """Run Rosetta cartesian_ddg and parse ddG."""
    from protein_design_hub.energy.rosetta import run_cartesian_ddg

    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_cartesian_ddg(model, out_dir, mutfile)
    console.print(f"[green]✓[/green] cartesian_ddg: {res['cartesian_ddg']:.3f} REU")


@app.command("foldx-ddg")
def foldx_ddg(
    model: Path = typer.Argument(..., help="Wildtype structure (PDB)"),
    mutant_file: Path = typer.Argument(..., help="FoldX mutant file (individual_list.txt style)"),
    out_dir: Path = typer.Option(Path("./foldx_out"), "--out", "-o", help="Output directory"),
):
    """Run FoldX BuildModel and parse ΔΔG."""
    from protein_design_hub.energy.foldx import run_foldx_buildmodel

    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_foldx_buildmodel(model, mutant_file, out_dir)
    console.print(f"[green]✓[/green] FoldX ΔΔG: {res['foldx_ddg_kcal_mol']:.3f} kcal/mol")


@app.callback()
def _callback():
    """Energy and scoring commands."""
    pass
