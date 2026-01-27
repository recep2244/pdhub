"""Design command (ProteinMPNN) and sequence analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Sequence design and analysis")
console = Console()


@app.command("sequence")
def analyze_sequence(
    fasta: Optional[Path] = typer.Argument(None, help="FASTA file (if omitted, reads from stdin)"),
):
    """Compute sequence metrics (pI, charge, GRAVY, etc.)."""
    import sys
    from protein_design_hub.analysis.sequence_metrics import compute_sequence_metrics

    if fasta is None:
        data = sys.stdin.read()
    else:
        if not fasta.exists():
            console.print(f"[red]File not found: {fasta}[/red]")
            raise typer.Exit(1)
        data = fasta.read_text()

    # Basic FASTA parse (first record)
    seq_lines = []
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq_lines:
                break
            continue
        seq_lines.append(line)

    seq = "".join(seq_lines)
    metrics = compute_sequence_metrics(seq)

    table = Table(title="Sequence Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Length", str(metrics.length))
    table.add_row("Molecular weight", f"{metrics.molecular_weight:.1f} Da")
    table.add_row("pI", f"{metrics.isoelectric_point:.2f}")
    table.add_row("Net charge (pH 7)", f"{metrics.net_charge_ph7:.2f}")
    table.add_row("GRAVY", f"{metrics.gravy:.3f}")
    table.add_row("Aromaticity", f"{metrics.aromaticity:.3f}")
    table.add_row("Instability index", f"{metrics.instability_index:.2f}")
    console.print(table)


@app.command("mpnn")
def design_mpnn(
    backbone: Path = typer.Argument(..., help="Backbone structure file (PDB/mmCIF)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    job_id: Optional[str] = typer.Option(None, "--job-id", "-j", help="Job identifier"),
    num_sequences: int = typer.Option(8, "--num-seqs", help="Number of sequences to sample"),
    temperature: float = typer.Option(0.1, "--temp", help="Sampling temperature"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    auto_install: bool = typer.Option(
        False, "--auto-install", help="Auto-install ProteinMPNN via git clone"
    ),
):
    """Run ProteinMPNN fixed-backbone sequence design."""
    from datetime import datetime
    import json

    from protein_design_hub.core.config import get_settings
    from protein_design_hub.design.registry import get_designer
    from protein_design_hub.design.types import DesignInput

    if not backbone.exists():
        console.print(f"[red]Backbone not found: {backbone}[/red]")
        raise typer.Exit(1)

    settings = get_settings()
    if output is None:
        output = settings.output.base_dir

    if job_id is None:
        job_id = f"mpnn_{backbone.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    job_dir = Path(output) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    designer = get_designer("proteinmpnn", settings)
    console.print(f"\n[bold]Running ProteinMPNN on {backbone.name}[/bold]")
    console.print(f"  Output: {job_dir}")

    result = designer.design(
        DesignInput(
            job_id=job_id,
            backbone_path=backbone,
            output_dir=job_dir,
            num_sequences=num_sequences,
            temperature=temperature,
            seed=seed,
        ),
        auto_install=auto_install,
    )

    if not result.success:
        console.print(f"[red]Design failed:[/red] {result.error_message}")
        raise typer.Exit(1)

    # Write FASTA
    fasta_path = job_dir / "designed.fasta"
    lines = []
    for s in result.sequences:
        lines.append(f">{s.id}")
        lines.append(s.sequence)
    fasta_path.write_text("\n".join(lines) + "\n")

    # Write summary
    summary = {
        "job_id": result.job_id,
        "designer": result.designer,
        "success": result.success,
        "runtime_seconds": result.runtime_seconds,
        "num_sequences": len(result.sequences),
        "fasta_path": str(fasta_path),
        "metadata": result.metadata,
    }
    (job_dir / "design_summary.json").write_text(json.dumps(summary, indent=2))

    console.print(f"\n[green]✓ Designed {len(result.sequences)} sequences[/green]")
    console.print(f"FASTA: {fasta_path}")


@app.callback()
def _callback():
    """Design tools."""
    pass
