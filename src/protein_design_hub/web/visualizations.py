"""Visualization utilities for protein structure analysis."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def create_pae_heatmap(
    pae_data: List[List[float]],
    title: str = "Predicted Aligned Error (PAE)",
    colorscale: str = "Greens_r",
    max_value: float = 30.0,
) -> "plotly.graph_objects.Figure":
    """
    Create a PAE (Predicted Aligned Error) heatmap visualization.

    Args:
        pae_data: 2D array of PAE values [residue_i][residue_j].
        title: Plot title.
        colorscale: Plotly colorscale name.
        max_value: Maximum value for color scale.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    pae_array = np.array(pae_data)
    n_residues = len(pae_array)

    fig = go.Figure(data=go.Heatmap(
        z=pae_array,
        x=list(range(1, n_residues + 1)),
        y=list(range(1, n_residues + 1)),
        colorscale=colorscale,
        zmin=0,
        zmax=max_value,
        colorbar=dict(
            title="PAE (Å)",
            titleside="right",
        ),
        hovertemplate="Residue %{x} vs %{y}<br>PAE: %{z:.2f} Å<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Aligned Residue", scaleanchor="y"),
        yaxis=dict(title="Scored Residue", autorange="reversed"),
        width=600,
        height=600,
    )

    return fig


def create_plddt_plot(
    plddt_values: List[float],
    title: str = "Per-Residue pLDDT Confidence",
    chain_breaks: Optional[List[int]] = None,
) -> "plotly.graph_objects.Figure":
    """
    Create a pLDDT confidence plot with quality regions.

    Args:
        plddt_values: List of pLDDT values per residue.
        title: Plot title.
        chain_breaks: Optional list of residue positions where chains break.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    residues = list(range(1, len(plddt_values) + 1))

    fig = go.Figure()

    # Add colored background regions for quality
    fig.add_hrect(y0=90, y1=100, fillcolor="#0053d6", opacity=0.15, line_width=0,
                  annotation_text="Very High", annotation_position="top right")
    fig.add_hrect(y0=70, y1=90, fillcolor="#65cbf3", opacity=0.15, line_width=0,
                  annotation_text="Confident", annotation_position="top right")
    fig.add_hrect(y0=50, y1=70, fillcolor="#ffdb13", opacity=0.15, line_width=0,
                  annotation_text="Low", annotation_position="top right")
    fig.add_hrect(y0=0, y1=50, fillcolor="#ff7d45", opacity=0.15, line_width=0,
                  annotation_text="Very Low", annotation_position="top right")

    # Add pLDDT line
    fig.add_trace(go.Scatter(
        x=residues,
        y=plddt_values,
        mode='lines',
        line=dict(color='#1f77b4', width=2),
        name='pLDDT',
        hovertemplate="Residue %{x}<br>pLDDT: %{y:.1f}<extra></extra>",
    ))

    # Add chain breaks if provided
    if chain_breaks:
        for pos in chain_breaks:
            fig.add_vline(x=pos, line_dash="dash", line_color="gray", opacity=0.5)

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Residue"),
        yaxis=dict(title="pLDDT", range=[0, 100]),
        height=400,
        showlegend=False,
    )

    return fig


def create_contact_map(
    model_contacts: np.ndarray,
    reference_contacts: Optional[np.ndarray] = None,
    threshold: float = 8.0,
    title: str = "Contact Map",
) -> "plotly.graph_objects.Figure":
    """
    Create a contact map visualization.

    Args:
        model_contacts: Distance matrix for model.
        reference_contacts: Optional distance matrix for reference (for comparison).
        threshold: Contact distance threshold in Angstroms.
        title: Plot title.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_residues = len(model_contacts)

    if reference_contacts is not None:
        # Create comparison view
        fig = make_subplots(rows=1, cols=1)

        # Model contacts (upper triangle) - blue
        model_binary = (model_contacts < threshold).astype(float)
        # Reference contacts (lower triangle) - green
        ref_binary = (reference_contacts < threshold).astype(float)

        # Create combined matrix
        combined = np.zeros_like(model_contacts)
        # Upper triangle: model
        combined[np.triu_indices(n_residues, k=1)] = model_binary[np.triu_indices(n_residues, k=1)]
        # Lower triangle: reference
        combined[np.tril_indices(n_residues, k=-1)] = ref_binary[np.tril_indices(n_residues, k=-1)] * 0.5

        # Color: 0=no contact, 0.5=reference only, 1=model only
        # For overlap, we need a different approach
        model_upper = np.triu(model_binary, k=1)
        ref_lower = np.tril(ref_binary, k=-1)

        fig.add_trace(go.Heatmap(
            z=model_upper + ref_lower.T,
            colorscale=[
                [0, 'white'],
                [0.25, 'lightgreen'],
                [0.5, 'green'],
                [0.75, 'lightblue'],
                [1, 'blue']
            ],
            showscale=False,
            hovertemplate="Residue %{x} vs %{y}<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=f"{title}<br><sup>Upper: Model, Lower: Reference</sup>", x=0.5),
        )
    else:
        # Model only
        model_binary = (model_contacts < threshold).astype(float)

        fig = go.Figure(data=go.Heatmap(
            z=model_binary,
            colorscale=[[0, 'white'], [1, '#1f77b4']],
            showscale=False,
            hovertemplate="Residue %{x} vs %{y}<extra></extra>",
        ))

        fig.update_layout(title=dict(text=title, x=0.5))

    fig.update_layout(
        xaxis=dict(title="Residue", scaleanchor="y"),
        yaxis=dict(title="Residue", autorange="reversed"),
        width=600,
        height=600,
    )

    return fig


def compute_contact_map_from_structure(
    structure_path: Path,
    atom_selection: str = "CA",
) -> np.ndarray:
    """
    Compute distance matrix from a structure file.

    Args:
        structure_path: Path to PDB/CIF file.
        atom_selection: Atom type to use (CA, CB, all).

    Returns:
        Distance matrix as numpy array.
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser
        from scipy.spatial.distance import cdist

        structure_path = Path(structure_path)

        # Load structure
        if structure_path.suffix.lower() in ['.cif', '.mmcif']:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure('structure', str(structure_path))

        # Get coordinates
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] == ' ':  # Standard residue
                        if atom_selection == "CA" and "CA" in residue:
                            coords.append(residue["CA"].get_coord())
                        elif atom_selection == "CB":
                            if "CB" in residue:
                                coords.append(residue["CB"].get_coord())
                            elif "CA" in residue:  # Glycine
                                coords.append(residue["CA"].get_coord())
            break  # First model only

        if not coords:
            raise ValueError("No atoms found in structure")

        coords = np.array(coords)
        distances = cdist(coords, coords)

        return distances

    except ImportError:
        raise ImportError("Biopython and scipy required. Install with: pip install biopython scipy")


def load_pae_from_json(json_path: Path) -> Optional[List[List[float]]]:
    """
    Load PAE data from AlphaFold/ColabFold JSON output.

    Args:
        json_path: Path to JSON file containing PAE data.

    Returns:
        2D list of PAE values or None if not found.
    """
    try:
        with open(json_path) as f:
            data = json.load(f)

        # Try different formats
        if "pae" in data:
            return data["pae"]
        if "predicted_aligned_error" in data:
            return data["predicted_aligned_error"]
        if isinstance(data, list) and len(data) > 0:
            if "predicted_aligned_error" in data[0]:
                return data[0]["predicted_aligned_error"]
            if "pae" in data[0]:
                return data[0]["pae"]

        return None
    except Exception:
        return None


def create_lddt_comparison_chart(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ["lddt", "tm_score", "rmsd"],
) -> "plotly.graph_objects.Figure":
    """
    Create a comparison bar chart for multiple predictors.

    Args:
        results: Dictionary mapping predictor names to their metric results.
        metrics: List of metrics to include.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    predictors = list(results.keys())

    fig = go.Figure()

    colors = {
        "lddt": "#1f77b4",
        "tm_score": "#2ca02c",
        "rmsd": "#d62728",
        "qs_score": "#9467bd",
        "dockq": "#ff7f0e",
    }

    for metric in metrics:
        values = []
        for pred in predictors:
            val = results[pred].get(metric, 0) or 0
            # Invert RMSD for consistent "higher is better" display
            if metric == "rmsd" and val > 0:
                val = 1 / (1 + val)  # Transform to 0-1 scale
            values.append(val)

        if any(v > 0 for v in values):
            fig.add_trace(go.Bar(
                name=metric.upper().replace("_", " "),
                x=[p.upper() for p in predictors],
                y=values,
                marker_color=colors.get(metric, "#7f7f7f"),
            ))

    fig.update_layout(
        barmode='group',
        title="Predictor Comparison",
        xaxis_title="Predictor",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        height=400,
    )

    return fig


def create_per_residue_comparison(
    model_lddt: List[float],
    reference_lddt: Optional[List[float]] = None,
    labels: Tuple[str, str] = ("Model", "Reference"),
) -> "plotly.graph_objects.Figure":
    """
    Create per-residue lDDT comparison plot.

    Args:
        model_lddt: Model per-residue lDDT values.
        reference_lddt: Optional reference per-residue values.
        labels: Labels for model and reference.

    Returns:
        Plotly Figure object.
    """
    import plotly.graph_objects as go

    residues = list(range(1, len(model_lddt) + 1))

    fig = go.Figure()

    # Add quality region backgrounds
    fig.add_hrect(y0=0.9, y1=1.0, fillcolor="#0053d6", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0.7, y1=0.9, fillcolor="#65cbf3", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0.5, y1=0.7, fillcolor="#ffdb13", opacity=0.1, line_width=0)
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor="#ff7d45", opacity=0.1, line_width=0)

    fig.add_trace(go.Scatter(
        x=residues,
        y=model_lddt,
        mode='lines',
        name=labels[0],
        line=dict(color='#1f77b4', width=2),
    ))

    if reference_lddt:
        fig.add_trace(go.Scatter(
            x=residues,
            y=reference_lddt,
            mode='lines',
            name=labels[1],
            line=dict(color='#2ca02c', width=2, dash='dash'),
        ))

    fig.update_layout(
        title="Per-Residue lDDT",
        xaxis_title="Residue",
        yaxis_title="lDDT",
        yaxis_range=[0, 1],
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    return fig
