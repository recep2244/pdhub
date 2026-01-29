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



def create_structure_viewer(
    structure_path: Path,
    width: int = 800,
    height: int = 600,
    style: str = "cartoon",
    color_by: str = "spectrum",
    spin: bool = False,
    background_color: str = "white",
) -> str:
    """
    Create a 3D structure viewer using 3Dmol.js.

    Args:
        structure_path: Path to the PDB/CIF file.
        width: Width of the viewer in pixels.
        height: Height of the viewer in pixels.
        style: Representation style ('cartoon', 'stick', 'sphere').
        color_by: Coloring scheme ('spectrum', 'chain', 'residue', 'secondary').
        spin: Whether to auto-spin the structure.
        background_color: Background color name or hex code.

    Returns:
        HTML string containing the viewer.
    """
    structure_path = Path(structure_path)
    model_data = structure_path.read_text()
    
    # Determine format
    file_fmt = "mmcif" if structure_path.suffix == ".cif" else "pdb"

    # Define simple color schemes map for non-standard 3Dmol names if needed
    # But 3Dmol supports 'spectrum', 'chain', etc. directly.
    
    style_spec = {}
    if style == "cartoon":
        style_spec = {"cartoon": {"color": color_by}}
    elif style == "stick":
        style_spec = {"stick": {}}
    elif style == "sphere":
        style_spec = {"sphere": {}}
    
    # Clean up string for JS safety
    model_data = model_data.replace("`", "\`")

    html = f"""
    <div id="mol_viewer_{structure_path.stem}" style="width: {width}px; height: {height}px; position: relative;"></div>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
        $(function() {{
            let element = $("#mol_viewer_{structure_path.stem}");
            let config = {{ backgroundColor: "{background_color}" }};
            let viewer = $3Dmol.createViewer(element, config);
            
            viewer.addModel(`{model_data}`, "{file_fmt}");
            viewer.setStyle({{}}, {json.dumps(style_spec)});
            viewer.zoomTo();
            viewer.render();
            {'viewer.spin("y", 1);' if spin else ''}
        }});
    </script>
    """
    return html


def create_structure_comparison_3d(
    model_path: Path,
    reference_path: Optional[Path] = None,
    highlight_differences: bool = True,
    rmsd_threshold: float = 2.0,
) -> str:
    """
    Create 3D structure comparison viewer HTML.

    Args:
        model_path: Path to model structure.
        reference_path: Optional reference structure for comparison.
        highlight_differences: Whether to highlight structural differences.
        rmsd_threshold: Threshold for highlighting differences.

    Returns:
        HTML string for py3Dmol viewer.
    """
    import uuid
    
    unique_id = f"structure_viewer_{uuid.uuid4().hex[:8]}"
    model_path = Path(model_path)

    # Read model structure
    model_pdb = model_path.read_text().replace("`", "\`")

    html = f"""
    <div id="{unique_id}" style="width: 100%; height: 500px; position: relative;">
    </div>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
    $(function() {{
        let viewer = $3Dmol.createViewer("{unique_id}", {{
            backgroundColor: 'white'
        }});
    """

    # Add model
    html += f"""
        let model = viewer.addModel(`{model_pdb}`, "pdb");
        model.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
    """

    if reference_path:
        reference_path = Path(reference_path)
        ref_pdb = reference_path.read_text().replace("`", "\`")

        html += f"""
        let reference = viewer.addModel(`{ref_pdb}`, "pdb");
        reference.setStyle({{}}, {{cartoon: {{color: 'gray', opacity: 0.5}}}});
        """

    html += """
        viewer.zoomTo();
        viewer.render();
    });
    </script>
    """

    return html



def export_pymol_session(
    structures: List[Tuple[Path, str]],
    output_path: Path,
    superimpose: bool = True,
) -> str:
    """
    Generate PyMOL script for structure comparison.

    Args:
        structures: List of (path, name) tuples for structures.
        output_path: Path for PyMOL script.
        superimpose: Whether to superimpose structures.

    Returns:
        PyMOL script content.
    """
    script_lines = [
        "# PyMOL session generated by Protein Design Hub",
        "from pymol import cmd",
        "",
        "# Set visualization defaults",
        "cmd.set('cartoon_fancy_helices', 1)",
        "cmd.set('cartoon_side_chain_helper', 1)",
        "cmd.bg_color('white')",
        "",
    ]

    # Load structures
    for i, (path, name) in enumerate(structures):
        script_lines.append(f"cmd.load('{path}', '{name}')")

    # Superimpose if requested
    if superimpose and len(structures) > 1:
        script_lines.append("")
        script_lines.append("# Superimpose structures")
        reference = structures[0][1]
        for path, name in structures[1:]:
            script_lines.append(f"cmd.align('{name}', '{reference}')")

    # Coloring
    script_lines.append("")
    script_lines.append("# Apply coloring")
    colors = ['marine', 'red', 'green', 'yellow', 'orange', 'purple']
    for i, (path, name) in enumerate(structures):
        color = colors[i % len(colors)]
        script_lines.append(f"cmd.color('{color}', '{name}')")

    # Final setup
    script_lines.append("")
    script_lines.append("# Show cartoon representation")
    script_lines.append("cmd.show('cartoon', 'all')")
    script_lines.append("cmd.hide('lines', 'all')")
    script_lines.append("cmd.center('all')")
    script_lines.append("cmd.zoom('all', 2)")

    script = "\n".join(script_lines)

    # Save script
    output_path = Path(output_path)
    output_path.write_text(script)

    return script


def export_chimerax_session(
    structures: List[Tuple[Path, str]],
    output_path: Path,
    superimpose: bool = True,
) -> str:
    """
    Generate ChimeraX script for structure comparison.

    Args:
        structures: List of (path, name) tuples for structures.
        output_path: Path for ChimeraX script.
        superimpose: Whether to superimpose structures.

    Returns:
        ChimeraX script content.
    """
    script_lines = [
        "# ChimeraX session generated by Protein Design Hub",
        "",
        "# Set visualization defaults",
        "set bgColor white",
        "lighting soft",
        "",
    ]

    # Load structures
    for i, (path, name) in enumerate(structures):
        script_lines.append(f"open {path}")

    # Superimpose if requested
    if superimpose and len(structures) > 1:
        script_lines.append("")
        script_lines.append("# Superimpose structures")
        for i in range(1, len(structures)):
            script_lines.append(f"matchmaker #{i+1} to #1")

    # Coloring
    script_lines.append("")
    script_lines.append("# Apply coloring")
    colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple']
    for i, (path, name) in enumerate(structures):
        color = colors[i % len(colors)]
        script_lines.append(f"color #{i+1} {color}")

    # Final setup
    script_lines.append("")
    script_lines.append("# Show cartoon representation")
    script_lines.append("hide atoms")
    script_lines.append("show cartoons")
    script_lines.append("view")

    script = "\n".join(script_lines)

    # Save script
    output_path = Path(output_path)
    output_path.write_text(script)

    return script


def calculate_rmsd_per_residue(
    model_path: Path,
    reference_path: Path,
    atom_type: str = "CA",
) -> Tuple[float, List[float]]:
    """
    Calculate per-residue RMSD between two structures.

    Args:
        model_path: Path to model structure.
        reference_path: Path to reference structure.
        atom_type: Atom type for RMSD calculation.

    Returns:
        Tuple of (global_rmsd, per_residue_rmsd_list).
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
        import numpy as np

        def load_structure(path):
            path = Path(path)
            if path.suffix.lower() in ['.cif', '.mmcif']:
                parser = MMCIFParser(QUIET=True)
            else:
                parser = PDBParser(QUIET=True)
            return parser.get_structure('struct', str(path))

        model_struct = load_structure(model_path)
        ref_struct = load_structure(reference_path)

        # Get atoms
        def get_atoms(struct):
            atoms = []
            for model in struct:
                for chain in model:
                    for residue in chain:
                        if residue.id[0] == ' ' and atom_type in residue:
                            atoms.append(residue[atom_type])
                break
            return atoms

        model_atoms = get_atoms(model_struct)
        ref_atoms = get_atoms(ref_struct)

        if len(model_atoms) != len(ref_atoms):
            # Try to align by sequence
            min_len = min(len(model_atoms), len(ref_atoms))
            model_atoms = model_atoms[:min_len]
            ref_atoms = ref_atoms[:min_len]

        if len(model_atoms) == 0:
            return 0.0, []

        # Superimpose
        sup = Superimposer()
        sup.set_atoms(ref_atoms, model_atoms)
        sup.apply(model_atoms)

        global_rmsd = sup.rms

        # Calculate per-residue distances
        per_residue_rmsd = []
        for m_atom, r_atom in zip(model_atoms, ref_atoms):
            dist = np.linalg.norm(m_atom.get_coord() - r_atom.get_coord())
            per_residue_rmsd.append(dist)

        return global_rmsd, per_residue_rmsd

    except ImportError:
        raise ImportError("Biopython required. Install with: pip install biopython")


def create_difference_map(
    model_path: Path,
    reference_path: Path,
    threshold: float = 2.0,
) -> "plotly.graph_objects.Figure":
    """
    Create a per-residue difference map between model and reference.

    Args:
        model_path: Path to model structure.
        reference_path: Path to reference structure.
        threshold: RMSD threshold for highlighting.

    Returns:
        Plotly Figure showing differences.
    """
    import plotly.graph_objects as go

    global_rmsd, per_res_rmsd = calculate_rmsd_per_residue(model_path, reference_path)

    residues = list(range(1, len(per_res_rmsd) + 1))

    # Create figure
    fig = go.Figure()

    # Add threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold ({threshold} Å)")

    # Color by quality
    colors = ['green' if r < threshold else 'red' for r in per_res_rmsd]

    fig.add_trace(go.Bar(
        x=residues,
        y=per_res_rmsd,
        marker_color=colors,
        hovertemplate="Residue %{x}<br>RMSD: %{y:.2f} Å<extra></extra>",
    ))

    # Add global RMSD annotation
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=f"Global RMSD: {global_rmsd:.2f} Å",
        showarrow=False,
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
    )

    fig.update_layout(
        title="Per-Residue Structural Differences",
        xaxis_title="Residue",
        yaxis_title="RMSD (Å)",
        height=400,
    )

    return fig


def create_ramachandran_plot(
    structure_path: Path,
    title: str = "Ramachandran Plot",
) -> "plotly.graph_objects.Figure":
    """
    Create a Ramachandran plot for a structure.

    Args:
        structure_path: Path to structure file.
        title: Plot title.

    Returns:
        Plotly Figure with Ramachandran plot.
    """
    import plotly.graph_objects as go
    import numpy as np

    try:
        from Bio.PDB import PDBParser, MMCIFParser
        from Bio.PDB.Polypeptide import PPBuilder

        structure_path = Path(structure_path)

        if structure_path.suffix.lower() in ['.cif', '.mmcif']:
            parser = MMCIFParser(QUIET=True)
        else:
            parser = PDBParser(QUIET=True)

        structure = parser.get_structure('struct', str(structure_path))

        # Get phi/psi angles
        ppb = PPBuilder()
        phi_psi = []

        for pp in ppb.build_peptides(structure):
            for phi, psi in pp.get_phi_psi_list():
                if phi is not None and psi is not None:
                    phi_psi.append((np.degrees(phi), np.degrees(psi)))

        if not phi_psi:
            fig = go.Figure()
            fig.add_annotation(text="No phi/psi angles found", x=0.5, y=0.5,
                              xref="paper", yref="paper", showarrow=False)
            return fig

        phi_vals, psi_vals = zip(*phi_psi)

        # Create figure with allowed regions
        fig = go.Figure()

        # Add allowed region backgrounds (simplified)
        # Alpha-helix region
        fig.add_shape(type="rect",
                     x0=-80, y0=-60, x1=-40, y1=-20,
                     fillcolor="lightblue", opacity=0.3, line_width=0)

        # Beta-sheet region
        fig.add_shape(type="rect",
                     x0=-150, y0=120, x1=-60, y1=180,
                     fillcolor="lightgreen", opacity=0.3, line_width=0)

        # Left-handed helix region
        fig.add_shape(type="rect",
                     x0=40, y0=20, x1=80, y1=60,
                     fillcolor="lightyellow", opacity=0.3, line_width=0)

        # Add data points
        fig.add_trace(go.Scatter(
            x=phi_vals,
            y=psi_vals,
            mode='markers',
            marker=dict(size=5, color='#1f77b4', opacity=0.7),
            hovertemplate="Phi: %{x:.1f}°<br>Psi: %{y:.1f}°<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            xaxis=dict(title="Phi (°)", range=[-180, 180]),
            yaxis=dict(title="Psi (°)", range=[-180, 180]),
            width=500,
            height=500,
        )

        return fig

    except ImportError:
        fig = go.Figure()
        fig.update_layout(title="Install Biopython for Ramachandran Plot")
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Error: {str(e)}")
        return fig


def create_msa_viewer(
    alignment: List[str],
    names: List[str],
    width: int = 800,
    height: int = 400,
    max_sequences: int = 100,
) -> str:
    """
    Create an HTML-based MSA viewer with coloring.

    Args:
        alignment: List of aligned sequences.
        names: List of sequence names.
        width: Widget width.
        height: Widget height.
        max_sequences: Limit sequences for performance.

    Returns:
        HTML string.
    """
    # Truncate if too many
    if len(alignment) > max_sequences:
        alignment = alignment[:max_sequences]
        names = names[:max_sequences]
    
    # Check length consistency
    seq_len = len(alignment[0])
    
    # Basic Clustal-like colors
    colors = {
        'G': '#f79d5c', 'P': '#f79d5c', 'S': '#f79d5c', 'T': '#f79d5c',
        'C': '#f2d388', 'A': '#95a5a6', 'V': '#95a5a6', 'L': '#95a5a6', 'I': '#95a5a6', 'M': '#95a5a6',
        'F': '#81ecec', 'W': '#81ecec', 'Y': '#81ecec',
        'N': '#a29bfe', 'Q': '#a29bfe', 'H': '#a29bfe',
        'D': '#ff7675', 'E': '#ff7675',
        'K': '#74b9ff', 'R': '#74b9ff',
        '-': '#ffffff'
    }
    
    rows_html = []
    
    # Header row (ruler)
    ruler_cells = []
    for i in range(1, seq_len + 1):
        label = str(i) if i % 10 == 0 or i == 1 else ""
        ruler_cells.append(f'<div class="msa-ruler-cell">{label}</div>')
    
    rows_html.append(f"""
    <div class="msa-row">
        <div class="msa-name"></div>
        <div class="msa-seq">{''.join(ruler_cells)}</div>
    </div>
    """)
    
    for name, seq in zip(names, alignment):
        residues = []
        for char in seq:
            c = colors.get(char.upper(), '#ffffff')
            residues.append(f'<div class="msa-res" style="background-color: {c};">{char}</div>')
        
        rows_html.append(f"""
        <div class="msa-row">
            <div class="msa-name" title="{name}">{name[:15]}</div>
            <div class="msa-seq">{''.join(residues)}</div>
        </div>
        """)
    
    css = """
    <style>
    .msa-container {
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
    }
    .msa-row {
        display: flex;
        height: 20px;
    }
    .msa-name {
        width: 150px;
        flex-shrink: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        border-right: 1px solid #eee;
        padding-left: 5px;
        font-size: 12px;
        line-height: 20px;
        background: #f9f9f9;
        position: sticky;
        left: 0;
    }
    .msa-seq {
        display: flex;
    }
    .msa-res {
        width: 12px;
        height: 20px;
        text-align: center;
        font-size: 12px;
        line-height: 20px;
        color: #333;
    }
    .msa-ruler-cell {
        width: 12px;
        height: 20px;
        font-size: 9px;
        color: #888;
        border-bottom: 1px solid #ccc;
    }
    </style>
    """
    
    html = f"""
    {css}
    <div class="msa-container" style="width: 100%; height: {height}px; overflow-y: auto;">
        {''.join(rows_html)}
    </div>
    """
    return html
    """
    Create a Ramachandran plot for a structure.

    Args:
        structure_path: Path to structure file.
        title: Plot title.

    Returns:
        Plotly Figure with Ramachandran plot.
    """
    import plotly.graph_objects as go
    import numpy as np

    try:
        from Bio.PDB import PDBParser, MMCIFParser
        from Bio.PDB.Polypeptide import PPBuilder

        structure_path = Path(structure_path)
        
        # Determine format (PDB/MMCIF)
        if structure_path.suffix.lower() in ['.cif', '.mmcif']:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('structure', str(structure_path))
        else:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('structure', str(structure_path))
            
        builder = PPBuilder()
        pp = builder.build_peptides(structure)
        
        phi_psi = []
        if pp:
            for poly in pp:
                phi_psi.extend(poly.get_phi_psi_list())
                
        # Filter None values
        phi = [pair[0] * 180 / np.pi for pair in phi_psi if pair[0]]
        psi = [pair[1] * 180 / np.pi for pair in phi_psi if pair[1]]
        
        fig = go.Figure(data=go.Scatter(
            x=phi, 
            y=psi, 
            mode='markers',
            marker=dict(
                size=5,
                color='black',
                opacity=0.5
            )
        ))
        
        # Add background regions (simplified)
        # General favored regions
        fig.add_shape(type="rect", x0=-180, y0=-180, x1=180, y1=180,
                      line=dict(color="RoyalBlue"), fillcolor="white", opacity=0.1, layer="below")
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis=dict(title="Phi (degrees)", range=[-180, 180], zeroline=True),
            yaxis=dict(title="Psi (degrees)", range=[-180, 180], zeroline=True),
            width=600,
            height=600,
            shapes=[
                # Alpha-helix
                dict(type="rect", x0=-100, x1=-30, y0=-70, y1=-10, 
                     fillcolor="red", opacity=0.2, line_width=0, layer="below"),
                # Beta-sheet
                dict(type="rect", x0=-180, x1=-45, y0=45, y1=180, 
                     fillcolor="blue", opacity=0.2, line_width=0, layer="below"),
                 dict(type="rect", x0=-180, x1=-45, y0=-180, y1=-135, # Wrap around
                     fillcolor="blue", opacity=0.2, line_width=0, layer="below")
            ]
        )
        
        return fig
        
    except ImportError:
        # Fallback empty figure
        fig = go.Figure()
        fig.update_layout(title="Install Biopython for Ramachandran Plot")
        return fig


def create_msa_viewer(
    alignment: List[str],
    names: List[str],
    width: int = 800,
    height: int = 400,
    max_sequences: int = 100,
) -> str:
    """
    Create an HTML-based MSA viewer with coloring.

    Args:
        alignment: List of aligned sequences.
        names: List of sequence names.
        width: Widget width.
        height: Widget height.
        max_sequences: Limit sequences for performance.

    Returns:
        HTML string.
    """
    # Truncate if too many
    if len(alignment) > max_sequences:
        alignment = alignment[:max_sequences]
        names = names[:max_sequences]
    
    # Check length consistency
    seq_len = len(alignment[0])
    
    # Basic Clustal-like colors
    colors = {
        'G': '#f79d5c', 'P': '#f79d5c', 'S': '#f79d5c', 'T': '#f79d5c',
        'C': '#f2d388', 'A': '#95a5a6', 'V': '#95a5a6', 'L': '#95a5a6', 'I': '#95a5a6', 'M': '#95a5a6',
        'F': '#81ecec', 'W': '#81ecec', 'Y': '#81ecec',
        'N': '#a29bfe', 'Q': '#a29bfe', 'H': '#a29bfe',
        'D': '#ff7675', 'E': '#ff7675',
        'K': '#74b9ff', 'R': '#74b9ff',
        '-': '#ffffff'
    }
    
    rows_html = []
    
    # Header row (ruler)
    ruler_cells = []
    for i in range(1, seq_len + 1):
        label = str(i) if i % 10 == 0 or i == 1 else ""
        ruler_cells.append(f'<div class="msa-ruler-cell">{label}</div>')
    
    rows_html.append(f"""
    <div class="msa-row">
        <div class="msa-name"></div>
        <div class="msa-seq">{''.join(ruler_cells)}</div>
    </div>
    """)
    
    for name, seq in zip(names, alignment):
        residues = []
        for char in seq:
            c = colors.get(char.upper(), '#ffffff')
            residues.append(f'<div class="msa-res" style="background-color: {c};">{char}</div>')
        
        rows_html.append(f"""
        <div class="msa-row">
            <div class="msa-name" title="{name}">{name[:15]}</div>
            <div class="msa-seq">{''.join(residues)}</div>
        </div>
        """)
    
    css = """
    <style>
    .msa-container {
        font-family: 'Courier New', monospace;
        overflow-x: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: white;
    }
    .msa-row {
        display: flex;
        height: 20px;
    }
    .msa-name {
        width: 150px;
        flex-shrink: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        border-right: 1px solid #eee;
        padding-left: 5px;
        font-size: 12px;
        line-height: 20px;
        background: #f9f9f9;
        position: sticky;
        left: 0;
    }
    .msa-seq {
        display: flex;
    }
    .msa-res {
        width: 12px;
        height: 20px;
        text-align: center;
        font-size: 12px;
        line-height: 20px;
        color: #333;
    }
    .msa-ruler-cell {
        width: 12px;
        height: 20px;
        font-size: 9px;
        color: #888;
        border-bottom: 1px solid #ccc;
    }
    </style>
    """
    
    html = f"""
    {css}
    <div class="msa-container" style="width: 100%; height: {height}px; overflow-y: auto;">
        {''.join(rows_html)}
    </div>
    """
    return html

