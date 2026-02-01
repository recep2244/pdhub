import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

# =============================================================================
# Visual Theme System
# =============================================================================

def apply_pro_theme(fig):
    """Apply the Cyber-Biology Pro theme to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#94a3b8"),
        title=dict(font=dict(size=20, weight="bold", color="#f1f5f9")),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig

# =============================================
# Analysis Plots
# =============================================

def create_pae_heatmap(
    pae_data: List[List[float]],
    title: str = "Predicted Aligned Error (PAE)",
    colorscale: Union[str, List] = "RdYlBu_r",
    max_value: float = 30.0,
) -> "plotly.graph_objects.Figure":
    """
    Create a PAE heatmap plot.

    Args:
        pae_data: 2D list/array of PAE values.
        title: Plot title.
        colorscale: Plotly colorscale or name.
        max_value: Max value for color scaling.

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

    return apply_pro_theme(fig)


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

    return apply_pro_theme(fig)


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
    width: str = "100%",
    height: int = 600,
    style: str = "cartoon",
    color_by: str = "spectrum",
    spin: bool = False,
    background_color: str = "#050508",
) -> str:
    """
    Create a 3D structure viewer using 3Dmol.js with Lab-OS Pro styling.
    """
    import uuid

    structure_path = Path(structure_path)
    model_data = structure_path.read_text()
    file_fmt = "mmcif" if structure_path.suffix.lower() in {".cif", ".mmcif"} else "pdb"
    viewer_id = f"mol_viewer_{uuid.uuid4().hex[:10]}"
    
    style_spec = {}
    if style == "cartoon":
        # Professional spectrum coloring for AlphaFold models
        if color_by == "spectrum":
            style_spec = {"cartoon": {"colorscheme": {"prop": "b", "gradient": "roygb", "min": 50, "max": 90}}}
        else:
            style_spec = {"cartoon": {"color": color_by}}
    elif style == "stick":
        style_spec = {"stick": {}}
    elif style == "sphere":
        style_spec = {"sphere": {}}
    
    # Clean up string for JS safety
    model_data = model_data.replace("`", "\`").replace("\\", "\\\\")

    html = f"""
    <div class="pro-viewer-container" style="width: {width}; height: {height}px; background: {background_color}; border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); overflow: hidden; position: relative;">
        <div id="{viewer_id}" style="width: 100%; height: 100%;"></div>
        <div style="position: absolute; bottom: 15px; right: 20px; font-family: 'JetBrains Mono'; font-size: 0.6rem; color: #475569; letter-spacing: 0.1em; pointer-events: none;">
            PDHUB-OS // 3D_VIEWPORT_01 // {structure_path.name.upper()}
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
        $(function() {{
            let element = $("#{viewer_id}");
            let config = {{ backgroundColor: "{background_color}" }};
            let viewer = $3Dmol.createViewer(element, config);
            
            viewer.addModel(`{model_data}`, "{file_fmt}");
            viewer.setStyle({{}}, {json.dumps(style_spec)});
            viewer.zoomTo();
            viewer.render();
            {'viewer.spin("y", 0.5);' if spin else ''}
            
            // Interaction glow
            element.on('mousedown', () => {{ element.css('box-shadow', 'inset 0 0 40px rgba(99, 102, 241, 0.1)'); }});
            element.on('mouseup', () => {{ element.css('box-shadow', 'none'); }});
        }});
    </script>
    """
    return html


def create_structure_comparison_3d(
    model_path: Path,
    reference_path: Optional[Path] = None,
    highlight_differences: bool = True,
    rmsd_threshold: float = 2.0,
    height: int = 500,
) -> str:
    """
    Create a 3D structural comparison viewer with synchronized Lab-OS Pro styling.
    """
    import uuid
    unique_id = f"structural_cmp_{uuid.uuid4().hex[:8]}"
    model_path = Path(model_path)
    model_fmt = "mmcif" if model_path.suffix.lower() in {".cif", ".mmcif"} else "pdb"
    model_pdb = model_path.read_text().replace("`", "\`").replace("\\", "\\\\")
    
    ref_html = (
        "<div style='font-family: \"JetBrains Mono\"; font-size: 0.65rem; color: #94a3b8; "
        "background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 4px; "
        "border: 1px solid rgba(255,255,255,0.1);'>REFERENCE ACTIVE</div>"
    ) if reference_path else ""

    html = f"""
    <div class="pro-viewer-container" style="width: 100%; height: {height}px; background: #050508; border-radius: 24px; border: 1px solid rgba(255,255,255,0.05); overflow: hidden; position: relative;">
        <div id="{unique_id}" style="width: 100%; height: 100%;"></div>
        <div style="position: absolute; top: 20px; right: 20px; display: flex; flex-direction: column; gap: 8px; align-items: flex-end; pointer-events: none;">
            <div style="font-family: 'JetBrains Mono'; font-size: 0.65rem; color: #6366f1; background: rgba(99, 102, 241, 0.1); padding: 4px 10px; border-radius: 4px; border: 1px solid rgba(99, 102, 241, 0.2);">
                MODEL: {model_path.name[:20]}
            </div>
            {ref_html}
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
    $(function() {{
        let viewer = $3Dmol.createViewer("{unique_id}", {{ backgroundColor: '#050508' }});
        
        // Add model with spectrum coloring
        let model = viewer.addModel(`{model_pdb}`, "{model_fmt}");
        model.setStyle({{}}, {{cartoon: {{colorscheme: {{prop: 'b', gradient: 'roygb', min: 50, max: 90}}}}}});
    """

    if reference_path:
        reference_path = Path(reference_path)
        ref_fmt = "mmcif" if reference_path.suffix.lower() in {".cif", ".mmcif"} else "pdb"
        ref_pdb = reference_path.read_text().replace("`", "\`").replace("\\", "\\\\")
        html += f"""
        let reference = viewer.addModel(`{ref_pdb}`, "{ref_fmt}");
        reference.setStyle({{}}, {{cartoon: {{color: '#475569', opacity: 0.6}}}});
        """

    html += """
        viewer.zoomTo();
        viewer.render();
        
        // Interaction micro-animations
        $("#" + "{unique_id}").on('mousedown', () => {{ $("#" + "{unique_id}").parent().css('border-color', 'rgba(99, 102, 241, 0.3)'); }});
        $("#" + "{unique_id}").on('mouseup', () => {{ $("#" + "{unique_id}").parent().css('border-color', 'rgba(255,255,255,0.05)'); }});
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
        "cmd.bg_color('black')",
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
        "set bgColor black",
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
        bgcolor="#111827",
        bordercolor="#334155",
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
        '-': '#1f2430'
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
            c = colors.get(char.upper(), '#1f2430')
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
        border: 1px solid #2a3342;
        border-radius: 4px;
        background: #0f141d;
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
        border-right: 1px solid #2a3342;
        padding-left: 5px;
        font-size: 12px;
        line-height: 20px;
        background: #141b26;
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
        color: #cbd5e1;
    }
    .msa-ruler-cell {
        width: 12px;
        height: 20px;
        font-size: 9px;
        color: #94a3b8;
        border-bottom: 1px solid #2a3342;
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


def create_plddt_sequence_viewer(
    sequence: str,
    plddt_values: Optional[List[float]] = None,
    label: str = "Sequence",
    show_ruler: bool = True,
) -> str:
    """
    Create an interactive pLDDT-colored sequence viewer similar to Alpha&ESM hFolds.

    Args:
        sequence: Amino acid sequence string.
        plddt_values: Per-residue pLDDT scores (0-100). If None, uses uniform coloring.
        label: Label for the sequence row.
        show_ruler: Whether to show position ruler.

    Returns:
        HTML string for the sequence viewer.
    """
    def get_plddt_color(score: float) -> str:
        """Get color based on pLDDT confidence score."""
        if score >= 90:
            return "#0053d6"  # Very high - dark blue
        elif score >= 70:
            return "#65cbf3"  # High - light blue
        elif score >= 50:
            return "#ffdb13"  # Low - yellow
        else:
            return "#ff7d45"  # Very low - orange

    seq_len = len(sequence)

    # Default pLDDT if not provided
    if plddt_values is None:
        plddt_values = [85.0] * seq_len  # Default confident

    # Build residue cells
    residue_cells = []
    for i, (aa, plddt) in enumerate(zip(sequence, plddt_values)):
        color = get_plddt_color(plddt)
        residue_cells.append(
            f'<div class="plddt-res" style="background-color: {color};" '
            f'title="Pos {i+1}: {aa} (pLDDT: {plddt:.1f})">{aa}</div>'
        )

    # Ruler
    ruler_html = ""
    if show_ruler:
        ruler_cells = []
        for i in range(1, seq_len + 1):
            if i == 1 or i % 10 == 0:
                ruler_cells.append(f'<div class="plddt-ruler-cell">{i}</div>')
            else:
                ruler_cells.append('<div class="plddt-ruler-cell"></div>')
        ruler_html = f'''
        <div class="plddt-row">
            <div class="plddt-label"></div>
            <div class="plddt-seq">{" ".join(ruler_cells)}</div>
        </div>
        '''

    css = """
    <style>
    .plddt-viewer {
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        background: #1a1f2e;
        border-radius: 12px;
        padding: 16px;
        overflow-x: auto;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .plddt-legend {
        display: flex;
        gap: 16px;
        margin-bottom: 12px;
        font-size: 12px;
        color: #94a3b8;
        flex-wrap: wrap;
    }
    .plddt-legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .plddt-legend-color {
        width: 16px;
        height: 16px;
        border-radius: 4px;
    }
    .plddt-row {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .plddt-label {
        width: 100px;
        flex-shrink: 0;
        font-size: 13px;
        font-weight: 600;
        color: #e2e8f0;
        padding-right: 12px;
    }
    .plddt-seq {
        display: flex;
        gap: 1px;
    }
    .plddt-res {
        width: 18px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 500;
        color: #e5e7eb;
        border-radius: 3px;
        cursor: pointer;
        transition: transform 0.1s, box-shadow 0.1s;
    }
    .plddt-res:hover {
        transform: scale(1.2);
        box-shadow: 0 0 8px rgba(229,231,235,0.35);
        z-index: 10;
        position: relative;
    }
    .plddt-ruler-cell {
        width: 18px;
        height: 16px;
        font-size: 9px;
        color: #64748b;
        text-align: center;
    }
    </style>
    """

    html = f"""
    {css}
    <div class="plddt-viewer">
        <div class="plddt-legend">
            <span style="color: #e2e8f0; font-weight: 600;">Model Confidence:</span>
            <div class="plddt-legend-item">
                <div class="plddt-legend-color" style="background: #0053d6;"></div>
                <span>Very high (pLDDT ≥ 90)</span>
            </div>
            <div class="plddt-legend-item">
                <div class="plddt-legend-color" style="background: #65cbf3;"></div>
                <span>High (90 > pLDDT ≥ 70)</span>
            </div>
            <div class="plddt-legend-item">
                <div class="plddt-legend-color" style="background: #ffdb13;"></div>
                <span>Low (70 > pLDDT ≥ 50)</span>
            </div>
            <div class="plddt-legend-item">
                <div class="plddt-legend-color" style="background: #ff7d45;"></div>
                <span>Very low (pLDDT < 50)</span>
            </div>
        </div>
        {ruler_html}
        <div class="plddt-row">
            <div class="plddt-label">{label}</div>
            <div class="plddt-seq">{''.join(residue_cells)}</div>
        </div>
    </div>
    """
    return html


def create_protein_info_table(
    protein_name: str,
    sequence: str,
    gene_name: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    pdb_id: Optional[str] = None,
    mean_plddt: Optional[float] = None,
    predictor: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a clean protein information table similar to Alpha&ESM hFolds.

    Args:
        protein_name: Name of the protein.
        sequence: Amino acid sequence.
        gene_name: Gene name.
        uniprot_id: UniProt accession.
        pdb_id: PDB ID if from experimental structure.
        mean_plddt: Mean pLDDT score.
        predictor: Name of the predictor used.
        additional_info: Additional key-value pairs to display.

    Returns:
        HTML string for the information table.
    """
    rows = []

    # Always show protein name
    rows.append(("Protein name", protein_name))

    if gene_name:
        rows.append(("Gene name", gene_name))

    if uniprot_id:
        link = f'<a href="https://www.uniprot.org/uniprotkb/{uniprot_id}" target="_blank" style="color: #60a5fa;">{uniprot_id}</a>'
        rows.append(("UniProt accession", link))

    rows.append(("Sequence length", str(len(sequence))))

    if pdb_id:
        link = f'<a href="https://www.rcsb.org/structure/{pdb_id}" target="_blank" style="color: #60a5fa;">{pdb_id}</a>'
        rows.append(("PDB ID", link))

    if mean_plddt is not None:
        color = "#0053d6" if mean_plddt >= 90 else "#65cbf3" if mean_plddt >= 70 else "#ffdb13" if mean_plddt >= 50 else "#ff7d45"
        rows.append(("Mean pLDDT", f'<span style="color: {color}; font-weight: 600;">{mean_plddt:.1f}</span>'))

    if predictor:
        rows.append(("Predictor", predictor))

    # Additional info
    if additional_info:
        for key, value in additional_info.items():
            rows.append((key, str(value)))

    # Build table HTML
    table_rows = ""
    for i, (label, value) in enumerate(rows):
        bg = "rgba(255,255,255,0.02)" if i % 2 == 0 else "transparent"
        table_rows += f'''
        <tr style="background: {bg};">
            <td style="padding: 10px 16px; font-weight: 500; color: #94a3b8; border-bottom: 1px solid rgba(255,255,255,0.05);">{label}</td>
            <td style="padding: 10px 16px; color: #e2e8f0; border-bottom: 1px solid rgba(255,255,255,0.05);">{value}</td>
        </tr>
        '''

    html = f"""
    <div style="background: #1a1f2e; border-radius: 12px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.1);">
            <h3 style="margin: 0; color: #e2e8f0; font-size: 16px; font-weight: 600;">📋 Protein Information</h3>
        </div>
        <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
            {table_rows}
        </table>
    </div>
    """
    return html


def create_expandable_section(
    title: str,
    content: str,
    icon: str = "📊",
    expanded: bool = False,
    section_id: Optional[str] = None,
) -> str:
    """
    Create an expandable/collapsible section similar to Alpha&ESM hFolds.

    Args:
        title: Section title.
        content: HTML content for the section.
        icon: Emoji icon for the title.
        expanded: Whether section starts expanded.
        section_id: Unique ID for the section (auto-generated if not provided).

    Returns:
        HTML string for the expandable section.
    """
    import uuid
    section_id = section_id or f"section_{uuid.uuid4().hex[:8]}"
    display = "block" if expanded else "none"
    chevron = "▼" if expanded else "▶"

    html = f"""
    <div style="background: #1a1f2e; border-radius: 12px; margin-bottom: 12px; border: 1px solid rgba(255,255,255,0.1); overflow: hidden;">
        <div
            onclick="
                var content = document.getElementById('{section_id}_content');
                var chevron = document.getElementById('{section_id}_chevron');
                if (content.style.display === 'none') {{
                    content.style.display = 'block';
                    chevron.textContent = '▼';
                }} else {{
                    content.style.display = 'none';
                    chevron.textContent = '▶';
                }}
            "
            style="display: flex; justify-content: space-between; align-items: center; padding: 14px 20px; cursor: pointer; background: rgba(255,255,255,0.03); border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.2s;"
            onmouseover="this.style.background='rgba(255,255,255,0.05)'"
            onmouseout="this.style.background='rgba(255,255,255,0.03)'"
        >
            <span style="color: #e2e8f0; font-weight: 600; font-size: 15px;">{icon} {title}</span>
            <span id="{section_id}_chevron" style="color: #64748b; font-size: 12px;">{chevron}</span>
        </div>
        <div id="{section_id}_content" style="display: {display}; padding: 16px 20px;">
            {content}
        </div>
    </div>
    """
    return html


def create_model_quality_summary(
    mean_plddt: float,
    ptm: Optional[float] = None,
    iptm: Optional[float] = None,
    clash_score: Optional[float] = None,
    ramachandran_favored: Optional[float] = None,
) -> str:
    """
    Create a model quality assessment summary panel.

    Args:
        mean_plddt: Mean pLDDT score.
        ptm: Predicted TM-score.
        iptm: Interface pTM (for multimers).
        clash_score: Clash score from evaluation.
        ramachandran_favored: Percentage of residues in favored regions.

    Returns:
        HTML string for the quality summary.
    """
    def quality_badge(value: float, thresholds: Tuple[float, float, float], labels: Tuple[str, str, str, str] = ("Excellent", "Good", "Fair", "Poor")) -> str:
        if value >= thresholds[0]:
            return f'<span style="background: #059669; color: #e5e7eb; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{labels[0]}</span>'
        elif value >= thresholds[1]:
            return f'<span style="background: #0284c7; color: #e5e7eb; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{labels[1]}</span>'
        elif value >= thresholds[2]:
            return f'<span style="background: #d97706; color: #e5e7eb; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{labels[2]}</span>'
        else:
            return f'<span style="background: #dc2626; color: #e5e7eb; padding: 2px 8px; border-radius: 4px; font-size: 11px;">{labels[3]}</span>'

    metrics = []

    # pLDDT
    badge = quality_badge(mean_plddt, (90, 70, 50))
    metrics.append(f'''
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
            <span style="color: #94a3b8;">Mean pLDDT</span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="color: #e2e8f0; font-weight: 600;">{mean_plddt:.1f}</span>
                {badge}
            </div>
        </div>
    ''')

    if ptm is not None:
        badge = quality_badge(ptm * 100, (80, 60, 40))
        metrics.append(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #94a3b8;">pTM Score</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #e2e8f0; font-weight: 600;">{ptm:.3f}</span>
                    {badge}
                </div>
            </div>
        ''')

    if iptm is not None:
        badge = quality_badge(iptm * 100, (80, 60, 40))
        metrics.append(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #94a3b8;">ipTM Score</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #e2e8f0; font-weight: 600;">{iptm:.3f}</span>
                    {badge}
                </div>
            </div>
        ''')

    if clash_score is not None:
        # Lower is better for clash score
        badge = quality_badge(100 - clash_score, (95, 85, 70))
        metrics.append(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);">
                <span style="color: #94a3b8;">Clash Score</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #e2e8f0; font-weight: 600;">{clash_score:.1f}</span>
                    {badge}
                </div>
            </div>
        ''')

    if ramachandran_favored is not None:
        badge = quality_badge(ramachandran_favored, (98, 95, 90))
        metrics.append(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 8px 0;">
                <span style="color: #94a3b8;">Ramachandran Favored</span>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: #e2e8f0; font-weight: 600;">{ramachandran_favored:.1f}%</span>
                    {badge}
                </div>
            </div>
        ''')

    html = f'''
    <div style="font-size: 14px;">
        {''.join(metrics)}
    </div>
    '''
    return html
