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
    color_by: str = "plddt",
    spin: bool = False,
    background_color: str = "#050508",
    show_toolbar: bool = True,
    show_surface: bool = False,
    surface_opacity: float = 0.7,
    title: str = "",
    highlight_residues: "Optional[List[int]]" = None,
    mutation_label: str = "",
    score_overlay: "Optional[Dict[str, str]]" = None,
) -> str:
    """
    Create an enhanced 3D structure viewer using 3Dmol.js.

    Features a full interactive toolbar (style/color/surface/spin/screenshot),
    pLDDT confidence coloring, chain selection, residue hover info, and
    secondary-structure interpretation overlay.
    """
    import uuid

    structure_path = Path(structure_path)
    model_data = structure_path.read_text()
    file_fmt = "mmcif" if structure_path.suffix.lower() in {".cif", ".mmcif"} else "pdb"
    vid = f"mv_{uuid.uuid4().hex[:10]}"
    display_name = title or structure_path.name

    # Escape for JS template literal
    model_data_js = model_data.replace("\\", "\\\\").replace("`", "\\`")

    # Build score overlay HTML
    _score_overlay_html = ""
    if score_overlay or mutation_label:
        _lines = []
        if mutation_label:
            _lines.append(
                f'<div style="color:#f59e0b;font-weight:700;font-size:12px;margin-bottom:4px;'
                f'letter-spacing:0.06em;">🔬 {mutation_label}</div>'
            )
        for _k, _v in (score_overlay or {}).items():
            _lines.append(
                f'<div><span style="color:#94a3b8;">{_k}:</span>'
                f' <span style="color:#e2e8f0;font-weight:600;">{_v}</span></div>'
            )
        _score_overlay_html = (
            '<div style="position:absolute;top:46px;right:14px;'
            'background:rgba(5,5,8,0.88);color:#e2e8f0;padding:8px 12px;'
            'border-radius:8px;font-size:11px;font-family:\'IBM Plex Mono\',monospace;'
            'z-index:100;line-height:1.75;border:1px solid rgba(255,255,255,0.1);'
            'pointer-events:none;">'
            + "".join(_lines)
            + "</div>"
        )

    # Build JS for highlighting specific residues
    _highlight_js = ""
    if highlight_residues:
        _resi_js = "[" + ",".join(str(r) for r in highlight_residues) + "]"
        _label_js = ""
        if mutation_label:
            _escaped_label = mutation_label.replace("'", "\\'")
            _label_js = (
                f"viewer.addLabel('{_escaped_label}', {{"
                f"resi: {highlight_residues[0]}, "
                f"backgroundOpacity: 0.85, backgroundColor: '#f59e0b', "
                f"fontColor: '#000', fontSize: 11, fontStyle: 'bold'"
                f"}});"
            )
        _highlight_js = f"""
        // Highlight mutant residues
        var hlResi = {_resi_js};
        if (hlResi.length > 0) {{
            viewer.addStyle({{resi: hlResi}}, {{stick: {{color: '#f59e0b', radius: 0.28}}}});
            viewer.addStyle({{resi: hlResi}}, {{sphere: {{color: '#f59e0b', radius: 0.45}}}});
            {_label_js}
        }}
        viewer.render();"""

    toolbar_html = ""
    if show_toolbar:
        toolbar_html = f"""
        <div id="{vid}_toolbar" style="
            position: absolute; top: 0; left: 0; right: 0;
            background: linear-gradient(180deg, rgba(5,5,8,0.95) 0%, rgba(5,5,8,0.0) 100%);
            padding: 10px 14px 20px;
            display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
            z-index: 10; pointer-events: auto;
        ">
            <!-- Style buttons -->
            <div style="display:flex;gap:4px;background:rgba(255,255,255,0.05);border-radius:8px;padding:3px;">
                <button onclick="setMolStyle_{vid}('cartoon')" id="{vid}_btn_cartoon"
                    style="background:rgba(63, 224, 197,0.3);color:#bfeee4;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Cartoon ribbon">Cartoon</button>
                <button onclick="setMolStyle_{vid}('surface')" id="{vid}_btn_surface"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Molecular surface">Surface</button>
                <button onclick="setMolStyle_{vid}('stick')" id="{vid}_btn_stick"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Stick representation">Sticks</button>
                <button onclick="setMolStyle_{vid}('sphere')" id="{vid}_btn_sphere"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Spacefill">Sphere</button>
                <button onclick="setMolStyle_{vid}('ribbon')" id="{vid}_btn_ribbon"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Ribbon">Ribbon</button>
            </div>
            <!-- Color buttons -->
            <div style="display:flex;gap:4px;background:rgba(255,255,255,0.05);border-radius:8px;padding:3px;">
                <button onclick="setMolColor_{vid}('plddt')" id="{vid}_clr_plddt"
                    style="background:rgba(34,197,94,0.2);color:#86efac;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Color by pLDDT confidence">pLDDT</button>
                <button onclick="setMolColor_{vid}('spectrum')" id="{vid}_clr_spectrum"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Rainbow N→C">Rainbow</button>
                <button onclick="setMolColor_{vid}('chain')" id="{vid}_clr_chain"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="By chain">Chain</button>
                <button onclick="setMolColor_{vid}('ss')" id="{vid}_clr_ss"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Secondary structure">SS</button>
            </div>
            <!-- Action buttons -->
            <div style="display:flex;gap:4px;margin-left:auto;">
                <button onclick="toggleSpin_{vid}()" id="{vid}_btn_spin"
                    style="background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);
                    border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Toggle spin">⟳ Spin</button>
                <button onclick="resetView_{vid}()"
                    style="background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);
                    border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Reset view">⊙ Reset</button>
                <button onclick="screenshot_{vid}()"
                    style="background:rgba(255,255,255,0.05);color:#94a3b8;border:1px solid rgba(255,255,255,0.1);
                    border-radius:6px;padding:3px 10px;font-size:11px;cursor:pointer;font-family:sans-serif;" title="Save PNG">📸</button>
            </div>
        </div>
        <!-- Hover info overlay -->
        <div id="{vid}_info" style="
            position: absolute; bottom: 14px; left: 14px;
            background: rgba(5,5,8,0.85); color: #94a3b8;
            font-family: 'IBM Plex Mono', monospace; font-size: 10px;
            padding: 4px 10px; border-radius: 6px;
            border: 1px solid rgba(255,255,255,0.08);
            pointer-events: none; display: none;
        "></div>
        <!-- File label -->
        <div style="
            position: absolute; bottom: 14px; right: 14px;
            font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem;
            color: #334155; letter-spacing: 0.08em; pointer-events: none;
        ">{display_name[:32].upper()}</div>
        {_score_overlay_html}
        """

    html = f"""
    <div style="position: relative; width: {width}; height: {height}px; background: {background_color};
        border-radius: 16px; border: 1px solid rgba(255,255,255,0.07); overflow: hidden;">
        <div id="{vid}" style="width: 100%; height: 100%;"></div>
        {toolbar_html}
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        var molData = `{model_data_js}`;
        var currentStyle = "cartoon";
        var currentColor = "plddt";
        var spinning = {'true' if spin else 'false'};
        var surfaceObj = null;

        // Defer init until the container has real dimensions (WebGL needs non-zero size)
        function _init_{vid}() {{
            var viewerEl = document.getElementById("{vid}");
            if (!viewerEl || viewerEl.offsetWidth === 0 || viewerEl.offsetHeight === 0) {{
                setTimeout(_init_{vid}, 80);
                return;
            }}

            var viewer = $3Dmol.createViewer(viewerEl, {{ backgroundColor: "{background_color}" }});
            viewer.addModel(molData, "{file_fmt}");

            function applyStyle_{vid}() {{
                viewer.setStyle({{}}, {{}});
                if (surfaceObj) {{ try {{ viewer.removeAllSurfaces(); }} catch(e) {{}} surfaceObj = null; }}

                var colorSpec = getColorSpec_{vid}(currentColor, currentStyle);

                if (currentStyle === "cartoon") {{
                    viewer.setStyle({{}}, {{cartoon: colorSpec}});
                }} else if (currentStyle === "surface") {{
                    viewer.setStyle({{}}, {{cartoon: {{opacity: 0.15, color: '#475569'}}}});
                    surfaceObj = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                        opacity: {surface_opacity},
                        colorscheme: colorSpec.colorscheme || {{prop: 'b', gradient: 'roygb', min: 50, max: 90}},
                    }});
                }} else if (currentStyle === "stick") {{
                    viewer.setStyle({{}}, {{stick: colorSpec}});
                }} else if (currentStyle === "sphere") {{
                    viewer.setStyle({{}}, {{sphere: colorSpec}});
                }} else if (currentStyle === "ribbon") {{
                    viewer.setStyle({{}}, {{ribbon: colorSpec}});
                }}
                viewer.render();
            }}

            function getColorSpec_{vid}(color, style) {{
                if (color === "plddt") {{
                    return {{colorscheme: {{prop: 'b', gradient: 'roygb', min: 50, max: 90}}}};
                }} else if (color === "spectrum") {{
                    return {{colorscheme: 'spectrum'}};
                }} else if (color === "chain") {{
                    return {{colorscheme: 'chain'}};
                }} else if (color === "ss") {{
                    return {{colorscheme: {{helix: 0xff6699, sheet: 0x6699ff, loop: 0x99cc99}}}};
                }}
                return {{colorscheme: 'spectrum'}};
            }}

            window["setMolStyle_{vid}"] = function(s) {{
                currentStyle = s;
                ['cartoon','surface','stick','sphere','ribbon'].forEach(function(n) {{
                    var btn = document.getElementById("{vid}_btn_" + n);
                    if (btn) {{
                        btn.style.background = (n === s) ? 'rgba(63, 224, 197,0.3)' : 'transparent';
                        btn.style.color = (n === s) ? '#bfeee4' : '#94a3b8';
                    }}
                }});
                applyStyle_{vid}();
            }};

            window["setMolColor_{vid}"] = function(c) {{
                currentColor = c;
                ['plddt','spectrum','chain','ss'].forEach(function(n) {{
                    var btn = document.getElementById("{vid}_clr_" + n);
                    if (btn) {{
                        btn.style.background = (n === c) ? 'rgba(34,197,94,0.2)' : 'transparent';
                        btn.style.color = (n === c) ? '#86efac' : '#94a3b8';
                    }}
                }});
                applyStyle_{vid}();
            }};

            window["toggleSpin_{vid}"] = function() {{
                spinning = !spinning;
                var btn = document.getElementById("{vid}_btn_spin");
                if (spinning) {{
                    viewer.spin('y', 0.5);
                    if (btn) {{ btn.style.color = '#3fe0c5'; btn.style.borderColor = 'rgba(63, 224, 197,0.4)'; }}
                }} else {{
                    viewer.spin(false);
                    if (btn) {{ btn.style.color = '#94a3b8'; btn.style.borderColor = 'rgba(255,255,255,0.1)'; }}
                }}
            }};

            window["resetView_{vid}"] = function() {{
                viewer.zoomTo();
                viewer.render();
            }};

            window["screenshot_{vid}"] = function() {{
                var png = viewer.pngURI();
                var a = document.createElement('a');
                a.href = png;
                a.download = '{display_name.replace(" ", "_")}.png';
                a.click();
            }};

            viewer.setHoverable({{}}, true,
                function(atom, v, event, container) {{
                    var info = document.getElementById("{vid}_info");
                    if (info && atom) {{
                        info.style.display = 'block';
                        info.textContent = (atom.chain ? 'Chain ' + atom.chain + '  ' : '') +
                            (atom.resn || '') + ' ' + (atom.resi || '') +
                            (atom.atom ? '  [' + atom.atom + ']' : '') +
                            (atom.b !== undefined ? '  pLDDT ' + atom.b.toFixed(1) : '');
                    }}
                }},
                function(atom, v, event, container) {{
                    var info = document.getElementById("{vid}_info");
                    if (info) info.style.display = 'none';
                }}
            );

            applyStyle_{vid}();
            viewer.zoomTo();
            viewer.render();
            if (spinning) viewer.spin('y', 0.5);
            {_highlight_js}
        }}

        // Start trying once scripts are ready; retry if element not yet sized
        setTimeout(_init_{vid}, 50);
    }})();
    </script>
    """
    return html


def create_structure_viewer_with_interpretation(
    structure_path: Path,
    plddt_values: Optional[List[float]] = None,
    sequence: Optional[str] = None,
    height: int = 500,
    title: str = "",
    extra_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Full structural interpretation panel: 3D viewer + pLDDT sequence strip +
    quality summary + secondary-structure breakdown + downloadable PyMOL script.

    Suitable as a drop-in replacement for ``create_structure_viewer`` on any
    result page that should show both the structure *and* explain what it means.
    """
    import uuid

    structure_path = Path(structure_path)
    display_name = title or structure_path.name
    panel_id = uuid.uuid4().hex[:8]

    # --- 3D viewer HTML (compact) ---
    viewer_html = create_structure_viewer(
        structure_path,
        height=height,
        show_toolbar=True,
        title=display_name,
    )

    # --- pLDDT sequence strip (if available) ---
    seq_strip_html = ""
    if sequence and plddt_values:
        seq_strip_html = create_plddt_sequence_viewer(
            sequence, plddt_values, label=display_name[:16]
        )

    # --- Quality interpretation card ---
    quality_html = ""
    if plddt_values:
        mean_pl = float(np.mean(plddt_values))
        high_frac = sum(1 for p in plddt_values if p >= 70) / max(len(plddt_values), 1) * 100
        very_high_frac = sum(1 for p in plddt_values if p >= 90) / max(len(plddt_values), 1) * 100
        low_frac = sum(1 for p in plddt_values if p < 50) / max(len(plddt_values), 1) * 100

        if mean_pl >= 85:
            verdict_color = "#22c55e"; verdict = "High-confidence model"
        elif mean_pl >= 70:
            verdict_color = "#60a5fa"; verdict = "Reasonably confident model"
        elif mean_pl >= 50:
            verdict_color = "#f59e0b"; verdict = "Low-confidence — interpret cautiously"
        else:
            verdict_color = "#ef4444"; verdict = "Very low confidence — likely disordered"

        metric_html = ""
        if extra_metrics:
            for k, v in extra_metrics.items():
                metric_html += f"""
                <div style="display:flex;justify-content:space-between;padding:5px 0;
                    border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;">
                    <span style="color:#94a3b8;">{k}</span>
                    <span style="color:#e2e8f0;font-weight:600;">{v}</span>
                </div>"""

        quality_html = f"""
        <div style="background:#1a1f2e;border-radius:12px;padding:16px;
            border:1px solid rgba(255,255,255,0.08);margin-top:10px;">
            <div style="font-size:13px;font-weight:700;color:#e2e8f0;margin-bottom:10px;">
                📊 Structure Quality — {display_name[:30]}
            </div>
            <div style="margin-bottom:8px;">
                <span style="font-size:12px;font-weight:600;color:{verdict_color};
                    background:rgba(255,255,255,0.06);border-radius:6px;
                    padding:3px 10px;">{verdict}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;">
                <span style="color:#94a3b8;">Mean pLDDT</span>
                <span style="color:#e2e8f0;font-weight:600;">{mean_pl:.1f}</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;">
                <span style="color:#94a3b8;">Residues ≥70 (confident)</span>
                <span style="color:#60a5fa;font-weight:600;">{high_frac:.0f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;">
                <span style="color:#94a3b8;">Residues ≥90 (very high)</span>
                <span style="color:#22c55e;font-weight:600;">{very_high_frac:.0f}%</span>
            </div>
            <div style="display:flex;justify-content:space-between;padding:5px 0;
                border-bottom:1px solid rgba(255,255,255,0.05);font-size:13px;">
                <span style="color:#94a3b8;">Residues &lt;50 (very low)</span>
                <span style="color:#ef4444;font-weight:600;">{low_frac:.0f}%</span>
            </div>
            {metric_html}
            <div style="margin-top:10px;font-size:11px;color:#64748b;line-height:1.5;">
                <b style="color:#94a3b8;">AlphaFold pLDDT guide:</b>
                <span style="color:#0053d6;">■</span> ≥90 very high —
                <span style="color:#65cbf3;">■</span> 70–90 confident —
                <span style="color:#ffdb13;">■</span> 50–70 low —
                <span style="color:#ff7d45;">■</span> &lt;50 very low (likely disordered)
            </div>
        </div>
        """

    full_html = f"""
    <div id="interp_{panel_id}">
        {viewer_html}
        {seq_strip_html}
        {quality_html}
    </div>
    """
    return full_html


def create_structure_comparison_3d(
    model_path: Path,
    reference_path: Optional[Path] = None,
    highlight_differences: bool = True,
    rmsd_threshold: float = 2.0,
    height: int = 500,
    model_label: str = "",
    reference_label: str = "Reference",
) -> str:
    """
    Enhanced 3D structural comparison viewer.

    Shows model (pLDDT-colored) alongside an optional semi-transparent reference.
    Includes a full toolbar and a legend for the two structures.
    """
    import uuid
    vid = f"cmp_{uuid.uuid4().hex[:8]}"
    model_path = Path(model_path)
    model_fmt = "mmcif" if model_path.suffix.lower() in {".cif", ".mmcif"} else "pdb"
    model_pdb = model_path.read_text().replace("\\", "\\\\").replace("`", "\\`")
    model_name = model_label or model_path.name[:20]

    ref_legend = ""
    ref_js = ""
    if reference_path:
        reference_path = Path(reference_path)
        ref_fmt = "mmcif" if reference_path.suffix.lower() in {".cif", ".mmcif"} else "pdb"
        ref_pdb = reference_path.read_text().replace("\\", "\\\\").replace("`", "\\`")
        ref_name = reference_label or reference_path.name[:20]

        ref_legend = f"""
        <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#94a3b8;">
            <span style="width:12px;height:3px;background:#64748b;display:inline-block;border-radius:2px;"></span>
            {ref_name} (reference)
        </div>"""

        ref_js = f"""
        var refModel = viewer.addModel(`{ref_pdb}`, "{ref_fmt}");
        refModel.setStyle({{}}, {{cartoon: {{color: '#475569', opacity: 0.45}}}});
        """

    html = f"""
    <div style="position:relative;width:100%;height:{height}px;background:#050508;
        border-radius:16px;border:1px solid rgba(255,255,255,0.07);overflow:hidden;">
        <div id="{vid}" style="width:100%;height:100%;"></div>

        <!-- Toolbar -->
        <div style="position:absolute;top:0;left:0;right:0;
            background:linear-gradient(180deg,rgba(5,5,8,0.95) 0%,rgba(5,5,8,0) 100%);
            padding:10px 14px 20px;display:flex;align-items:center;gap:8px;z-index:100;">
            <div style="display:flex;gap:4px;background:rgba(255,255,255,0.05);border-radius:8px;padding:3px;">
                <button onclick="setCmpStyle_{vid}('cartoon')"
                    style="background:rgba(63, 224, 197,0.3);color:#bfeee4;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;" title="Cartoon">Cartoon</button>
                <button onclick="setCmpStyle_{vid}('stick')"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;" title="Sticks">Sticks</button>
                <button onclick="setCmpStyle_{vid}('sphere')"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;" title="Sphere">Sphere</button>
            </div>
            <div style="display:flex;gap:4px;background:rgba(255,255,255,0.05);border-radius:8px;padding:3px;">
                <button onclick="setCmpColor_{vid}('plddt')"
                    style="background:rgba(34,197,94,0.2);color:#86efac;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;">pLDDT</button>
                <button onclick="setCmpColor_{vid}('spectrum')"
                    style="background:transparent;color:#94a3b8;border:none;border-radius:6px;
                    padding:3px 10px;font-size:11px;cursor:pointer;">Rainbow</button>
            </div>
            <button onclick="resetCmp_{vid}()"
                style="margin-left:auto;background:rgba(255,255,255,0.05);color:#94a3b8;
                border:1px solid rgba(255,255,255,0.1);border-radius:6px;
                padding:3px 10px;font-size:11px;cursor:pointer;">⊙ Reset</button>
        </div>

        <!-- Legend -->
        <div style="position:absolute;bottom:14px;left:14px;
            display:flex;flex-direction:column;gap:5px;pointer-events:none;">
            <div style="display:flex;align-items:center;gap:6px;font-size:11px;color:#bfeee4;">
                <span style="width:12px;height:3px;background:#3fe0c5;display:inline-block;border-radius:2px;"></span>
                {model_name} (model)
            </div>
            {ref_legend}
        </div>

        <!-- Filename -->
        <div style="position:absolute;bottom:14px;right:14px;
            font-family:'IBM Plex Mono',monospace;font-size:0.58rem;
            color:#334155;letter-spacing:0.08em;pointer-events:none;">
            STRUCTURAL COMPARISON
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
    (function() {{
        var viewer = $3Dmol.createViewer("{vid}", {{backgroundColor: '#050508'}});
        var currentStyle = "cartoon";
        var currentColor = "plddt";

        var modelMol = viewer.addModel(`{model_pdb}`, "{model_fmt}");
        {ref_js}

        function applyStyle_{vid}() {{
            var cs = (currentColor === "plddt")
                ? {{colorscheme: {{prop: 'b', gradient: 'roygb', min: 50, max: 90}}}}
                : {{colorscheme: 'spectrum'}};
            var styleObj = {{}};
            styleObj[currentStyle] = cs;
            modelMol.setStyle({{}}, styleObj);
            viewer.render();
        }}

        window["setCmpStyle_{vid}"] = function(s) {{ currentStyle = s; applyStyle_{vid}(); }};
        window["setCmpColor_{vid}"] = function(c) {{ currentColor = c; applyStyle_{vid}(); }};
        window["resetCmp_{vid}"] = function() {{ viewer.zoomTo(); viewer.render(); }};

        applyStyle_{vid}();
        viewer.zoomTo();
        viewer.render();
    }})();
    </script>
    """
    return html



def _preload_system_libstdcxx() -> None:
    """Preload the system libstdc++ to fix GLIBCXX version mismatch.

    PyMOL pip-wheels are often compiled against a newer GLIBCXX than
    Anaconda's bundled libstdc++.  Loading the system copy via ctypes
    before importing pymol2 resolves the 'GLIBCXX_3.4.30 not found' error.
    """
    import ctypes, glob
    candidates = [
        "/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
        "/usr/lib64/libstdc++.so.6",
    ] + glob.glob("/usr/lib/gcc/x86_64-linux-gnu/*/libstdc++.so.6")
    for path in candidates:
        try:
            ctypes.CDLL(path)
            return
        except OSError:
            continue


def is_pymol_available() -> bool:
    """Check if pymol2 (headless PyMOL API) is importable."""
    _preload_system_libstdcxx()
    try:
        import pymol2  # noqa: F401
        return True
    except (ImportError, OSError):
        try:
            import pymol  # noqa: F401
            return True
        except (ImportError, OSError):
            return False


def render_pymol_headless(
    structures: List[Tuple[Path, str]],
    output_image: Path,
    superimpose: bool = True,
    width: int = 1200,
    height: int = 900,
    ray: bool = False,
    highlight_residues: "Optional[List[int]]" = None,
    restraint_residues: "Optional[List[int]]" = None,
    mutation_label: str = "",
    score_labels: "Optional[Dict[str, str]]" = None,
    dark_bg: bool = True,
) -> bool:
    """
    Render structures to a PNG using the pymol2 headless API.

    Requires ``pip install pymol-open-source`` or a licensed PyMOL.
    Falls back to False if PyMOL is not installed.

    Args:
        structures: List of (path, name) tuples.
        output_image: Output PNG path.
        superimpose: Align all structures to the first.
        width: Image width in pixels.
        height: Image height in pixels.
        ray: Use ray-tracing (slower but higher quality).
        highlight_residues: Mutation residue positions to highlight in amber.
        restraint_residues: Restraint/fixed residue positions to highlight in green.
        mutation_label: Label text to show on highlighted residues.
        score_labels: Dict of score_name -> value_str for text overlays.
        dark_bg: Use dark background matching app theme.

    Returns:
        True if rendering succeeded, False otherwise.
    """
    _preload_system_libstdcxx()
    try:
        import pymol2
    except (ImportError, OSError):
        return False

    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    with pymol2.PyMOL() as p:
        cmd = p.cmd

        # Visualization settings
        cmd.set("cartoon_fancy_helices", 1)
        cmd.set("cartoon_side_chain_helper", 1)
        cmd.set("ray_shadows", 0)
        cmd.set("antialias", 2)
        cmd.set("depth_cue", 0)
        if dark_bg:
            cmd.bg_color("black")
        else:
            cmd.bg_color("white")

        # Load structures
        colors = ["0x3b82f6", "0xef4444", "0x22c55e", "0xf97316", "0xa855f7", "0x06b6d4"]
        for i, (path, name) in enumerate(structures):
            cmd.load(str(path), name)
            cmd.color(colors[i % len(colors)], name)

        # Superimpose
        if superimpose and len(structures) > 1:
            reference = structures[0][1]
            for _, name in structures[1:]:
                cmd.align(name, reference)

        # Display
        cmd.show("cartoon", "all")
        cmd.hide("lines", "all")
        cmd.set("cartoon_transparency", 0.1)
        cmd.center("all")
        cmd.zoom("all", 2)

        # Highlight mutation residues in amber/orange
        if highlight_residues:
            resi_str = "+".join(str(r) for r in highlight_residues)
            cmd.select("mutations", f"resi {resi_str}")
            cmd.show("sticks", "mutations")
            cmd.color("0xf59e0b", "mutations")  # amber
            cmd.set("stick_radius", 0.25, "mutations")
            if mutation_label:
                cmd.label("mutations and name CA", f'"{mutation_label}"')
                cmd.set("label_color", "0xf59e0b")
                cmd.set("label_size", 14)
                cmd.set("label_font_id", 7)

        # Highlight restraint residues in green
        if restraint_residues:
            resi_str = "+".join(str(r) for r in restraint_residues)
            cmd.select("restraints", f"resi {resi_str}")
            cmd.show("sticks", "restraints")
            cmd.color("0x10b981", "restraints")  # emerald green
            cmd.set("stick_radius", 0.20, "restraints")

        # Render
        if ray:
            cmd.ray(width, height)
        cmd.png(str(output_image), width=width, height=height, dpi=150, quiet=1)

    return output_image.exists()


# Best available PyMOL GUI binary (checked once at import time)
def _find_pymol_binary() -> Optional[str]:
    """Find the best available PyMOL binary for launching the GUI."""
    import shutil, os
    candidates = []
    # Allow explicit override via environment variable
    env_override = os.environ.get("PYMOL_BINARY")
    if env_override:
        candidates.append(env_override)
    # PATH-based discovery (works across all systems)
    candidates.append(shutil.which("pymol"))
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def build_pymol_script(
    structures: List[Tuple[Path, str]],
    highlight_residues: "Optional[List[int]]" = None,
    restraint_residues: "Optional[List[int]]" = None,
    mutation_label: str = "",
    score_labels: "Optional[Dict[str, str]]" = None,
    superimpose: bool = True,
    dark_bg: bool = True,
) -> str:
    """Generate a PyMOL .pml script for interactive visualization.

    Works with PyMOL GUI and PyMOL AI. Highlights mutation residues in amber
    and restraint/fixed residues in green.
    """
    lines = [
        "# PyMOL visualization script — Protein Design Hub",
        f"# Mutation: {mutation_label}" if mutation_label else "# Structure viewer",
        "",
        f"bg_color {'black' if dark_bg else 'white'}",
        "set cartoon_fancy_helices, 1",
        "set cartoon_side_chain_helper, 1",
        "set depth_cue, 0",
        "set ray_shadows, 0",
        "set antialias, 2",
        "",
    ]

    colors = ["marine", "salmon", "forest", "orange", "magenta", "cyan"]
    for i, (path, name) in enumerate(structures):
        lines.append(f"load {path}, {name}")

    lines.append("")
    if superimpose and len(structures) > 1:
        ref = structures[0][1]
        for _, name in structures[1:]:
            lines.append(f"align {name}, {ref}")
        lines.append("")

    for i, (_, name) in enumerate(structures):
        lines.append(f"color {colors[i % len(colors)]}, {name}")

    lines.extend([
        "",
        "hide lines, all",
        "show cartoon, all",
        "set cartoon_transparency, 0.1",
        "",
    ])

    if highlight_residues:
        resi_str = "+".join(str(r) for r in highlight_residues)
        lines.extend([
            f"select mutations, resi {resi_str}",
            "show sticks, mutations",
            "color tv_orange, mutations",
            "set stick_radius, 0.25, mutations",
        ])
        if mutation_label:
            lines.extend([
                f'label mutations and name CA, "{mutation_label}"',
                "set label_color, tv_orange",
                "set label_size, 14",
                "set label_font_id, 7",
            ])
        lines.append("")

    if restraint_residues:
        resi_str = "+".join(str(r) for r in restraint_residues)
        lines.extend([
            f"select restraints, resi {resi_str}",
            "show sticks, restraints",
            "color green, restraints",
            "set stick_radius, 0.20, restraints",
            "",
        ])

    if score_labels:
        for k, v in score_labels.items():
            lines.append(f"# Score: {k} = {v}")
        lines.append("")

    lines.extend([
        "center all",
        "zoom all, 2",
        "deselect",
    ])

    return "\n".join(lines)


def launch_pymol_interactive(
    structures,
    highlight_residues: "Optional[List[int]]" = None,
    restraint_residues: "Optional[List[int]]" = None,
    mutation_label: str = "",
    score_labels: "Optional[Dict[str, str]]" = None,
    superimpose: bool = True,
) -> Optional[int]:
    """Launch PyMOL GUI interactively with structures and mutation/restraint highlights.

    Non-blocking — opens a separate PyMOL window and returns immediately.

    Args:
        structures: Path | str | List[Path | (Path, name)].
        highlight_residues: Mutation residue positions to highlight in amber.
        restraint_residues: Restraint/fixed residue positions to highlight in green.
        mutation_label: Label for mutation residues.
        score_labels: Dict of score_name -> value_str shown as comments.
        superimpose: Align structures if multiple.

    Returns:
        PID of the launched PyMOL process, or None if launch failed.
    """
    import subprocess, tempfile, time

    binary = _find_pymol_binary()
    if not binary:
        return None

    # Normalise structures list
    def _norm(p):
        if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[1], str):
            return (Path(p[0]), p[1])
        return (Path(p), Path(p).stem)

    if isinstance(structures, (str, Path)):
        structs = [_norm(structures)]
    elif isinstance(structures, list):
        structs = [_norm(s) for s in structures]
    else:
        structs = [_norm(structures)]
    structs = [(p, n) for p, n in structs if p.exists()]
    if not structs:
        return None

    script = build_pymol_script(
        structs,
        highlight_residues=highlight_residues,
        restraint_residues=restraint_residues,
        mutation_label=mutation_label,
        score_labels=score_labels,
        superimpose=superimpose,
    )

    # Write to a persistent temp file (PyMOL needs to read it after launch)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".pml", prefix="pdh_pymol_", delete=False
    )
    tmp.write(script)
    tmp.close()

    try:
        proc = subprocess.Popen(
            [binary, tmp.name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return proc.pid
    except Exception:
        return None


def export_pymol_session(
    structures: List[Tuple[Path, str]],
    output_path: Path,
    superimpose: bool = True,
) -> str:
    """
    Generate PyMOL script for structure comparison.

    When ``pymol2`` is installed this also attempts a headless render to
    ``output_path.with_suffix('.png')`` alongside the script.

    Args:
        structures: List of (path, name) tuples for structures.
        output_path: Path for PyMOL script.
        superimpose: Whether to superimpose structures.

    Returns:
        PyMOL script content.
    """
    script_lines = [
        "# PyMOL session generated by Protein Design Hub",
        "# Run with: pymol this_script.py  OR  python this_script.py",
        "from pymol import cmd",
        "",
        "# Set visualization defaults",
        "cmd.set('cartoon_fancy_helices', 1)",
        "cmd.set('cartoon_side_chain_helper', 1)",
        "cmd.bg_color('white')",
        "cmd.set('antialias', 2)",
        "cmd.set('ray_shadows', 0)",
        "",
    ]

    # Load structures
    colors = ["marine", "red", "forest", "orange", "magenta", "cyan"]
    for i, (path, name) in enumerate(structures):
        script_lines.append(f"cmd.load('{path}', '{name}')")

    # Superimpose if requested
    if superimpose and len(structures) > 1:
        script_lines.append("")
        script_lines.append("# Superimpose structures")
        reference = structures[0][1]
        for _, name in structures[1:]:
            script_lines.append(f"cmd.align('{name}', '{reference}')")

    # Coloring
    script_lines.append("")
    script_lines.append("# Apply coloring")
    for i, (_, name) in enumerate(structures):
        script_lines.append(f"cmd.color('{colors[i % len(colors)]}', '{name}')")

    # Final setup
    script_lines.extend([
        "",
        "# Show cartoon representation",
        "cmd.show('cartoon', 'all')",
        "cmd.hide('lines', 'all')",
        "cmd.center('all')",
        "cmd.zoom('all', 2)",
        "",
        "# Uncomment to save a PNG:",
        "# cmd.ray(1200, 900)",
        "# cmd.png('output.png', width=1200, height=900, dpi=150, quiet=1)",
    ])

    script = "\n".join(script_lines)

    # Save script
    output_path = Path(output_path)
    output_path.write_text(script)

    # Attempt headless render if pymol2 is available
    if is_pymol_available():
        png_path = output_path.with_suffix(".png")
        render_pymol_headless(structures, png_path, superimpose=superimpose)

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
    show_conservation_bar: bool = True,
) -> str:
    """
    Create an HTML-based MSA viewer with ClustalX-style coloring,
    per-column conservation bar, hover tooltips, and sticky name column.

    Args:
        alignment: List of aligned sequences (all same length).
        names: List of sequence names.
        width: Widget width in pixels.
        height: Widget height in pixels.
        max_sequences: Truncate at this many sequences for performance.
        show_conservation_bar: Show per-column conservation bar at the bottom.

    Returns:
        HTML string.
    """
    import html as _html_mod

    if not alignment:
        return "<p>No alignment data</p>"

    if len(alignment) > max_sequences:
        alignment = alignment[:max_sequences]
        names = names[:max_sequences]

    seq_len = len(alignment[0])
    n_seqs = len(alignment)

    # ── ClustalX-style residue colors ──────────────────────────────────────
    RESIDUE_COLOR = {
        'G': '#f79d5c', 'P': '#f79d5c', 'S': '#f5a623', 'T': '#f5a623',
        'C': '#f2d388',
        'A': '#7f8c8d', 'V': '#7f8c8d', 'L': '#7f8c8d', 'I': '#7f8c8d', 'M': '#7f8c8d',
        'F': '#81ecec', 'W': '#81ecec', 'Y': '#81ecec',
        'N': '#a29bfe', 'Q': '#a29bfe', 'H': '#a29bfe',
        'D': '#ff7675', 'E': '#ff7675',
        'K': '#74b9ff', 'R': '#74b9ff',
        '-': '#1a2030', '.': '#1a2030',
    }

    # ── Per-column conservation score (0–1) ────────────────────────────────
    cons_scores: List[float] = []
    for col in range(seq_len):
        counts: dict = {}
        for seq in alignment:
            aa = seq[col].upper() if col < len(seq) else '-'
            if aa not in ('-', '.'):
                counts[aa] = counts.get(aa, 0) + 1
        total = sum(counts.values())
        if total == 0:
            cons_scores.append(0.0)
        else:
            top = max(counts.values())
            cons_scores.append(top / total)

    # ── Ruler row ──────────────────────────────────────────────────────────
    ruler_cells = []
    for i in range(1, seq_len + 1):
        if i == 1 or i % 10 == 0:
            label = str(i)
            tick = 'border-top:2px solid #4a5568;'
        elif i % 5 == 0:
            label = ''
            tick = 'border-top:1px solid #3a4455;'
        else:
            label = ''
            tick = ''
        ruler_cells.append(
            f'<div class="msa-cell msa-ruler" style="{tick}" title="Position {i}">{label}</div>'
        )

    rows_html = [
        f'<div class="msa-row">'
        f'<div class="msa-label msa-ruler-label">Pos</div>'
        f'<div class="msa-seq">{"".join(ruler_cells)}</div>'
        f'</div>'
    ]

    # ── Sequence rows ───────────────────────────────────────────────────────
    ref_seq = alignment[0]
    for row_idx, (name, seq) in enumerate(zip(names, alignment)):
        # Identity to first sequence (excluding gaps in reference)
        matches = sum(
            1 for a, b in zip(ref_seq, seq)
            if a == b and a not in ('-', '.')
        )
        ref_len = sum(1 for c in ref_seq if c not in ('-', '.'))
        pct_id = f"{matches/ref_len*100:.0f}%" if ref_len > 0 and row_idx > 0 else "query"

        cells = []
        for col, char in enumerate(seq):
            bg = RESIDUE_COLOR.get(char.upper(), '#1a2030')
            tip = _html_mod.escape(f"{name} | pos {col+1} | {char}")
            # Dim residues that match reference exactly (except reference itself)
            opacity = '0.55' if (row_idx > 0 and char == ref_seq[col] and char not in ('-', '.')) else '1'
            cells.append(
                f'<div class="msa-cell msa-res" '
                f'style="background:{bg};opacity:{opacity}" title="{tip}">{char}</div>'
            )

        safe_name = _html_mod.escape(name)
        rows_html.append(
            f'<div class="msa-row">'
            f'<div class="msa-label" title="{safe_name}">'
            f'<span class="msa-name-text">{safe_name[:18]}</span>'
            f'<span class="msa-pct-id">{pct_id}</span>'
            f'</div>'
            f'<div class="msa-seq">{"".join(cells)}</div>'
            f'</div>'
        )

    # ── Conservation bar ────────────────────────────────────────────────────
    if show_conservation_bar:
        bar_cells = []
        for score in cons_scores:
            bar_h = max(2, int(score * 18))
            # Colour: low cons = dim grey, high cons = vivid teal
            r = int(32 + (0 - 32) * score)
            g = int(52 + (200 - 52) * score)
            b = int(80 + (180 - 80) * score)
            bg = f'rgb({r},{g},{b})'
            bar_cells.append(
                f'<div class="msa-cell msa-cons-cell" title="Conservation {score:.2f}">'
                f'<div style="height:{bar_h}px;background:{bg};margin-top:auto;border-radius:1px"></div>'
                f'</div>'
            )
        rows_html.append(
            f'<div class="msa-row msa-cons-row">'
            f'<div class="msa-label" style="font-size:9px;color:#64748b">Cons.</div>'
            f'<div class="msa-seq">{"".join(bar_cells)}</div>'
            f'</div>'
        )

    css = """<style>
    .msa-wrap { font-family:'Courier New',monospace; overflow-x:auto;
                border:1px solid #1e2a3a; border-radius:6px; background:#0b0f1a; }
    .msa-row  { display:flex; align-items:stretch; border-bottom:1px solid #111827; }
    .msa-row:last-child { border-bottom:none; }
    .msa-label {
        width:160px; flex-shrink:0; display:flex; align-items:center;
        justify-content:space-between; padding:0 6px;
        background:#111827; border-right:1px solid #1e2a3a;
        position:sticky; left:0; z-index:2;
    }
    .msa-name-text { font-size:11px; color:#cbd5e1; overflow:hidden;
                     text-overflow:ellipsis; white-space:nowrap; max-width:105px; }
    .msa-pct-id    { font-size:9px; color:#64748b; flex-shrink:0; margin-left:4px; }
    .msa-ruler-label { font-size:9px; color:#64748b; }
    .msa-seq  { display:flex; }
    .msa-cell { width:13px; height:20px; text-align:center; font-size:11px;
                line-height:20px; flex-shrink:0; cursor:default; }
    .msa-res  { color:#e2e8f0; font-weight:500; }
    .msa-res:hover { outline:1px solid #f8fafc; z-index:3; position:relative; }
    .msa-ruler { font-size:9px; color:#94a3b8; background:#0d1117; }
    .msa-cons-row { height:20px; }
    .msa-cons-cell { display:flex; flex-direction:column; justify-content:flex-end; }
    </style>"""

    return (
        f"{css}"
        f'<div class="msa-wrap" style="height:{height}px;overflow-y:auto;">'
        + "".join(rows_html)
        + "</div>"
    )


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
        font-family: 'IBM Plex Mono', 'Courier New', monospace;
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


# =============================================================================
# PyMOL API interactive viewer (embedded HTTP server) + fallbacks
# =============================================================================

def show_structure_with_pymol_fallback(
    structure_paths,
    title: str = "",
    height: int = 400,
    superimpose: bool = True,
    ray: bool = False,
    key: str = "struct",
    highlight_residues: "Optional[List[int]]" = None,
    restraint_residues: "Optional[List[int]]" = None,
    mutation_label: str = "",
    score_overlay: "Optional[Dict[str, str]]" = None,
) -> None:
    """Display structure(s) using an embedded interactive PyMOL API session.

    Primary flow (no WebGL required):
    1. Start singleton PyMOL HTTP server (port 8592) on first call
    2. Load structure into PyMOL with mutation/restraint highlights
    3. Embed interactive viewer via <iframe> — drag to rotate, scroll to zoom,
       toolbar for style/color/surface changes, keyboard shortcuts
    4. Offer "Open in PyMOL" button to launch full PyMOL GUI separately

    Falls back to 3Dmol.js iframe if PyMOL is unavailable.

    Args:
        structure_paths: One of:
            - A single ``Path``/``str``
            - A list of ``Path``/``str``
            - A list of ``(Path, name)`` tuples
        title: Caption shown above the viewer.
        height: Viewer height in pixels.
        superimpose: Align structures to first when using PyMOL.
        ray: Unused (kept for API compatibility).
        key: Unique Streamlit widget key prefix.
        highlight_residues: Mutation residue positions to highlight in amber.
        restraint_residues: Restraint/fixed residue positions to highlight in green.
        mutation_label: Label shown on the highlighted residue and overlay.
        score_overlay: Dict of score_name -> value_str displayed as an overlay panel.
    """
    import streamlit as st
    import streamlit.components.v1 as components

    from protein_design_hub.web.pymol_server import get_pymol_server

    # ---------- normalise input ----------
    def _to_tuple(p):
        if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[1], str):
            return (Path(p[0]), p[1])
        return (Path(p), Path(p).stem)

    if isinstance(structure_paths, (str, Path)):
        structs = [_to_tuple(structure_paths)]
    elif isinstance(structure_paths, list):
        structs = [_to_tuple(p) for p in structure_paths]
    else:
        structs = [_to_tuple(structure_paths)]

    structs = [(p, n) for p, n in structs if p.exists()]
    if not structs:
        st.warning("No structure files found.")
        return

    if title:
        st.caption(f"🔬 **{title}**")

    # ---------- Try PyMOL interactive server (primary, no WebGL needed) ----------
    server = get_pymol_server()

    if server is not None:
        pdb_data = structs[0][0].read_text()
        server.load_structure(
            pdb_data=pdb_data,
            name=structs[0][1],
            highlight_residues=highlight_residues,
            restraint_residues=restraint_residues,
            mutation_label=mutation_label,
        )

        # Build legend/score HTML for postMessage injection into the iframe
        _legend_parts = []
        if mutation_label:
            _legend_parts.append(
                f'<span class="leg-mut">&#9632; Mutations (amber): {mutation_label}</span>'
            )
        if restraint_residues:
            _legend_parts.append(
                f'<span class="leg-res">&#9632; Restraints (green): '
                f'{", ".join(str(r) for r in restraint_residues)}</span>'
            )
        _legend_html = "<br>".join(_legend_parts)

        _scores_html = ""
        if score_overlay:
            _scores_html = "".join(
                f'<div class="score-row"><span class="score-k">{k}</span>'
                f'<span class="score-v">{v}</span></div>'
                for k, v in score_overlay.items()
            )

        _port = server.server_address[1]
        _iframe_id = f"pymol-iframe-{key}"
        # Cache-bust the iframe src so the browser always reloads the viewer
        # when Streamlit rerenders (otherwise same-src iframes are not reloaded).
        import time as _time
        _ts = int(_time.time())
        _pymol_host = "localhost"
        try:
            _req_host = st.context.headers.get("host", "localhost")
            _pymol_host = _req_host.split(":")[0]
        except Exception:
            pass

        iframe_html = f"""
        <iframe
          id="{_iframe_id}"
          src="http://{_pymol_host}:{_port}/?t={_ts}"
          width="100%"
          height="{height}px"
          style="border:none; border-radius:8px; background:#0a0a0f; display:block;"
          allow="*"
        ></iframe>
        <script>
        (function() {{
          var fr = document.getElementById('{_iframe_id}');
          fr.addEventListener('load', function() {{
            fr.contentWindow.postMessage({{
              type: 'pdh-load',
              legend_html: {json.dumps(_legend_html)},
              scores_html: {json.dumps(_scores_html)},
            }}, '*');
          }});
        }})();
        </script>
        """
        st.markdown(iframe_html, unsafe_allow_html=True)

        # ---------- Action buttons ----------
        _bcols = st.columns([1, 1, 1])
        pymol_binary = _find_pymol_binary()

        with _bcols[0]:
            if pymol_binary and st.button(
                "🔬 Open in PyMOL",
                key=f"{key}_open_pymol",
                help="Launch full PyMOL GUI window",
                width='stretch',
            ):
                pid = launch_pymol_interactive(
                    structs,
                    highlight_residues=highlight_residues,
                    restraint_residues=restraint_residues,
                    mutation_label=mutation_label,
                    score_labels=score_overlay,
                    superimpose=superimpose,
                )
                if pid:
                    st.success(f"PyMOL GUI launched (PID {pid})")
                else:
                    st.error("Failed to launch PyMOL")

        with _bcols[1]:
            if st.button("🔄 Reload", key=f"{key}_reload",
                         help="Reload structure into viewer",
                         width='stretch'):
                server.load_structure(
                    pdb_data=pdb_data,
                    name=structs[0][1],
                    highlight_residues=highlight_residues,
                    restraint_residues=restraint_residues,
                    mutation_label=mutation_label,
                )
                st.rerun()

        with _bcols[2]:
            if st.button("📸 Save PNG", key=f"{key}_savepng",
                         help="Download current PyMOL view as PNG",
                         width='stretch'):
                png = server.get_frame()
                st.download_button(
                    "⬇ Download PNG",
                    data=png,
                    file_name=f"{structs[0][1]}_view.png",
                    mime="image/png",
                    key=f"{key}_dl",
                )

    else:
        # ---------- Fallback: 3Dmol.js (needs WebGL) ----------
        if len(structs) == 1:
            html_str = create_structure_viewer(
                structs[0][0],
                title=structs[0][1],
                height=height,
                highlight_residues=highlight_residues,
                mutation_label=mutation_label,
                score_overlay=score_overlay,
            )
        else:
            html_str = create_structure_comparison_3d(
                model_path=structs[0][0],
                reference_path=structs[1][0],
                title=title,
                height=height,
            )
        components.html(html_str, height=height + 20, scrolling=False)

        pymol_binary = _find_pymol_binary()
        if pymol_binary and st.button("🔬 Open in PyMOL", key=f"{key}_open_pymol2"):
            launch_pymol_interactive(
                structs,
                highlight_residues=highlight_residues,
                restraint_residues=restraint_residues,
                mutation_label=mutation_label,
                score_labels=score_overlay,
            )
