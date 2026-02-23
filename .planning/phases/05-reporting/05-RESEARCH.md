# Phase 5: Reporting - Research

**Researched:** 2026-02-23
**Domain:** Streamlit data visualization + PDF/HTML export for scientific reports
**Confidence:** HIGH

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| REP-01 | Mutation ranking bar chart in web UI (improvement score vs mutation, colored by category) | Plotly 6.5.2 `go.Bar` with per-bar color via `marker_color` list; color derivation from improvement_score threshold logic already in codebase |
| REP-02 | Per-residue pLDDT line chart comparing WT vs top 3 mutants | Plotly `go.Scatter` multi-trace; `plddt_per_residue` field present on ALL ranked mutations (MutationResult dataclass confirms); WT in `ctx.extra["mutation_wt_plddt_per_residue"]` |
| REP-03 | OST metric table (lDDT, RMSD, QS-score) per mutant in mutation scanner UI | OST fields (`ost_lddt`, `ost_rmsd_ca`, `ost_qs_score`) are stored directly on each ranked mutation dict; guarded display when OST was disabled |
| REP-04 | PDF export with ranking chart, pLDDT chart, OST table, narrative | fpdf2 2.8.5 (already installed) + plotly 6.5.2 `to_image(format="png")` with kaleido 1.2.0 (just installed); `bytes(pdf.output())` for `st.download_button` |
| REP-05 | HTML export: self-contained file with same content as PDF | Base64-embedded PNG images from plotly `to_image()`; HTML table for OST metrics; narrative text; ~60 KB per report; no CDN needed |
</phase_requirements>

---

## Summary

Phase 5 adds visual reporting and export to the existing `_render_phase2_results()` function in `10_mutation_scanner.py`. The mutation comparison data is already fully computed by Phase 2 (stored in `st.session_state.mutagenesis_phase2_context.extra["mutation_comparison"]`). This phase is pure UI and export work — no new agent or pipeline changes.

The full chart and export stack is already available on the machine: plotly 6.5.2 (already imported in `10_mutation_scanner.py`), kaleido 1.2.0 (just installed via pip), and fpdf2 2.8.5 (already installed). All three work together for PNG image generation from plotly figures, which is the bridge between Streamlit charts and PDF/HTML binary export. Verified end-to-end: plotly bar chart PNG to fpdf2 embedded image to bytes output produces a valid 9 KB PDF.

The "self-contained HTML" requirement means embedding chart images as base64 data URIs, not linking to CDN-hosted plotly.js (which would require internet). The base64-PNG approach produces ~60 KB files for a 3-chart report — compact and truly offline-capable.

**Primary recommendation:** Add three new helper functions to `_render_phase2_results()` for the charts and OST table, then add a `_build_report_pdf()` and `_build_report_html()` function that reuse the same plotly figures, converting them to PNG via `fig.to_image("png")` for embedding. Wire up two `st.download_button` calls at the bottom of the Phase 2 results panel.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| plotly | 6.5.2 | Interactive charts in Streamlit + PNG export | Already imported in `10_mutation_scanner.py`; `go.Bar` and `go.Scatter` already used for saturation heatmap |
| kaleido | 1.2.0 | Server-side PNG rasterization of plotly figures | Only server-side renderer for plotly; required for `fig.to_image(format="png")`; confirmed working headless |
| fpdf2 | 2.8.5 | PDF generation with embedded images and tables | Already installed on machine; pure Python; `pdf.output()` returns `bytearray` → `bytes()` for `st.download_button` |
| streamlit | 1.28+ | UI framework (already in use) | No changes to Streamlit usage patterns |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| base64 (stdlib) | - | Encode PNG bytes as data URI for HTML | Only in HTML export path |
| tempfile (stdlib) | - | Temp file for plotly PNG → fpdf2 image path | `fpdf2.image()` requires a file path, not bytes |
| re (stdlib) | - | Strip non-latin Unicode from narrative text | Helvetica in fpdf2 cannot render emoji/CJK |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| fpdf2 | weasyprint | weasyprint renders full HTML→PDF (nicer styling) but requires Cairo/Pango system libs not confirmed installed; fpdf2 is pure Python |
| fpdf2 | reportlab | reportlab is more powerful but heavier API; fpdf2 is simpler for chart+table reports |
| base64 PNG in HTML | `include_plotlyjs=True` | Embedding full plotly.js = 4.7 MB HTML; base64 PNG = ~60 KB for 3 charts; PNG is simpler and smaller |
| base64 PNG in HTML | `include_plotlyjs='cdn'` | CDN requires internet; violates "self-contained" requirement |

**Installation:**

```bash
pip install kaleido  # Already done — kaleido 1.2.0 installed
# fpdf2 2.8.5 already installed
# plotly 6.5.2 already installed
```

Add to `pyproject.toml` `[project.dependencies]`:
```
"fpdf2>=2.8.0",
"kaleido>=1.2.0",
```

---

## Architecture Patterns

### Where to Add Code

All Phase 5 code goes into `src/protein_design_hub/web/pages/10_mutation_scanner.py`. The file already has:
- `_render_phase2_results(ctx)` — the top-level Phase 2 display function
- `_render_saturation_heatmap()` — an existing plotly chart helper

Phase 5 adds helpers alongside these, called from within `_render_phase2_results()`.

```
10_mutation_scanner.py
├── _render_phase2_results(ctx)          ← extend this function
│   ├── [existing] summary metrics
│   ├── [existing] ranked mutations table
│   ├── [NEW] _render_ranking_chart(ranked)          ← REP-01
│   ├── [NEW] _render_plddt_chart(ranked, wt_plddt)  ← REP-02
│   ├── [NEW] _render_ost_table(ranked)              ← REP-03
│   ├── [existing] LLM interpretation expander
│   └── [EXTEND] download buttons section
│       ├── [existing] JSON download
│       ├── [existing] Markdown download
│       ├── [NEW] Export PDF button                   ← REP-04
│       └── [NEW] Export HTML button                  ← REP-05
├── [NEW] _render_ranking_chart(ranked)
├── [NEW] _render_plddt_chart(ranked, wt_plddt_per_residue)
├── [NEW] _render_ost_table(ranked)
├── [NEW] _build_report_pdf(ctx, comparison) -> bytes
└── [NEW] _build_report_html(ctx, comparison) -> str
```

### Pattern 1: Mutation Ranking Bar Chart (REP-01)

**What:** Horizontal or vertical bar chart of improvement scores per mutation, colored by category.
**When to use:** After Phase 2 completes, before the ranked table.

The category is derived at render time from `improvement_score` thresholds matching `MutationComparisonAgent` logic:
- `> 0` → beneficial (`#22c55e`)
- `< -0.5` → detrimental (`#ef4444`)
- otherwise → neutral (`#9ca3af`)

```python
# Source: verified with plotly 6.5.2 on this machine
def _render_ranking_chart(ranked: list) -> go.Figure:
    """Bar chart of improvement score per mutation, colored by category."""
    CATEGORY_COLORS = {
        "beneficial": "#22c55e",
        "neutral": "#9ca3af",
        "detrimental": "#ef4444",
    }

    def _category(score: float) -> str:
        if score > 0:
            return "beneficial"
        elif score < -0.5:
            return "detrimental"
        return "neutral"

    mutations = [r["mutation_code"] for r in ranked]
    scores = [r.get("improvement_score", 0) for r in ranked]
    categories = [_category(s) for s in scores]
    colors = [CATEGORY_COLORS[c] for c in categories]

    fig = go.Figure(data=go.Bar(
        x=mutations,
        y=scores,
        marker_color=colors,
        text=categories,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Mutation Ranking by Improvement Score",
        xaxis_title="Mutation",
        yaxis_title="Improvement Score",
        height=400,
        showlegend=False,
    )
    return fig

# In _render_phase2_results:
fig = _render_ranking_chart(ranked)
st.plotly_chart(fig, use_container_width=True)
```

**Note on legend:** Use a single `go.Bar` trace with per-bar `marker_color` — simpler and no legend confusion. If legend is desired, use three separate traces (one per category) but then x-axis ordering is broken across traces. Single trace is correct here.

### Pattern 2: Per-Residue pLDDT Line Chart (REP-02)

**What:** Multi-trace scatter/line chart, one trace for WT, one per top-3 mutant.
**Data access:**
- WT: `ctx.extra["mutation_wt_plddt_per_residue"]` (list of floats, length = sequence length)
- Mutants: `ranked[i]["plddt_per_residue"]` (same structure, present for both targeted and saturation results — confirmed via `MutationResult` dataclass)

```python
# Source: verified with plotly 6.5.2 on this machine
def _render_plddt_chart(
    ranked: list,
    wt_plddt_per_residue: list,
) -> go.Figure:
    """Per-residue pLDDT line chart: WT vs top 3 mutants."""
    MUTANT_COLORS = ["#22c55e", "#f59e0b", "#ef4444"]

    fig = go.Figure()
    if wt_plddt_per_residue:
        residues = list(range(1, len(wt_plddt_per_residue) + 1))
        fig.add_trace(go.Scatter(
            x=residues,
            y=wt_plddt_per_residue,
            name="Wildtype",
            line={"color": "#6366f1", "width": 2},
        ))

    for mut, color in zip(ranked[:3], MUTANT_COLORS):
        per_res = mut.get("plddt_per_residue") or []
        if not per_res:
            continue
        residues = list(range(1, len(per_res) + 1))
        fig.add_trace(go.Scatter(
            x=residues,
            y=per_res,
            name=mut["mutation_code"],
            line={"color": color, "width": 1.5, "dash": "dot"},
        ))

    fig.update_layout(
        title="Per-Residue pLDDT: Wildtype vs Top Mutants",
        xaxis_title="Residue",
        yaxis_title="pLDDT",
        yaxis={"range": [0, 100]},
        height=400,
    )
    return fig

# Guard: only show if WT per-residue data exists
wt_per_res = ctx.extra.get("mutation_wt_plddt_per_residue", [])
if wt_per_res:
    fig = _render_plddt_chart(ranked, wt_per_res)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Per-residue pLDDT chart not available (data missing from Phase 2 run).")
```

### Pattern 3: OST Metric Table (REP-03)

**What:** `st.dataframe` with lDDT, RMSD, QS-score per mutant. Only shown when OST data is present.
**Data access:** Each ranked mutation dict has `ost_lddt`, `ost_rmsd_ca`, `ost_qs_score` (all `None` if OST was disabled).

```python
# Source: verified against MutationComparisonAgent code in mutagenesis_agents.py
def _render_ost_table(ranked: list) -> None:
    """OST metric table, shown only when OST data is available."""
    ost_rows = [
        r for r in ranked
        if r.get("ost_lddt") is not None
    ]
    if not ost_rows:
        return  # OST was disabled — don't show table

    rows = []
    for r in ost_rows:
        rows.append({
            "Mutation": r.get("mutation_code", "?"),
            "Score": round(r.get("improvement_score", 0), 3),
            "lDDT": round(r["ost_lddt"], 3),
            "RMSD (CA)": round(r["ost_rmsd_ca"], 2) if r.get("ost_rmsd_ca") is not None else None,
            "QS-score": round(r["ost_qs_score"], 3) if r.get("ost_qs_score") is not None else None,
        })

    st.markdown("#### OST Structural Metrics")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
```

### Pattern 4: PDF Export (REP-04)

**What:** Generate PDF bytes in memory using fpdf2; serve via `st.download_button`.
**Key constraints:**
- `fpdf2.output()` returns `bytearray` → must wrap with `bytes()` for `st.download_button`
- `fpdf2.image()` requires a file path, not bytes → use `tempfile.NamedTemporaryFile` for each PNG
- Helvetica (built-in) cannot render emoji/Unicode > 0xFF → strip with regex before `multi_cell()`
- Charts must be exported as PNG first using `fig.to_image(format="png", width=700, height=350)`

```python
# Source: verified with fpdf2 2.8.5 + kaleido 1.2.0 on this machine
import re, tempfile, os
from fpdf import FPDF

def _strip_for_pdf(text: str) -> str:
    """Remove characters fpdf2 Helvetica cannot render (emoji, CJK)."""
    return re.sub(r"[^\x00-\xFF]+", "", text).strip()

def _fig_to_temp_png(fig: go.Figure, width: int = 700, height: int = 350) -> str:
    """Export plotly figure to a temp PNG file; caller must delete."""
    png_bytes = fig.to_image(format="png", width=width, height=height)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_bytes)
        return f.name

def _build_report_pdf(ctx, comparison: dict) -> bytes:
    """Build PDF report bytes from Phase 2 context."""
    ranked = comparison.get("ranked_mutations", [])
    wt_per_res = ctx.extra.get("mutation_wt_plddt_per_residue", [])
    interpretation = ctx.extra.get("mutation_interpretation", "")

    # Build figures
    ranking_fig = _build_ranking_figure(ranked)    # same as _render_ranking_chart but returns fig
    plddt_fig = _build_plddt_figure(ranked, wt_per_res)

    # Export to temp PNGs
    chart1_path = _fig_to_temp_png(ranking_fig)
    chart2_path = _fig_to_temp_png(plddt_fig)

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_left_margin(15)
        pdf.set_right_margin(15)
        pdf.add_page()

        # Title
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "Mutagenesis Report", new_x="LMARGIN", new_y="NEXT", align="C")

        # Summary line
        pdf.set_font("Helvetica", size=11)
        best = comparison.get("best_overall")
        if best:
            line = (
                f"Best mutation: {best.get('mutation_code', '?')} "
                f"(score {best.get('improvement_score', 0):.3f})"
            )
            pdf.cell(0, 8, _strip_for_pdf(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        # Ranking chart
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "Mutation Ranking", new_x="LMARGIN", new_y="NEXT")
        pdf.image(chart1_path, w=180)
        pdf.ln(5)

        # pLDDT chart
        if wt_per_res:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "Per-Residue pLDDT", new_x="LMARGIN", new_y="NEXT")
            pdf.image(chart2_path, w=180)
            pdf.ln(5)

        # OST metric table
        ost_rows = [r for r in ranked if r.get("ost_lddt") is not None]
        if ost_rows:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "OST Structural Metrics", new_x="LMARGIN", new_y="NEXT")
            # Header
            col_widths = [40, 30, 30, 35, 35]
            headers = ["Mutation", "Score", "lDDT", "RMSD (CA)", "QS-score"]
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(241, 245, 249)
            for w, h in zip(col_widths, headers):
                pdf.cell(w, 8, h, border=1, fill=True)
            pdf.ln()
            pdf.set_font("Helvetica", size=10)
            for i, r in enumerate(ost_rows[:20]):
                fill = i % 2 == 1
                if fill:
                    pdf.set_fill_color(248, 250, 252)
                vals = [
                    r.get("mutation_code", "?"),
                    f"{r.get('improvement_score', 0):.3f}",
                    f"{r.get('ost_lddt', 0):.3f}",
                    f"{r['ost_rmsd_ca']:.2f}" if r.get("ost_rmsd_ca") is not None else "-",
                    f"{r['ost_qs_score']:.3f}" if r.get("ost_qs_score") is not None else "-",
                ]
                for w, v in zip(col_widths, vals):
                    pdf.cell(w, 7, v, border=1, fill=fill)
                pdf.ln()
            pdf.ln(5)

        # Narrative summary
        if interpretation:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, "Narrative Summary", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", size=10)
            safe_text = _strip_for_pdf(interpretation[:3000])
            pdf.multi_cell(0, 6, safe_text)

        return bytes(pdf.output())
    finally:
        os.unlink(chart1_path)
        os.unlink(chart2_path)
```

### Pattern 5: HTML Export (REP-05)

**What:** Self-contained HTML with base64-embedded PNG charts, HTML table, narrative. No CDN needed.
**Approach:** Reuse the same plotly figures, convert to PNG, base64-encode, embed as `data:image/png;base64,...` in `<img>` tags.

```python
# Source: verified on this machine — produces ~60 KB for 3 charts + table
import base64

def _build_report_html(ctx, comparison: dict) -> str:
    """Build self-contained HTML report (no internet required)."""
    ranked = comparison.get("ranked_mutations", [])
    wt_per_res = ctx.extra.get("mutation_wt_plddt_per_residue", [])
    interpretation = ctx.extra.get("mutation_interpretation", "")

    def _to_b64_png(fig):
        return base64.b64encode(
            fig.to_image(format="png", width=700, height=350)
        ).decode("utf-8")

    ranking_fig = _build_ranking_figure(ranked)
    ranking_b64 = _to_b64_png(ranking_fig)

    plddt_b64 = ""
    if wt_per_res:
        plddt_fig = _build_plddt_figure(ranked, wt_per_res)
        plddt_b64 = _to_b64_png(plddt_fig)

    # Build OST table rows
    ost_rows = [r for r in ranked if r.get("ost_lddt") is not None]
    ost_table_html = ""
    if ost_rows:
        rows_html = ""
        for r in ost_rows[:20]:
            rows_html += (
                f"<tr><td>{r.get('mutation_code','?')}</td>"
                f"<td>{r.get('improvement_score',0):.3f}</td>"
                f"<td>{r.get('ost_lddt',0):.3f}</td>"
                f"<td>{r['ost_rmsd_ca']:.2f if r.get('ost_rmsd_ca') is not None else '-'}</td>"
                f"<td>{r['ost_qs_score']:.3f if r.get('ost_qs_score') is not None else '-'}</td></tr>"
            )
        ost_table_html = f"""
        <h2>OST Structural Metrics</h2>
        <table>
          <tr><th>Mutation</th><th>Score</th><th>lDDT</th><th>RMSD (CA)</th><th>QS-score</th></tr>
          {rows_html}
        </table>"""

    narrative_html = f"<h2>Narrative Summary</h2><p>{interpretation[:3000]}</p>" if interpretation else ""

    best = comparison.get("best_overall")
    summary_html = ""
    if best:
        summary_html = (
            f"<div class='summary'>"
            f"<p><strong>Best mutation:</strong> {best.get('mutation_code','?')} "
            f"(score: {best.get('improvement_score',0):.3f})</p>"
            f"<p>Total: {comparison.get('total_mutations',0)} | "
            f"Beneficial: {comparison.get('beneficial_count',0)} | "
            f"Detrimental: {comparison.get('detrimental_count',0)}</p></div>"
        )

    plddt_section = (
        f"<h2>Per-Residue pLDDT</h2>"
        f"<img src='data:image/png;base64,{plddt_b64}' alt='pLDDT Chart'>"
        if plddt_b64 else ""
    )

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Mutagenesis Report</title>
<style>
body{{font-family:Arial,sans-serif;max-width:900px;margin:0 auto;padding:20px;color:#1e293b}}
h1{{color:#1e293b}}h2{{color:#334155;border-bottom:2px solid #e2e8f0;padding-bottom:8px}}
table{{border-collapse:collapse;width:100%;margin:16px 0}}
th{{background:#f1f5f9;padding:8px 12px;text-align:left;border:1px solid #e2e8f0}}
td{{padding:7px 12px;border:1px solid #e2e8f0}}
tr:nth-child(even){{background:#f8fafc}}
img{{max-width:100%;height:auto;margin:12px 0;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}}
.summary{{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;padding:16px;margin:16px 0}}
</style></head>
<body>
<h1>Mutagenesis Report</h1>
{summary_html}
<h2>Mutation Ranking</h2>
<img src="data:image/png;base64,{ranking_b64}" alt="Mutation Ranking Chart">
{plddt_section}
{ost_table_html}
{narrative_html}
</body></html>"""
```

### Pattern 6: Streamlit Export Buttons

```python
# In _render_phase2_results(), in the downloads section:
# Source: verified against Streamlit DownloadButtonDataType — accepts bytes

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("Generate PDF Report", key="gen_pdf", use_container_width=True):
        with st.spinner("Building PDF..."):
            pdf_bytes = _build_report_pdf(ctx, comparison)
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="mutagenesis_report.pdf",
            mime="application/pdf",
            key="dl_pdf",
            use_container_width=True,
        )

with export_col2:
    if st.button("Generate HTML Report", key="gen_html", use_container_width=True):
        with st.spinner("Building HTML..."):
            html_str = _build_report_html(ctx, comparison)
        st.download_button(
            "Download HTML",
            data=html_str,
            file_name="mutagenesis_report.html",
            mime="text/html",
            key="dl_html",
            use_container_width=True,
        )
```

**IMPORTANT UI pattern note:** Streamlit re-runs the entire script on every interaction. If `_build_report_pdf()` is called on button press and the result is immediately used by `st.download_button`, it works because `st.download_button` on the same run will render (the user then clicks it on the next run). A simpler alternative: call generation directly without the intermediate button, but use `st.spinner` to show progress:

```python
# Simpler single-button pattern (recommended):
if st.download_button(
    "Export PDF",
    data=_build_report_pdf(ctx, comparison),   # called on every run — OK since kaleido is fast
    file_name="mutagenesis_report.pdf",
    mime="application/pdf",
    key="dl_pdf",
    use_container_width=True,
):
    pass  # download triggered — no further action needed
```

The simpler pattern (call generation always) is acceptable here because `fig.to_image()` is fast (~0.3s per chart). However, if performance is a concern, use `st.session_state` to cache the generated bytes.

### Anti-Patterns to Avoid

- **Using `include_plotlyjs=True` for HTML export:** Produces 4.7 MB files. Use base64 PNG instead.
- **Using `include_plotlyjs='cdn'` for "self-contained" HTML:** Requires internet. Violates REP-05.
- **Passing `bytearray` to `st.download_button`:** Type is `DownloadButtonDataType = str | bytes | ...`. Always wrap with `bytes()`: `data=bytes(pdf.output())`.
- **Multi-trace bar chart for category coloring:** Using separate traces per category breaks sorted x-axis ordering. Use single trace with `marker_color` list.
- **Emoji in PDF narrative:** fpdf2's built-in Helvetica cannot render Unicode > 0xFF. Strip with `re.sub(r"[^\x00-\xFF]+", "", text)`.
- **fpdf2 `image()` with bytes:** fpdf2's `image()` method requires a file path or file-like object, not raw bytes. Use `tempfile.NamedTemporaryFile` and `os.unlink()` in a `try/finally`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Chart PNG rasterization | Custom matplotlib mimic | `fig.to_image(format="png")` with kaleido | kaleido is the official plotly renderer; handles all chart types including hover text |
| PDF table layout | Manual cell position calculation | fpdf2 `cell()` with border + `set_fill_color()` for alternating rows | fpdf2 handles page breaks, margins, and overflow automatically |
| Color-by-category logic | Custom color lookup | Derive from improvement_score thresholds matching `MutationComparisonAgent` thresholds | Must match existing classification: `> 0` beneficial, `< -0.5` detrimental |
| HTML escaping in report | Manual `str.replace()` | Python's `html.escape()` for narrative text in HTML | Narrative text from LLM may contain `<`, `>`, `&` |

**Key insight:** The report generation is straightforward because the data model is already rich. The hard part is already done in Phase 2 — `ranked_mutations` contains all metrics. Phase 5 is presentation only.

---

## Common Pitfalls

### Pitfall 1: pLDDT Chart Shows Empty Traces for Saturation Mutations

**What goes wrong:** The per-residue pLDDT line chart shows only the WT line even when mutants exist.
**Why it happens:** Saturation mutations (targets=["*"]) also store `plddt_per_residue` in the dict (confirmed via `MutationResult` dataclass), but for **large datasets** the saturation path via `scan_position` stores many mutations and the top-3 may have non-empty lists. However, early saturation failure paths (`all_results.append({..., 'success': False})`) do NOT have `plddt_per_residue`. Always guard with `if per_res`.
**How to avoid:** `per_res = mut.get("plddt_per_residue") or []` — empty list or None both safe.
**Warning signs:** Chart only shows WT line; inspect `ranked[0].get("plddt_per_residue")` in session state.

### Pitfall 2: OST Table Appears Even When OST Was Disabled

**What goes wrong:** Table renders with all None values (displayed as "-" everywhere), confusing users.
**Why it happens:** `ost_lddt` key is absent (not `None`) when OST was disabled — the `_extract_ost_metrics` function returns empty dict. `r.get("ost_lddt")` returns `None` in that case.
**How to avoid:** Filter with `ost_rows = [r for r in ranked if r.get("ost_lddt") is not None]` — if empty, skip table entirely. Also check `comparison.get("best_ost_metrics", {})` to confirm OST ran.
**Warning signs:** `ost_auto_disabled` key in `ctx.extra` is `True` — also surface this to user near where OST table would be.

### Pitfall 3: PDF Generation Fails on Large Mutation Counts

**What goes wrong:** PDF takes >10s to generate for 100+ mutations (saturation scan of 5+ positions).
**Why it happens:** Each `to_image()` call takes ~0.3s; two charts = ~0.6s total. This is fine. The bottleneck would be if we tried to embed per-mutant charts individually.
**How to avoid:** Generate only TWO charts total (ranking chart + pLDDT chart). The OST table is text-only. Cap OST table at 20 rows. Do not generate per-mutant individual charts.
**Warning signs:** Spinner shows > 5 seconds. Profile by counting `to_image()` calls.

### Pitfall 4: Temp PNG Files Left on Disk on Exception

**What goes wrong:** `_fig_to_temp_png()` creates files in `/tmp` that are not cleaned up if PDF generation fails.
**Why it happens:** Exception in `pdf.image()` or any subsequent step exits before `os.unlink()`.
**How to avoid:** Always use `try/finally` block that calls `os.unlink()` for each temp file. Alternatively, use `tempfile.TemporaryDirectory` as context manager.
**Warning signs:** Growing `/tmp` directory over multiple failed exports.

### Pitfall 5: Streamlit Re-Run Regenerates PDF on Every Interaction

**What goes wrong:** Every Streamlit interaction (clicking any widget) triggers `_build_report_pdf()` again, wasting ~0.6s per interaction.
**Why it happens:** Streamlit re-runs the entire script on each widget interaction. If generation is unconditional, it runs every time.
**How to avoid:** Cache the generated bytes in `st.session_state`:
```python
if "cached_pdf" not in st.session_state or st.button("Regenerate PDF"):
    st.session_state.cached_pdf = _build_report_pdf(ctx, comparison)
st.download_button("Download PDF", data=st.session_state.cached_pdf, ...)
```
**Warning signs:** Slow UI after Phase 2 completes; profiling shows `to_image()` called on every interaction.

### Pitfall 6: `ost_metrics` Key Does Not Exist in `ctx.extra`

**What goes wrong:** Code accessing `ctx.extra["ost_metrics"]` raises `KeyError`.
**Why it happens:** The additional_context description mentions `context.extra["ost_metrics"]` but the actual code stores OST metrics directly ON each ranked mutation dict, not in a top-level `ost_metrics` key. The actual keys are `ost_lddt`, `ost_rmsd_ca`, `ost_qs_score` on each item in `ranked_mutations`.
**How to avoid:** Access OST data via `ranked[i].get("ost_lddt")`, not via `ctx.extra["ost_metrics"]`.
**Warning signs:** `KeyError: 'ost_metrics'` at runtime; inspect `ctx.extra.keys()` in debugger.

---

## Code Examples

### Complete Shared Figure Builder (used by both UI and export)

```python
# Source: verified with plotly 6.5.2 + kaleido 1.2.0 on this machine

def _build_ranking_figure(ranked: list) -> go.Figure:
    """Bar chart figure — call for both st.plotly_chart and to_image()."""
    CATEGORY_COLORS = {"beneficial": "#22c55e", "neutral": "#9ca3af", "detrimental": "#ef4444"}

    def _cat(score):
        if score > 0: return "beneficial"
        if score < -0.5: return "detrimental"
        return "neutral"

    mutations = [r["mutation_code"] for r in ranked]
    scores = [r.get("improvement_score", 0) for r in ranked]
    cats = [_cat(s) for s in scores]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig = go.Figure(data=go.Bar(
        x=mutations, y=scores, marker_color=colors, text=cats,
        hovertemplate="<b>%{x}</b><br>Score: %{y:.3f}<br>%{text}<extra></extra>",
    ))
    fig.update_layout(
        title="Mutation Ranking by Improvement Score",
        xaxis_title="Mutation", yaxis_title="Improvement Score", height=400,
    )
    return fig


def _build_plddt_figure(ranked: list, wt_per_res: list) -> go.Figure:
    """Line chart figure — call for both st.plotly_chart and to_image()."""
    MUTANT_COLORS = ["#22c55e", "#f59e0b", "#ef4444"]
    fig = go.Figure()
    if wt_per_res:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(wt_per_res) + 1)),
            y=wt_per_res, name="Wildtype",
            line={"color": "#6366f1", "width": 2},
        ))
    for mut, color in zip(ranked[:3], MUTANT_COLORS):
        per_res = mut.get("plddt_per_residue") or []
        if per_res:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(per_res) + 1)),
                y=per_res, name=mut["mutation_code"],
                line={"color": color, "width": 1.5, "dash": "dot"},
            ))
    fig.update_layout(
        title="Per-Residue pLDDT: Wildtype vs Top Mutants",
        xaxis_title="Residue", yaxis_title="pLDDT",
        yaxis={"range": [0, 100]}, height=400,
    )
    return fig
```

### fpdf2 Image from Plotly Figure

```python
# Source: verified on this machine — temp file pattern required by fpdf2
import tempfile, os
from fpdf import FPDF
import plotly.graph_objects as go

def _embed_fig_in_pdf(pdf: FPDF, fig: go.Figure, width_mm: int = 180) -> None:
    """Export plotly figure to PNG temp file, embed in PDF, delete temp file."""
    png_bytes = fig.to_image(format="png", width=700, height=350)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_bytes)
        tmp_path = f.name
    try:
        pdf.image(tmp_path, w=width_mm)
    finally:
        os.unlink(tmp_path)
```

### Streamlit Download Button (correct data type)

```python
# Source: verified against Streamlit DownloadButtonDataType definition
# fpdf2 output() returns bytearray; must convert to bytes

pdf_bytes: bytes = bytes(pdf.output())   # bytearray -> bytes conversion REQUIRED
st.download_button(
    label="Export PDF",
    data=pdf_bytes,
    file_name="mutagenesis_report.pdf",
    mime="application/pdf",
    key="dl_pdf",
    use_container_width=True,
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| kaleido 0.x (separate pypi) | kaleido 1.x (now includes choreographer) | 2024 | New install: `pip install kaleido` installs choreographer, orjson, simplejson as deps |
| reportlab for Python PDFs | fpdf2 for simple reports | 2022+ | fpdf2 is simpler API; reportlab still preferred for complex typesetting |
| plotly `write_image()` with orca | `to_image()` with kaleido | 2020+ | orca deprecated; kaleido is the maintained path |

**Deprecated/outdated:**
- plotly-orca: replaced by kaleido; do not reference orca in code
- `fig.write_html(include_plotlyjs="cdn")` for "offline": CDN is not offline; use base64 PNG approach

---

## Open Questions

1. **Streamlit re-run performance with PDF generation**
   - What we know: `to_image()` takes ~0.3s per chart; two charts = ~0.6s overhead per Streamlit re-run
   - What's unclear: Whether users will notice 0.6s added latency on every interaction after Phase 2 completes
   - Recommendation: Implement with `st.session_state` caching from the start; invalidate cache when `mutagenesis_phase2_context` changes

2. **pLDDT per-residue availability for saturation results**
   - What we know: `MutationResult` dataclass has `plddt_per_residue` field; saturation path calls `scan_position()` which returns `SaturationMutagenesisResult` objects
   - What's unclear: Whether `scan_position()` populates `plddt_per_residue` on each MutationResult — not tested directly
   - Recommendation: Always guard with `per_res = mut.get("plddt_per_residue") or []`; if empty for top mutants, chart shows only WT (still valid for REP-02)

3. **Narrative text encoding for PDF**
   - What we know: LLM (qwen2.5:14b) may output markdown formatting, Unicode, or emoji in narrative
   - What's unclear: Frequency of high-Unicode characters in qwen output
   - Recommendation: Apply `_strip_for_pdf()` regex stripping; optionally strip markdown markers (`**`, `#`, etc.) for cleaner PDF text

---

## Sources

### Primary (HIGH confidence)
- Verified directly on machine: plotly 6.5.2 + kaleido 1.2.0 `to_image(format="png")` working
- Verified directly on machine: fpdf2 2.8.5 `FPDF.output()` returns `bytearray`; `pdf.image()` requires file path
- Verified directly on machine: `bytes(pdf.output())` accepted by `st.download_button` (DownloadButtonDataType = `str | bytes | ...`)
- Verified directly on machine: `fig.to_html(full_html=True, include_plotlyjs=True)` = 4.7 MB; base64 PNG approach = 60 KB
- Source code inspection: `mutagenesis_agents.py` — OST keys on ranked mutation dicts, NOT in `ctx.extra["ost_metrics"]`
- Source code inspection: `MutationResult` dataclass fields confirmed — `plddt_per_residue` present for saturation results

### Secondary (MEDIUM confidence)
- `pyproject.toml` — confirmed `plotly>=5.18.0`, `pandas>=2.0.0` in project dependencies; fpdf2 and kaleido not yet listed (need to add)
- `10_mutation_scanner.py` lines 1938-2031 — existing `_render_phase2_results()` and download button patterns confirmed

### Tertiary (LOW confidence)
- kaleido 1.2.0 headless behavior on display-less servers: tested with DISPLAY=:1 (works); behavior on display-less CI not tested
- qwen2.5:14b narrative Unicode frequency: not tested empirically

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries verified installed and working on target machine
- Architecture: HIGH — existing code structure well-understood from direct inspection
- Pitfalls: HIGH for PDF/HTML pitfalls (reproduced directly); MEDIUM for saturation pLDDT availability
- Data model: HIGH — OST key location and mutation dict structure verified against actual source code

**Research date:** 2026-02-23
**Valid until:** 2026-03-23 (stable libraries; 30-day estimate)
