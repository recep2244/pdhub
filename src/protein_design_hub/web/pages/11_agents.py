"""Agent Pipeline page -- LLM-guided protein design with scientist agents."""

import html as _html
import json
import time
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Agent Pipeline - Protein Design Hub",
    page_icon="🤖",
    layout="wide",
)

from protein_design_hub.web.ui import (
    inject_base_css,
    sidebar_nav,
    sidebar_system_status,
    page_header,
    section_header,
    metric_card,
    info_box,
    empty_state,
    workflow_breadcrumb,
)
from protein_design_hub.web.agent_helpers import agent_sidebar_status, render_agent_chatbot

inject_base_css()
sidebar_nav(current="Agents")
sidebar_system_status()
agent_sidebar_status()

# ── helpers ──────────────────────────────────────────────────────────

def _esc(text):
    return _html.escape(str(text))

def _short(text, n=120):
    t = str(text)
    return _esc(t[:n] + "..." if len(t) > n else t)

def _get_llm_cfg():
    try:
        from protein_design_hub.core.config import get_settings
        return get_settings().llm.resolve()
    except Exception:
        return None

@st.cache_data(ttl=30, show_spinner=False)
def _check_llm():
    """Return (connected, model_ok, message)."""
    cfg = _get_llm_cfg()
    if cfg is None:
        return False, False, "Could not load LLM config"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        models = client.models.list()
        ids = [m.id for m in models.data]
        if cfg.model in ids:
            return True, True, f"{cfg.model} @ {cfg.provider}"
        return True, False, f"Connected but model '{cfg.model}' not listed"
    except Exception as e:
        return False, False, str(e)

def _load_team_presets():
    from protein_design_hub.agents import scientists as S
    return {
        "default":       ("General prediction pipeline",     S.DEFAULT_TEAM_LEAD, S.DEFAULT_TEAM_MEMBERS),
        "design":        ("Protein design & engineering",    S.DEFAULT_TEAM_LEAD, S.DESIGN_TEAM_MEMBERS),
        "nanobody":      ("Antibody / nanobody engineering", S.DEFAULT_TEAM_LEAD, S.NANOBODY_TEAM_MEMBERS),
        "evaluation":    ("Structure quality assessment",    S.DEFAULT_TEAM_LEAD, S.EVALUATION_TEAM_MEMBERS),
        "refinement":    ("Structure refinement workflow",   S.DEFAULT_TEAM_LEAD, S.REFINEMENT_TEAM_MEMBERS),
        "mutagenesis":   ("Mutagenesis & sequence design",   S.DEFAULT_TEAM_LEAD, S.MUTAGENESIS_TEAM_MEMBERS),
        "mpnn_design":   ("MPNN inverse folding design",     S.DEFAULT_TEAM_LEAD, S.MPNN_DESIGN_TEAM_MEMBERS),
        "full_pipeline": ("Full pipeline review (all experts)", S.DEFAULT_TEAM_LEAD, S.FULL_PIPELINE_TEAM_MEMBERS),
    }

# ── CSS ──────────────────────────────────────────────────────────────

st.markdown("""<style>
.ps{display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:10px;margin-bottom:5px;border:1px solid var(--pdhub-border)}
.ps-llm{border-left:3px solid #a855f7;background:rgba(168,85,247,.06)}
.ps-comp{border-left:3px solid #06b6d4;background:rgba(6,182,212,.04)}
.ps-done{border-left-color:#22c55e!important;opacity:.85}
.ps-num{font-family:'JetBrains Mono',monospace;font-weight:700;font-size:.82rem;min-width:26px;height:26px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,.08);color:var(--pdhub-text-secondary)}
.ps-lbl{flex:1;font-weight:500;font-size:.87rem;color:var(--pdhub-text)}
.ps-bdg{font-size:.68rem;font-weight:600;padding:2px 9px;border-radius:12px;text-transform:uppercase;letter-spacing:.03em}
.bdg-l{background:rgba(168,85,247,.14);color:#a855f7}
.bdg-c{background:rgba(6,182,212,.14);color:#06b6d4}
.ps-t{font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--pdhub-text-muted)}
.ac{background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);border-radius:12px;padding:1rem 1.2rem;margin-bottom:.6rem}
.ac:hover{border-color:var(--pdhub-border-strong)}
.trn{background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.55rem}
.trn-a{font-weight:700;font-size:.86rem;margin-bottom:4px}
.trn-ai{color:#a855f7}.trn-u{color:#06b6d4}
.trn-b{color:var(--pdhub-text-secondary);font-size:.83rem;line-height:1.6;white-space:pre-wrap}
.pvc{background:var(--pdhub-bg-card);border:1px solid var(--pdhub-border);border-radius:10px;padding:11px 14px;margin-bottom:7px;font-size:.83rem}
.pvc-on{border-color:var(--pdhub-primary);background:rgba(99,102,241,.06)}
</style>""", unsafe_allow_html=True)

# ── header ───────────────────────────────────────────────────────────

page_header(
    "Agent Pipeline",
    "LLM-guided protein design with 10 specialist scientist agents",
    "🤖",
)

workflow_breadcrumb(
    ["Configure LLM", "Select Agents", "Run Pipeline", "Review Results"],
    current=0,
)

# ── quick status ─────────────────────────────────────────────────────

_cfg = _get_llm_cfg()
_conn, _mok, _cmsg = _check_llm()

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card(_cfg.provider if _cfg else "N/A", "LLM Provider", "info", "🔌")
with c2:
    metric_card(_cfg.model if _cfg else "N/A", "Model", "info", "🤖")
with c3:
    _st = "Online" if _conn and _mok else "Warning" if _conn else "Offline"
    _sv = "success" if _conn and _mok else "warning" if _conn else "error"
    metric_card(_st, "LLM Status", _sv, "🔗")
with c4:
    metric_card("10 steps", "Pipeline", "default", "📋")

if not _conn:
    info_box(
        f"LLM backend not reachable: {_cmsg}. "
        "Start Ollama with `ollama serve` or configure another provider in the LLM Status tab.",
        variant="warning", icon="⚠️",
    )

st.markdown("")

# ── session state ────────────────────────────────────────────────────

for _k, _v in {
    "pipe_running": False, "pipe_log": [], "pipe_result": None, "pipe_ctx": None,
    "meet_running": False, "meet_result": None, "meet_transcript": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── tabs ─────────────────────────────────────────────────────────────

tabs = st.tabs([
    "💬 Quick Chat",
    "🚀 Run Pipeline",
    "🧪 Agent Meeting",
    "🔍 Agent Explorer",
    "⚙️ LLM Status",
    "📂 Meeting History",
])


###############################################################################
# TAB 0 — QUICK CHAT
###############################################################################
with tabs[0]:
    section_header("Quick Chat", "Have a conversation with any scientist agent", "💬")

    info_box(
        "Chat with 10 specialist agents: ask about protein design, interpret results, "
        "plan experiments, or get a second opinion. The conversation stays in context "
        "so follow-up questions work naturally.",
        variant="info",
        icon="💡",
    )

    render_agent_chatbot(key_prefix="main_chat")


###############################################################################
# TAB 1 — RUN PIPELINE
###############################################################################
with tabs[1]:
    section_header("Run Agent Pipeline", "Full prediction workflow with optional LLM guidance", "🚀")

    # Quick-start guide
    with st.expander("📖 How does the pipeline work? (click to expand)", expanded=False):
        st.markdown("""
**The Agent Pipeline runs your protein through a multi-step workflow:**

| Step | What happens | Agent type |
|------|-------------|------------|
| 1. **Parse Input** | Reads your FASTA, validates sequence, counts residues | Compute |
| 2. **Planning Meeting** | LLM agents discuss which predictors to use and why | LLM Team |
| 3. **Structure Prediction** | Runs selected predictors (ESMFold, ColabFold, etc.) | Compute |
| 4. **Prediction Review** | Structural Biologist + Liam assess prediction quality | LLM Team |
| 5. **Evaluation** | Computes metrics: pLDDT, clash score, Ramachandran, etc. | Compute |
| 6. **Comparison** | Ranks predictors by composite score | Compute |
| 7. **Evaluation Review** | Biophysicist + Liam interpret metrics, flag concerns | LLM Team |
| 8. **Refinement Strategy** | Digital Recep + Liam advise on structure refinement | LLM Team |
| 9. **Mutagenesis Planning** | Protein Engineer + ML Specialist plan mutations/design | LLM Team |
| 10. **Report** | Saves results, metrics, and meeting summaries to disk | Compute |

---
**Quick Example — Predict a small protein:**
1. Select **"Paste sequence"** below
2. Paste this ubiquitin sequence: `MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG`
3. Set pipeline mode to **"Step-only (fast, no LLM)"** for a quick run, or **"LLM-guided"** for full agent discussion
4. Click **Run Pipeline** and watch the progress log

**Performance tips:**
- **Step-only mode** is much faster (seconds to minutes) — use for quick predictions
- **LLM-guided mode** adds ~30-60s per meeting (depends on your LLM backend speed)
- Use **Ollama with a small model** (e.g. `llama3.2:3b`) for faster LLM responses
- For large proteins (>500 residues), expect longer prediction times
        """)

    cin, ccf = st.columns([3, 2])

    with cin:
        st.markdown("#### Input Sequence")
        inp_method = st.radio(
            "How to provide input",
            ["Upload FASTA", "Paste sequence", "Select existing file"],
            horizontal=True, key="p_inp",
        )

        fasta_path = None

        if inp_method == "Upload FASTA":
            up = st.file_uploader("Choose a FASTA file", type=["fasta", "fa", "faa"], key="p_up")
            if up:
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", prefix="pdhub_")
                tmp.write(up.getvalue()); tmp.close()
                fasta_path = Path(tmp.name)
                st.success(f"Uploaded **{up.name}** ({len(up.getvalue()):,} bytes)")

        elif inp_method == "Paste sequence":
            sn = st.text_input("Sequence name", "query", key="p_sn")
            sq = st.text_area("Paste amino-acid sequence", height=90,
                              placeholder="MKFLVLLFNIAL...", key="p_sq")
            if sq and sq.strip():
                import tempfile
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".fasta", prefix="pdhub_")
                clean = sq.strip().replace(" ", "").replace("\n", "")
                tmp.write(f">{sn}\n{clean}\n".encode()); tmp.close()
                fasta_path = Path(tmp.name)
                st.caption(f"{len(clean)} residues")
        else:
            try:
                from protein_design_hub.core.config import get_settings
                od = Path(get_settings().output.base_dir)
                fastas = sorted(od.rglob("*.fasta")) + sorted(od.rglob("*.fa"))
                if fastas:
                    fasta_path = st.selectbox("Pick a FASTA file", fastas,
                        format_func=lambda p: str(p.relative_to(od)), key="p_ex")
                else:
                    st.info("No FASTA files found in outputs.")
            except Exception:
                st.info("Could not scan output directory.")

        ref_up = st.file_uploader("Reference structure (optional, for evaluation)", type=["pdb","cif"], key="p_ref")
        ref_path = None
        if ref_up:
            import tempfile as _tf
            tr = _tf.NamedTemporaryFile(delete=False, suffix=".pdb", prefix="pdhub_ref_")
            tr.write(ref_up.getvalue()); tr.close()
            ref_path = Path(tr.name)

    with ccf:
        st.markdown("#### Pipeline Settings")
        pm = st.selectbox("Pipeline mode", [
            "LLM-guided (recommended)",
            "Step-only (fast, no LLM)",
        ], key="p_mode")
        use_llm = "LLM" in pm

        try:
            from protein_design_hub.predictors.registry import PredictorRegistry
            av_preds = PredictorRegistry.list_available()
        except Exception:
            av_preds = ["colabfold", "chai1", "boltz2", "esmfold"]

        sel_preds = st.multiselect("Predictors (empty = all available)", av_preds, key="p_preds")
        job_id_in = st.text_input("Job ID (leave blank to auto-generate)", key="p_jid")

        provider = None
        model_ov = ""
        num_rounds = 1
        if use_llm:
            st.markdown("#### LLM Settings")
            try:
                from protein_design_hub.core.config import LLM_PROVIDER_PRESETS
                prov_names = list(LLM_PROVIDER_PRESETS.keys())
                ci = prov_names.index(_cfg.provider) if _cfg and _cfg.provider in prov_names else 0
            except Exception:
                prov_names = ["ollama","lmstudio","vllm","llamacpp","deepseek","openai","gemini","kimi"]
                ci = 0
            provider = st.selectbox("LLM Provider", prov_names, index=ci, key="p_prov")
            model_ov = st.text_input("Model override (empty = use provider default)", key="p_mov")
            num_rounds = st.slider("Discussion rounds per meeting", 1, 5, 1, key="p_rnd")

    st.markdown("---")

    # pipeline preview
    try:
        from protein_design_hub.agents.orchestrator import AgentOrchestrator, _AGENT_LABELS
        _po = AgentOrchestrator(mode="llm" if use_llm else "step")
        _steps = _po.describe_pipeline()
    except Exception:
        _steps = []
        _AGENT_LABELS = {}

    if _steps:
        st.markdown("#### Pipeline Steps Preview")
        for i, s in enumerate(_steps, 1):
            ll = s["type"] == "llm"
            cls = "ps-llm" if ll else "ps-comp"
            bc = "bdg-l" if ll else "bdg-c"
            bl = "LLM" if ll else "Compute"
            ic = "🧠" if ll else "⚙️"
            st.markdown(
                f'<div class="ps {cls}">'
                f'<div class="ps-num">{i}</div>'
                f'<span style="font-size:1.05rem">{ic}</span>'
                f'<div class="ps-lbl">{_esc(s["label"])}</div>'
                f'<div class="ps-bdg {bc}">{bl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # buttons
    br, bx, _ = st.columns([1, 1, 2])
    with br:
        run_dis = fasta_path is None or st.session_state.pipe_running
        run_btn = st.button(
            "▶ Run Pipeline" if not st.session_state.pipe_running else "⏳ Running...",
            type="primary", use_container_width=True, disabled=run_dis, key="btn_run",
        )
    with bx:
        if st.button("🗑 Clear Results", use_container_width=True, key="btn_clr"):
            for k in ("pipe_log", "pipe_result", "pipe_ctx"):
                st.session_state[k] = [] if k == "pipe_log" else None
            st.rerun()

    # execution
    if run_btn and fasta_path:
        st.session_state.pipe_running = True
        st.session_state.pipe_log = []
        st.session_state.pipe_result = None
        st.session_state.pipe_ctx = None

        if use_llm and provider:
            try:
                from protein_design_hub.core.config import get_settings
                s = get_settings()
                s.llm.provider = provider
                s.llm.base_url = ""
                s.llm.model = model_ov or ""
                s.llm.api_key = ""
            except Exception:
                pass

        mode = "llm" if use_llm else "step"
        t0 = time.time()

        def _prog(stage, item, current, total):
            lbl = _AGENT_LABELS.get(item, item)
            st.session_state.pipe_log.append({
                "step": current, "total": total, "name": item,
                "label": lbl, "is_llm": item.startswith("llm_"),
                "elapsed": time.time() - t0,
            })

        try:
            orch = AgentOrchestrator(
                mode=mode, progress_callback=_prog,
                **({"num_rounds": num_rounds} if use_llm else {}),
            )
            with st.spinner("Running agent pipeline... this may take several minutes."):
                result = orch.run(
                    input_path=fasta_path, output_dir=None,
                    reference_path=ref_path,
                    predictors=sel_preds or None,
                    job_id=job_id_in or None,
                )
            st.session_state.pipe_result = result
            if result.success and result.context:
                st.session_state.pipe_ctx = result.context
            st.session_state.pipe_running = False
            st.rerun()
        except Exception as e:
            st.session_state.pipe_running = False
            st.error(f"Pipeline error: {e}")
            import traceback
            with st.expander("Show traceback"):
                st.code(traceback.format_exc(), language="text")

    # execution log
    if st.session_state.pipe_log:
        st.markdown("#### Execution Log")
        for en in st.session_state.pipe_log:
            ic = "🧠" if en["is_llm"] else "✅"
            cls = "ps-llm ps-done" if en["is_llm"] else "ps-comp ps-done"
            st.markdown(
                f'<div class="ps {cls}">'
                f'<div class="ps-num">{en["step"]}</div>'
                f'<span style="font-size:1rem">{ic}</span>'
                f'<div class="ps-lbl">{_esc(en["label"])}</div>'
                f'<div class="ps-t">{en["elapsed"]:.1f}s</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # results
    result = st.session_state.pipe_result
    ctx = st.session_state.pipe_ctx
    if result is not None:
        if result.success:
            tt = st.session_state.pipe_log[-1]["elapsed"] if st.session_state.pipe_log else 0
            st.success(f"Pipeline completed in {tt:.1f}s")

            if ctx and ctx.prediction_results:
                section_header("Prediction Results", "", "🔮")
                cols = st.columns(min(len(ctx.prediction_results), 4))
                for i, (nm, pr) in enumerate(ctx.prediction_results.items()):
                    with cols[i % len(cols)]:
                        if pr.success:
                            bp = "-"
                            if pr.scores:
                                pp = [s.plddt for s in pr.scores if s.plddt]
                                if pp:
                                    bp = f"{max(pp):.1f}"
                            metric_card(bp, f"{nm} pLDDT", "success", "🔮")
                            st.caption(f"{len(pr.structure_paths)} structures, {pr.runtime_seconds:.1f}s")
                        else:
                            metric_card("FAIL", nm, "error", "❌")
                            st.caption(pr.error_message or "Unknown error")

            if ctx and ctx.evaluation_results:
                section_header("Evaluation Results", "", "📊")
                rows = []
                for nm, ev in ctx.evaluation_results.items():
                    rows.append({
                        "Predictor": nm,
                        "lDDT": f"{ev.lddt:.3f}" if ev.lddt else "-",
                        "TM-score": f"{ev.tm_score:.3f}" if ev.tm_score else "-",
                        "RMSD": f"{ev.rmsd:.2f}" if ev.rmsd else "-",
                    })
                if rows:
                    import pandas as pd
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if ctx and ctx.comparison_result and ctx.comparison_result.ranking:
                section_header("Ranking", "", "🏆")
                for rank, (nm, sc) in enumerate(ctx.comparison_result.ranking, 1):
                    ri = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
                    rc1, rc2, rc3 = st.columns([1, 4, 2])
                    with rc1:
                        st.markdown(f"### {ri}")
                    with rc2:
                        st.markdown(f"**{_esc(nm)}**")
                    with rc3:
                        st.code(f"{sc:.3f}", language=None)

            if ctx and ctx.extra:
                _lk = [("plan","Planning Meeting"),("prediction_review","Prediction Review"),
                       ("evaluation_review","Evaluation Review"),("refinement_review","Refinement Strategy"),
                       ("mutagenesis_plan","Mutagenesis & Design Planning")]
                if any(ctx.extra.get(k) for k,_ in _lk):
                    section_header("LLM Agent Discussions", "", "🧠")
                    for key, title in _lk:
                        txt = ctx.extra.get(key)
                        if txt:
                            with st.expander(f"📝 {title}", expanded=(key == "refinement_review")):
                                st.markdown(txt)

            if ctx and ctx.job_dir:
                st.markdown("---")
                info_box(f"Results saved to: `{ctx.job_dir}`", variant="success", icon="📂")
        else:
            st.error(f"Pipeline failed: {result.message}")
            if result.error:
                with st.expander("Error details"):
                    st.code(str(result.error), language="text")


###############################################################################
# TAB 2 — AGENT MEETING
###############################################################################
with tabs[2]:
    section_header("Agent Meeting", "Run ad-hoc LLM discussions on any topic", "💬")

    with st.expander("💡 Example questions to try", expanded=False):
        st.markdown("""
**Try pasting one of these into the agenda field:**

- *"I have a 250-residue enzyme that I want to thermostabilize for industrial use at 65C. Which computational strategy should I use?"*
- *"Compare ESMFold vs ColabFold vs Chai-1 for predicting a homodimeric cytokine receptor. Which is best and why?"*
- *"My pLDDT scores are 85 average but I see a loop region at residues 120-140 with pLDDT below 50. Should I be concerned?"*
- *"Design a ProteinMPNN campaign for a TIM barrel enzyme. Which positions should I fix and which should I redesign?"*
- *"I have a nanobody with a 22-residue CDR3 loop. How reliable will the structure prediction be?"*

**Team selection guide:**
- **Default** — general questions about prediction and pipelines
- **Mutagenesis** — mutation scanning, library design, stability engineering
- **MPNN Design** — ProteinMPNN settings, inverse folding, sequence recovery
- **Evaluation** — structure quality, model assessment, MolProbity scores
- **Refinement** — AMBER/Rosetta relaxation, fixing clashes, improving geometry
        """)

    cm1, cm2 = st.columns([3, 2])
    with cm1:
        agenda = st.text_area(
            "Meeting agenda or question", height=120,
            placeholder="e.g. Which predictor is best for a 300-residue monomer?",
            key="m_agenda",
        )
    with cm2:
        mtype = st.selectbox("Meeting type", [
            "Team meeting (all members discuss)",
            "Individual meeting (one expert + critic)",
        ], key="m_type")

        try:
            _presets = _load_team_presets()
            preset_labels = {k: f"{k} — {d}" for k, (d, _, _) in _presets.items()}
        except Exception:
            _presets = {}
            preset_labels = {"default": "default"}

        mteam = st.selectbox("Team", list(preset_labels.keys()),
                             format_func=lambda k: preset_labels.get(k, k), key="m_team")
        mrounds = st.slider("Discussion rounds", 1, 5, 1, key="m_rnd")

        m_expert_agent = None
        if "Individual" in mtype and _presets:
            try:
                _, _, _mems = _presets[mteam]
                noncrit = [m for m in _mems if m.title != "Scientific Critic"]
                if noncrit:
                    m_expert_agent = st.selectbox(
                        "Pick the expert", noncrit,
                        format_func=lambda a: f"{a.title}",
                        key="m_exp",
                    )
            except Exception:
                pass

    # team preview
    if _presets and mteam in _presets:
        _d, _ld, _ms = _presets[mteam]
        st.markdown("#### Selected Team")
        ncols = min(len(_ms) + 1, 5)
        tcols = st.columns(ncols)
        with tcols[0]:
            st.markdown(
                f'<div class="ac" style="border-left:3px solid #f59e0b">'
                f'<div style="font-weight:700;color:#f59e0b">👑 {_esc(_ld.title)}</div>'
                f'<div style="color:var(--pdhub-text-secondary);font-size:.8rem">Team Lead</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        for i, mem in enumerate(_ms):
            with tcols[(i + 1) % ncols]:
                st.markdown(
                    f'<div class="ac" style="border-left:3px solid #a855f7">'
                    f'<div style="font-weight:700;color:#a855f7">🧪 {_esc(mem.title)}</div>'
                    f'<div style="color:var(--pdhub-text-secondary);font-size:.8rem">{_short(mem.expertise, 60)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")
    bm, bmr, _ = st.columns([1, 1, 2])
    with bm:
        mdis = not agenda or not agenda.strip() or st.session_state.meet_running
        mbtn = st.button(
            "▶ Start Meeting" if not st.session_state.meet_running else "⏳ Running...",
            type="primary", use_container_width=True, disabled=mdis, key="btn_meet",
        )
    with bmr:
        if st.button("🗑 Clear", use_container_width=True, key="btn_mclr"):
            st.session_state.meet_result = None
            st.session_state.meet_transcript = None
            st.rerun()

    if mbtn and agenda and agenda.strip() and _presets:
        st.session_state.meet_running = True
        st.session_state.meet_result = None
        st.session_state.meet_transcript = None
        try:
            from protein_design_hub.agents.meeting import run_meeting
            _d, _ld, _ms = _presets[mteam]
            is_team = "Team" in mtype
            sd = Path("./outputs/meetings")
            sn = f"meeting_{int(time.time())}"
            with st.spinner("Running agent meeting... this may take a few minutes."):
                if is_team:
                    summary = run_meeting(
                        meeting_type="team", agenda=agenda, save_dir=sd, save_name=sn,
                        team_lead=_ld, team_members=_ms, num_rounds=mrounds, return_summary=True,
                    )
                else:
                    expert = m_expert_agent
                    if expert is None:
                        expert = next((m for m in _ms if m.title != "Scientific Critic"), _ms[0])
                    summary = run_meeting(
                        meeting_type="individual", agenda=agenda, save_dir=sd, save_name=sn,
                        team_member=expert, num_rounds=mrounds, return_summary=True,
                    )
            st.session_state.meet_result = summary
            tp = sd / f"{sn}.json"
            if tp.exists():
                with open(tp) as f:
                    st.session_state.meet_transcript = json.load(f)
            st.session_state.meet_running = False
            st.rerun()
        except Exception as e:
            st.session_state.meet_running = False
            st.error(f"Meeting failed: {e}")
            if "Connection" in str(e) or "refused" in str(e):
                info_box("Make sure Ollama is running: `ollama serve`", variant="warning", icon="⚠️")
            import traceback
            with st.expander("Show traceback"):
                st.code(traceback.format_exc(), language="text")

    if st.session_state.meet_result:
        section_header("Meeting Summary", "", "📝")
        st.markdown(st.session_state.meet_result)

    if st.session_state.meet_transcript:
        with st.expander("📜 Full Transcript", expanded=False):
            for turn in st.session_state.meet_transcript:
                an = turn.get("agent", "Unknown")
                msg = turn.get("message", "")
                isu = an == "User"
                if isu and len(msg) > 500:
                    msg = msg[:500] + "..."
                ac = "trn-u" if isu else "trn-ai"
                ic = "💬" if isu else "🧪"
                st.markdown(
                    f'<div class="trn">'
                    f'<div class="trn-a {ac}">{ic} {_esc(an)}</div>'
                    f'<div class="trn-b">{_esc(msg)}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


###############################################################################
# TAB 3 — AGENT EXPLORER
###############################################################################
with tabs[3]:
    section_header("Agent Explorer", "All 10 scientist agents and team configurations", "🔍")

    try:
        from protein_design_hub.agents import scientists as S

        _agents_list = [
            ("👑", S.PRINCIPAL_INVESTIGATOR, "Team Lead",  "#f59e0b"),
            ("🔬", S.STRUCTURAL_BIOLOGIST,   "Expert",     "#3b82f6"),
            ("💻", S.COMPUTATIONAL_BIOLOGIST, "Expert",     "#06b6d4"),
            ("🤖", S.MACHINE_LEARNING_SPECIALIST, "Expert","#8b5cf6"),
            ("🧬", S.IMMUNOLOGIST,           "Expert",     "#22c55e"),
            ("🔧", S.PROTEIN_ENGINEER,       "Expert",     "#f97316"),
            ("⚡", S.BIOPHYSICIST,           "Expert",     "#eab308"),
            ("🛠️", S.DIGITAL_RECEP,          "Refinement", "#ec4899"),
            ("📊", S.LIAM,                   "QA / ModFold","#14b8a6"),
            ("🎯", S.SCIENTIFIC_CRITIC,      "Critic",     "#ef4444"),
        ]

        for icon, agent, role, color in _agents_list:
            with st.container(border=True):
                ci, cd = st.columns([1, 3])
                with ci:
                    st.markdown(
                        f'<div style="text-align:center">'
                        f'<div style="font-size:2.2rem;margin-bottom:4px">{icon}</div>'
                        f'<div style="font-weight:700;font-size:1rem;color:{color}">{_esc(agent.title)}</div>'
                        f'<div style="font-size:.7rem;font-weight:600;text-transform:uppercase;'
                        f'letter-spacing:.05em;color:var(--pdhub-text-muted);margin-top:2px">{role}</div>'
                        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:.7rem;'
                        f'color:var(--pdhub-text-muted);margin-top:5px">{_esc(agent.resolved_model)}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with cd:
                    st.markdown(f"**Expertise:** {agent.expertise}")
                    st.markdown(f"**Goal:** {agent.goal}")
                    with st.expander("Full role description"):
                        st.markdown(agent.role)

        st.markdown("")
        section_header("Team Compositions", "Pre-built team configurations for different tasks", "👥")
        _td = [
            ("Default",       "General prediction pipeline",           S.DEFAULT_TEAM_LEAD, S.DEFAULT_TEAM_MEMBERS),
            ("Design",        "Protein design & engineering",           S.DEFAULT_TEAM_LEAD, S.DESIGN_TEAM_MEMBERS),
            ("Nanobody",      "Antibody / nanobody engineering",        S.DEFAULT_TEAM_LEAD, S.NANOBODY_TEAM_MEMBERS),
            ("Evaluation",    "Structure quality assessment",            S.DEFAULT_TEAM_LEAD, S.EVALUATION_TEAM_MEMBERS),
            ("Refinement",    "Structure refinement workflow",           S.DEFAULT_TEAM_LEAD, S.REFINEMENT_TEAM_MEMBERS),
            ("Mutagenesis",   "Mutation scanning & sequence design",     S.DEFAULT_TEAM_LEAD, S.MUTAGENESIS_TEAM_MEMBERS),
            ("MPNN Design",   "ProteinMPNN inverse folding",             S.DEFAULT_TEAM_LEAD, S.MPNN_DESIGN_TEAM_MEMBERS),
            ("Full Pipeline", "End-to-end pipeline review (all experts)", S.DEFAULT_TEAM_LEAD, S.FULL_PIPELINE_TEAM_MEMBERS),
            ("All Experts",   "Comprehensive review with all scientist personas", S.DEFAULT_TEAM_LEAD, S.ALL_EXPERTS_TEAM_MEMBERS),
        ]
        for name, desc, lead, members in _td:
            with st.container(border=True):
                st.markdown(f"**{name}** — {desc}")
                st.caption(f"👑 {lead.title} (lead) • " + " • ".join(f"🧪 {m.title}" for m in members))

    except Exception as e:
        st.error(f"Could not load agents: {e}")


###############################################################################
# TAB 4 — LLM STATUS
###############################################################################
with tabs[4]:
    section_header("LLM Backend Status", "Configuration, connectivity, and provider presets", "⚙️")

    cr, _ = st.columns([1, 4])
    with cr:
        if st.button("🔄 Refresh Status", key="btn_ref_llm", use_container_width=True):
            _check_llm.clear()
            st.rerun()

    try:
        from protein_design_hub.core.config import get_settings, LLM_PROVIDER_PRESETS
        cfg = get_settings().llm.resolve()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card(cfg.provider, "Provider", "info", "🔌")
        with c2:
            metric_card(cfg.model, "Model", "info", "🤖")
        with c3:
            metric_card(str(cfg.temperature), "Temperature", "default", "🌡️")
        with c4:
            metric_card(str(cfg.max_tokens), "Max Tokens", "default", "📏")

        st.markdown("")

        with st.container(border=True):
            st.markdown("#### Current Configuration")
            for k, v in {
                "Provider": cfg.provider, "Base URL": cfg.base_url,
                "Model": cfg.model,
                "API Key": (cfg.api_key[:8] + "...") if len(cfg.api_key) > 8 else cfg.api_key,
                "Temperature": str(cfg.temperature), "Max Tokens": str(cfg.max_tokens),
            }.items():
                ck, cv = st.columns([1, 3])
                with ck:
                    st.markdown(f"**{k}**")
                with cv:
                    st.code(v, language=None)

        st.markdown("")
        section_header("Connectivity Test", "Click to test connection", "🔗")

        ct_btn = st.button("🔗 Test Connection", key="btn_conn_test", type="primary")
        if ct_btn:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
                with st.spinner("Connecting to LLM backend..."):
                    models = client.models.list()
                model_ids = sorted(m.id for m in models.data)
                st.session_state["_llm_models"] = model_ids
                st.session_state["_llm_connected"] = True
            except Exception as e:
                st.session_state["_llm_connected"] = False
                st.session_state["_llm_conn_error"] = str(e)

        if st.session_state.get("_llm_connected"):
            model_ids = st.session_state.get("_llm_models", [])
            if cfg.model in model_ids:
                st.success(f"Connected — model `{cfg.model}` is available")
            else:
                st.warning(f"Connected but model `{cfg.model}` not in listed models")
            with st.expander(f"Available models ({len(model_ids)})"):
                for mid in model_ids:
                    act = " **← active**" if mid == cfg.model else ""
                    st.markdown(f"- `{mid}`{act}")
        elif st.session_state.get("_llm_connected") is False:
            st.error(f"Connection failed: {st.session_state.get('_llm_conn_error', 'Unknown error')}")
            if cfg.provider == "ollama":
                info_box("Make sure Ollama is running: `ollama serve`", variant="warning", icon="💡")

        st.markdown("")
        section_header("Quick LLM Test", "Send a test prompt to verify everything works", "🧪")
        tp = st.text_input("Test prompt", value="Explain protein folding in 2 sentences.", key="llm_tp")
        if st.button("Send Test", key="btn_test", type="primary"):
            with st.spinner("Calling LLM..."):
                try:
                    from openai import OpenAI
                    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
                    resp = client.chat.completions.create(
                        model=cfg.model,
                        messages=[{"role": "user", "content": tp}],
                        temperature=cfg.temperature, max_tokens=256,
                    )
                    reply = resp.choices[0].message.content or "(empty)"
                    st.success("Response received!")
                    st.markdown(reply)
                except Exception as e2:
                    st.error(f"LLM call failed: {e2}")
                    if cfg.provider == "ollama":
                        info_box("Make sure Ollama is running: `ollama serve`", variant="warning", icon="💡")

        st.markdown("")
        section_header("All Providers", "", "🌐")
        local_p, cloud_p = [], []
        for name, (url, model, _) in LLM_PROVIDER_PRESETS.items():
            is_loc = name in ("ollama", "lmstudio", "vllm", "llamacpp")
            entry = (name, url, model, name == cfg.provider)
            (local_p if is_loc else cloud_p).append(entry)

        cl, cc = st.columns(2)
        with cl:
            st.markdown("##### Local (no API key needed)")
            for name, url, model, active in local_p:
                cls = "pvc pvc-on" if active else "pvc"
                ab = " ✓" if active else ""
                st.markdown(
                    f'<div class="{cls}">'
                    f'<div style="font-weight:700">{name}{ab}</div>'
                    f'<div style="font-family:monospace;font-size:.78rem;color:var(--pdhub-text-secondary)">{model}</div>'
                    f'<div style="font-size:.67rem;font-weight:600;text-transform:uppercase;color:#22c55e">LOCAL</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        with cc:
            st.markdown("##### Cloud (API key required)")
            for name, url, model, active in cloud_p:
                cls = "pvc pvc-on" if active else "pvc"
                ab = " ✓" if active else ""
                st.markdown(
                    f'<div class="{cls}">'
                    f'<div style="font-weight:700">{name}{ab}</div>'
                    f'<div style="font-family:monospace;font-size:.78rem;color:var(--pdhub-text-secondary)">{model}</div>'
                    f'<div style="font-size:.67rem;font-weight:600;text-transform:uppercase;color:#f59e0b">CLOUD</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"Could not load LLM configuration: {e}")
        import traceback
        st.code(traceback.format_exc(), language="text")


###############################################################################
# TAB 5 — MEETING HISTORY
###############################################################################
with tabs[5]:
    section_header("Meeting History", "Browse saved meeting transcripts", "📂")

    cr2, _ = st.columns([1, 4])
    with cr2:
        if st.button("🔄 Refresh", key="btn_ref_hist", use_container_width=True):
            st.rerun()

    try:
        from protein_design_hub.core.config import get_settings
        base_dir = Path(get_settings().output.base_dir)

        meeting_dirs = []
        gmd = Path("./outputs/meetings")
        if gmd.exists():
            meeting_dirs.append(("Global", gmd))
        if base_dir.exists():
            for jd in sorted(base_dir.iterdir(), reverse=True):
                md = jd / "meetings"
                if md.exists() and md.is_dir():
                    meeting_dirs.append((jd.name, md))

        if not meeting_dirs:
            empty_state(
                "No meetings yet",
                "Run a pipeline with LLM mode or start an ad-hoc meeting first.",
                "📭",
            )
        else:
            total = 0
            for jn, md in meeting_dirs:
                jfs = sorted(md.glob("*.json"), reverse=True)
                if not jfs:
                    continue
                total += len(jfs)
                with st.expander(f"📁 {jn} ({len(jfs)} meeting(s))", expanded=len(meeting_dirs) == 1):
                    for jf in jfs:
                        with st.container(border=True):
                            st.markdown(f"**{jf.stem}**")
                            try:
                                with open(jf) as f:
                                    tr = json.load(f)
                                if tr:
                                    agents_in = sorted(set(
                                        t.get("agent") for t in tr if t.get("agent") != "User"
                                    ))
                                    st.caption(
                                        f"Agents: {', '.join(agents_in)} | "
                                        f"Turns: {len(tr)}"
                                    )
                            except Exception:
                                st.caption("Could not read transcript")

                            mdp = jf.with_suffix(".md")
                            if mdp.exists():
                                mdc = mdp.read_text()
                                st.download_button(
                                    "📥 Download .md", mdc,
                                    file_name=f"{jf.stem}.md", mime="text/markdown",
                                    key=f"dl_{jn}_{jf.stem}", use_container_width=True,
                                )

                            with st.expander("View transcript"):
                                try:
                                    with open(jf) as f:
                                        tr = json.load(f)
                                    for turn in tr:
                                        an = turn.get("agent", "Unknown")
                                        msg = turn.get("message", "")
                                        isu = an == "User"
                                        if isu and len(msg) > 500:
                                            msg = msg[:500] + "..."
                                        ac = "trn-u" if isu else "trn-ai"
                                        ic = "💬" if isu else "🧪"
                                        st.markdown(
                                            f'<div class="trn">'
                                            f'<div class="trn-a {ac}">{ic} {_esc(an)}</div>'
                                            f'<div class="trn-b">{_esc(msg)}</div>'
                                            f'</div>',
                                            unsafe_allow_html=True,
                                        )
                                except Exception as e:
                                    st.error(f"Error: {e}")

            if total:
                st.caption(f"Total: {total} meeting(s)")

    except Exception as e:
        st.error(f"Could not scan meetings: {e}")

# footer
st.markdown("")
st.markdown(
    '<div style="text-align:center;color:var(--pdhub-text-muted);font-size:.73rem;padding:2rem 0">'
    'Protein Design Hub &bull; 10 scientist agents &bull; 8 team configs &bull; 10-step LLM pipeline'
    '</div>',
    unsafe_allow_html=True,
)
