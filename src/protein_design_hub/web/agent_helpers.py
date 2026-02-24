"""Shared agent-integration helpers for Streamlit pages.

Provides lightweight wrappers so every page can offer:
  * Quick LLM advice (single-turn call with a domain expert)
  * Multi-turn chatbot conversation with any agent
  * Agent status badge (cached, non-blocking)
"""

from __future__ import annotations

from contextlib import contextmanager
import html as _html
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
from protein_design_hub.agents.ollama_gpu import ensure_ollama_gpu, ollama_extra_body

# Cross-page context (lazy import to avoid circular imports at module load)
def _get_cross_page_context() -> str:
    try:
        from protein_design_hub.web.shared_context import get_all_context_summary
        return get_all_context_summary()
    except Exception:
        return ""

# ── LLM connectivity (cached to avoid blocking every page load) ──────

@st.cache_data(ttl=60, show_spinner=False)
def _cached_llm_status() -> tuple:
    """Return (ok: bool, label: str, color: str).  Cached for 60 s.

    Uses a short timeout to avoid blocking page loads when LLM is down.
    """
    try:
        from protein_design_hub.core.config import get_settings
        cfg = get_settings().llm.resolve()
    except Exception:
        return False, "LLM: not configured", "#ef4444"
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=5.0,  # 5s timeout to avoid blocking UI
        )
        client.models.list()
        return True, f"LLM: {cfg.provider} / {cfg.model}", "#22c55e"
    except Exception:
        return False, f"LLM: offline ({cfg.provider})", "#f59e0b"


def _get_llm_cfg():
    try:
        from protein_design_hub.core.config import get_settings
        return get_settings().llm.resolve()
    except Exception:
        return None


def llm_available() -> bool:
    ok, _, _ = _cached_llm_status()
    return ok


# ── Model listing and switching ──────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def list_available_models() -> list[str]:
    """Return model IDs from the active LLM backend.  Cached 30 s."""
    cfg = _get_llm_cfg()
    if cfg is None:
        return []
    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=5.0)
        models = client.models.list()
        ids = sorted(m.id for m in models.data)
        # Normalize legacy ollama model names so "llama3.2" shows as "qwen2.5:14b"
        if cfg.provider == "ollama":
            try:
                from protein_design_hub.core.config import normalize_ollama_model_name
                ids = sorted(set(normalize_ollama_model_name(m) for m in ids))
            except Exception:
                pass
        return ids
    except Exception:
        return []


def switch_llm_model(model: str) -> None:
    """Switch the active LLM model in-memory (applies to all subsequent calls)."""
    try:
        from protein_design_hub.core.config import get_settings, normalize_ollama_model_name
        from protein_design_hub.agents.meeting import reset_llm_client
        settings = get_settings()
        if settings.llm.provider == "ollama":
            model = normalize_ollama_model_name(model)
        settings.llm.model = model
        reset_llm_client()
        # Invalidate caches so the UI picks up the change
        _cached_llm_status.clear()
        list_available_models.clear()
    except Exception:
        pass


def switch_llm_provider(provider: str, model: str = "") -> None:
    """Switch LLM provider and optionally model.  Resets client caches."""
    try:
        from protein_design_hub.core.config import get_settings, normalize_ollama_model_name
        from protein_design_hub.agents.meeting import reset_llm_client
        settings = get_settings()
        settings.llm.provider = provider
        settings.llm.base_url = ""
        settings.llm.api_key = ""
        if provider == "ollama":
            model = normalize_ollama_model_name(model)
        settings.llm.model = model
        reset_llm_client()
        _cached_llm_status.clear()
        list_available_models.clear()
    except Exception:
        pass


def render_model_switcher(key_prefix: str = "model_sw") -> None:
    """Compact model switcher widget: provider selector + model dropdown."""
    cfg = _get_llm_cfg()
    if cfg is None:
        st.warning("LLM not configured")
        return

    try:
        from protein_design_hub.core.config import LLM_PROVIDER_PRESETS
        prov_names = list(LLM_PROVIDER_PRESETS.keys())
    except Exception:
        prov_names = ["ollama"]

    col_prov, col_model, col_apply = st.columns([1, 2, 1])

    with col_prov:
        ci = prov_names.index(cfg.provider) if cfg.provider in prov_names else 0
        new_prov = st.selectbox(
            "Provider", prov_names, index=ci, key=f"{key_prefix}_prov",
        )

    with col_model:
        available = list_available_models()
        # Ensure current model is in the list
        if cfg.model and cfg.model not in available:
            available = [cfg.model] + available
        if not available:
            available = [cfg.model or "unknown"]
        mi = available.index(cfg.model) if cfg.model in available else 0
        new_model = st.selectbox(
            "Model", available, index=mi, key=f"{key_prefix}_model",
        )

    with col_apply:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Apply", key=f"{key_prefix}_apply", type="primary",
                      use_container_width=True):
            if new_prov != cfg.provider:
                switch_llm_provider(new_prov, new_model)
            elif new_model != cfg.model:
                switch_llm_model(new_model)
            st.rerun()


# ── Single-turn quick advice ─────────────────────────────────────────

_AGENT_MAP = None

def _resolve_agent_map():
    global _AGENT_MAP
    if _AGENT_MAP is not None:
        return _AGENT_MAP
    try:
        from protein_design_hub.agents import scientists as S
        _AGENT_MAP = {
            "Structural Biologist": S.STRUCTURAL_BIOLOGIST,
            "Computational Biologist": S.COMPUTATIONAL_BIOLOGIST,
            "Machine Learning Specialist": S.MACHINE_LEARNING_SPECIALIST,
            "Protein Engineer": S.PROTEIN_ENGINEER,
            "Biophysicist": S.BIOPHYSICIST,
            "Digital Recep": S.DIGITAL_RECEP,
            "Liam": S.LIAM,
            "Immunologist": S.IMMUNOLOGIST,
            "Scientific Critic": S.SCIENTIFIC_CRITIC,
            "Principal Investigator": S.PRINCIPAL_INVESTIGATOR,
        }
    except Exception:
        _AGENT_MAP = {}
    return _AGENT_MAP


AGENT_OPTIONS = [
    "Structural Biologist",
    "Computational Biologist",
    "Machine Learning Specialist",
    "Protein Engineer",
    "Biophysicist",
    "Digital Recep",
    "Liam",
    "Immunologist",
    "Scientific Critic",
    "Principal Investigator",
]


def ask_agent_advice(
    question: str,
    agent_name: str = "Structural Biologist",
    context: str = "",
    max_tokens: int = 512,
) -> str:
    """Single-turn LLM call with a domain-expert system prompt."""
    cfg = _get_llm_cfg()
    if cfg is None:
        return "[Error] LLM not configured. Go to Agents > LLM Status to configure."

    agents = _resolve_agent_map()
    agent = agents.get(agent_name)
    if agent is not None:
        sys_msg = agent.system_message
    else:
        sys_msg = {"role": "system", "content": f"You are an expert {agent_name}."}

    user_content = question
    if context:
        user_content = f"Context:\n{context}\n\nQuestion:\n{question}"

    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        ensure_ollama_gpu(cfg.provider, cfg.model)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[sys_msg, {"role": "user", "content": user_content}],
            temperature=cfg.temperature,
            max_tokens=max_tokens,
            **ollama_extra_body(cfg.provider),
        )
        ensure_ollama_gpu(cfg.provider, cfg.model)
        return resp.choices[0].message.content or "(empty response)"
    except Exception as e:
        return f"[Error] LLM call failed: {e}"


# ── Streamlit UI components ──────────────────────────────────────────

def agent_sidebar_status() -> None:
    """Compact LLM status badge in sidebar. Cached, non-blocking."""
    ok, label, color = _cached_llm_status()
    st.sidebar.markdown(
        f'<div style="font-size:.78rem;color:{color};display:flex;align-items:center;'
        f'gap:6px;padding:2px 0">'
        f'<span style="width:7px;height:7px;background:{color};border-radius:50%;'
        f'flex-shrink:0"></span>{_html.escape(label)}</div>',
        unsafe_allow_html=True,
    )
    # Sidebar quick-chat
    _render_sidebar_chat()


def _render_sidebar_chat() -> None:
    """Minimal sidebar chat — one question, one answer, always visible."""
    sb_key = "_sb_chat"
    reply_key = "_sb_chat_reply"

    with st.sidebar.expander("💬 Quick Chat", expanded=False):
        agent = st.selectbox(
            "Expert",
            AGENT_OPTIONS[:6],  # Show most common agents
            index=0,
            key=f"{sb_key}_agent",
            label_visibility="collapsed",
        )
        q = st.text_input(
            "Ask",
            placeholder="Quick question...",
            key=f"{sb_key}_q",
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([3, 1])
        with c1:
            ask = st.button("Ask", key=f"{sb_key}_btn", use_container_width=True, type="primary")
        with c2:
            if st.button("X", key=f"{sb_key}_clr", use_container_width=True):
                st.session_state.pop(reply_key, None)
                st.rerun()

        if ask and q and q.strip():
            with st.spinner("..."):
                reply = ask_agent_advice(q.strip(), agent_name=agent, max_tokens=300)
            st.session_state[reply_key] = {"agent": agent, "text": reply}

        stored = st.session_state.get(reply_key)
        if stored:
            if stored["text"].startswith("[Error]"):
                st.error(stored["text"])
            else:
                st.markdown(
                    f'<div style="font-size:.8rem;color:var(--pdhub-accent,#a78bfa);font-weight:600;'
                    f'margin-bottom:3px">{_html.escape(stored["agent"])}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(stored["text"])


def render_agent_advice_panel(
    page_context: str,
    default_question: str = "",
    expert: str = "Structural Biologist",
    key_prefix: str = "agent",
    tool_data: Optional[dict] = None,
) -> None:
    """Professional 'Consult Expert' panel with styled agent response cards.

    Args:
        tool_data: Optional structured data for tool pre-execution. Recognised keys:
            ``sequences``: List[Tuple[str,str]] — (name, seq) pairs for biophysics
            ``pdb_path``: Path — for structure analysis
            ``records``: list of dicts — for ML suite
            ``numeric_keys``: List[str] (optional, used with records)
            ``target_key``: str (optional, used with records)
    """
    reply_key = f"_agent_reply_{key_prefix}"
    adv_tool_key = f"_agent_tools_{key_prefix}"
    icon = AGENT_ICONS.get(expert, "🧬")
    color = AGENT_COLORS.get(expert, "#6366f1")

    with st.expander(f"{icon} Consult Domain Expert", expanded=False):
        col_q, col_e = st.columns([3, 1])
        with col_q:
            question = st.text_area(
                "Your question",
                value=default_question,
                height=80,
                placeholder="e.g. How can I improve the pLDDT score? What mutations should I prioritize?",
                key=f"{key_prefix}_q",
                label_visibility="collapsed",
            )
        with col_e:
            agent_choice = st.selectbox(
                "Expert",
                AGENT_OPTIONS,
                index=AGENT_OPTIONS.index(expert) if expert in AGENT_OPTIONS else 0,
                key=f"{key_prefix}_expert",
                label_visibility="collapsed",
            )
            _acolor = AGENT_COLORS.get(agent_choice, "#6366f1")
            _arole = AGENT_ROLES.get(agent_choice, "")
            st.markdown(
                f'<div style="font-size:.7rem;color:{_acolor};padding:2px 0">'
                f'{AGENT_ICONS.get(agent_choice,"🧬")} {_html.escape(_arole)}</div>',
                unsafe_allow_html=True,
            )

        deep_mode = st.checkbox(
            "🔍 Deep analysis (longer, step-by-step reasoning)",
            value=False,
            key=f"{key_prefix}_deep",
            help="Extends response length and enables chain-of-thought reasoning. ~2× slower.",
        )

        col_btn, col_clear = st.columns([3, 1])
        with col_btn:
            clicked = st.button(
                f"💬 Consult {agent_choice}",
                key=f"{key_prefix}_btn",
                use_container_width=True,
                type="primary",
            )
        with col_clear:
            if st.button("↺ Clear", key=f"{key_prefix}_clr", use_container_width=True):
                st.session_state.pop(reply_key, None)
                st.session_state.pop(adv_tool_key, None)
                st.rerun()

        if clicked:
            if not question or not question.strip():
                st.warning("Please type a question first.")
            else:
                max_tok = 1200 if deep_mode else 600
                q = question.strip()
                if deep_mode:
                    q = (
                        "Think step by step. Provide a thorough, structured analysis. "
                        "Use numbered sections where appropriate.\n\n" + q
                    )

                # ── Tool pre-execution (agent-type aware) ──────────────────
                tool_context_addition = ""
                if tool_data:
                    try:
                        from protein_design_hub.web.agent_tools import (
                            run_sequence_biophysics, run_structure_analysis,
                            run_ml_tool_suite, AgentToolReport,
                        )
                        _bio_agents = {"Protein Engineer", "Biophysicist"}
                        _struct_agents = {"Structural Biologist", "Digital Recep", "Liam"}
                        _ml_agents = {"Machine Learning Specialist"}

                        adv_report: Optional[AgentToolReport] = None
                        if agent_choice in _bio_agents and "sequences" in tool_data:
                            with st.spinner("Running biophysical analysis…"):
                                adv_report = AgentToolReport()
                                adv_report.tool_results.append(
                                    run_sequence_biophysics(tool_data["sequences"])
                                )
                        elif agent_choice in _struct_agents and "pdb_path" in tool_data:
                            with st.spinner("Analyzing structure…"):
                                adv_report = AgentToolReport()
                                adv_report.tool_results.append(
                                    run_structure_analysis(tool_data["pdb_path"])
                                )
                        elif agent_choice in _ml_agents and "records" in tool_data:
                            with st.spinner("Running ML tools…"):
                                adv_report = run_ml_tool_suite(
                                    tool_data["records"],
                                    tool_data.get("numeric_keys"),
                                    tool_data.get("target_key"),
                                )

                        if adv_report and adv_report.any_success:
                            st.session_state[adv_tool_key] = adv_report
                            tool_context_addition = (
                                "\n\nTool analysis results:\n" + adv_report.context_string
                            )
                    except Exception:
                        pass

                full_context = page_context + tool_context_addition
                with st.spinner(f"Consulting {agent_choice}…"):
                    reply = ask_agent_advice(
                        question=q,
                        agent_name=agent_choice,
                        context=full_context,
                        max_tokens=max_tok,
                    )
                st.session_state[reply_key] = {"agent": agent_choice, "text": reply}

        stored = st.session_state.get(reply_key)
        if stored:
            if stored["text"].startswith("[Error]"):
                st.error(stored["text"])
                st.info("Make sure Ollama is running (`ollama serve`) or configure LLM on the Agents page.")
            else:
                # Show tool computation results first (if any)
                adv_tool_report = st.session_state.get(adv_tool_key)
                if adv_tool_report:
                    try:
                        from protein_design_hub.web.agent_tools import render_agent_tool_report
                        render_agent_tool_report(adv_tool_report)
                    except Exception:
                        pass
                _render_agent_response_card(stored["agent"], stored["text"])

        st.caption("🔌 Configure LLM backend · Agents page")


def render_contextual_insight(
    page_name: str,
    data_dict: dict,
    key_prefix: str = "ctx_insight",
) -> None:
    """One-click AI analysis button that auto-generates a question from page data.

    Args:
        page_name: Name of the current page (e.g. "Prediction", "Evaluation")
        data_dict: Dict of metric names -> values for context
        key_prefix: Unique key prefix
    """
    reply_key = f"_ctx_reply_{key_prefix}"

    if not data_dict:
        return

    # Build context string
    ctx_lines = [f"- {k}: {v}" for k, v in data_dict.items() if v is not None]
    if not ctx_lines:
        return

    context = f"{page_name} results:\n" + "\n".join(ctx_lines)

    # Append cross-page workflow context so agents see the full picture
    cross_ctx = _get_cross_page_context()
    if cross_ctx and "No cross-page" not in cross_ctx:
        context = context + "\n\n" + cross_ctx

    # Auto-generated question based on page type
    questions = {
        "Prediction": (
            "Assess the quality of this structure prediction. Interpret pLDDT "
            "(>90=excellent, 70-90=good, <70=uncertain), pTM (>0.8=confident fold), "
            "and iPTM (for complexes). Are there regions of low confidence that "
            "indicate disorder (use DISOclust for validation) or prediction failure? "
            "Should the user re-predict with a different method (e.g. switch from "
            "ESMFold to ColabFold for better MSA coverage, or use IntFOLD7 for "
            "integrated prediction+QA, or MultiFOLD2 for multimers with stoichiometry)? "
            "Recommend independent QA with ModFOLD9 (p-value < 0.001 = high confidence). "
            "For multimer interfaces, suggest ModFOLDdock2 validation."
        ),
        "Evaluation": (
            "Interpret these evaluation metrics in the context of protein quality. "
            "Assess structural quality (clash score <10=excellent per MolProbity, "
            "Ramachandran >98%=good), fold accuracy (TM-score, RMSD if reference "
            "available), and energy scores. Cross-validate with ModFOLD9 global "
            "score and p-values for independent QA. Is this structure suitable for "
            "downstream applications (docking requires pLDDT>80 at binding site, "
            "design requires stable core)? If quality is borderline, recommend "
            "ReFOLD3 quality-guided refinement using ModFOLD per-residue scores "
            "to target specific low-confidence regions."
        ),
        "Compare": (
            "Compare these predictor results using rigorous criteria. Which produced "
            "the best structure based on: (1) confidence metrics (pLDDT, pTM), "
            "(2) structural quality (clash score, Ramachandran), (3) fold accuracy "
            "(if reference available: TM-score, GDT-TS, RMSD). Recommend independent "
            "ranking with ModFOLD9 (global score + p-value). Consider: ESMFold is "
            "fast but may miss MSA contacts; ColabFold best for conserved folds; "
            "MultiFOLD2 for multimers (CASP16 top); diffusion models (Chai-1, Boltz-2) "
            "for complexes. For multimer interfaces, validate with ModFOLDdock2."
        ),
        "Mutation": (
            "Analyze these mutation scanning results. For each position: is it "
            "conserved (risky to mutate) or variable (safer)? For stabilising mutations: "
            "does the ddG prediction account for local structural context (buried vs "
            "surface, hydrogen bonding network, packing)? Which substitutions maintain "
            "fold integrity (check predicted pLDDT of mutant)? Use ModFOLD9 per-residue "
            "scores to identify regions where mutations are most/least risky. "
            "Flag any mutations that might disrupt active site (check FunFOLD binding "
            "site predictions), binding interface, or disulfide bonds."
        ),
        "MPNN": (
            "Evaluate these ProteinMPNN-designed sequences. Assess: (1) sequence "
            "recovery at key positions (active site, interface, core), (2) overall "
            "log-likelihood score, (3) predicted biophysical properties (GRAVY, charge, "
            "instability index). Recommend self-consistency validation: re-predict the "
            "designed sequence with ESMFold/ColabFold and check TM-score > 0.9 to the "
            "design template. Validate with ModFOLD9 QA. For functional positions, "
            "cross-reference with FunFOLD5 binding site predictions. Flag any designs "
            "with unusual amino acid composition."
        ),
        "Refine": (
            "Evaluate these refinement results. Compare before/after metrics: "
            "clash score, Ramachandran statistics, ModFOLD9 global score, and "
            "GDT-TS to reference (if available). Did refinement improve local "
            "quality without distorting the global fold? Consider: ReFOLD3 is "
            "ideal for quality-guided refinement (uses ModFOLD per-residue scores "
            "as restraints); AMBER relaxation for quick stereochemical cleanup; "
            "MultiFOLD_refine for iterative AF2-recycling refinement."
        ),
        "Evolution": (
            "Assess the directed evolution results. Is the fitness improvement "
            "statistically significant or likely noise/overfitting? Examine the "
            "fitness trajectory: does it plateau (convergence) or still climb "
            "(more generations needed)? For each mutation in the best variant, "
            "assess whether it is likely causal (conserved site, known stabilizing "
            "substitution) or incidental (neutral hitchhiker). Recommend validation "
            "strategy: re-predict the evolved sequence with ESMFold/ColabFold, "
            "check TM-score to parent, and validate with ModFOLD9 QA."
        ),
        "MSA": (
            "Interpret the MSA conservation and coevolution signals. Which "
            "positions are highly conserved (>0.8 conservation score) and should "
            "NOT be mutated? Which variable positions are safe targets for "
            "engineering? Cross-reference PSSM-derived mutation suggestions with "
            "structural context: are suggested mutations at surface or core "
            "positions? Use coevolution pairs to identify compensatory mutation "
            "opportunities. Recommend a shortlist of safe, high-impact mutations "
            "prioritized by conservation, PSSM improvement, and structural context."
        ),
        "Design": (
            "Review the current protein design. Assess the sequence edits made "
            "so far: do the mutations maintain fold stability (check hydrophobic "
            "core packing, hydrogen bond networks, salt bridges)? Are any edits "
            "at conserved positions that could disrupt function? Evaluate ligand "
            "attachments for chemical compatibility with nearby residues. "
            "Recommend next design actions: additional stabilizing mutations, "
            "positions to avoid, and validation steps (re-predict structure, "
            "check pLDDT, evaluate with ModFOLD9)."
        ),
        "Batch": (
            "Analyze the batch processing results. Assess the success/failure "
            "rate: is the failure pattern random (network/timeout issues) or "
            "systematic (specific sequences failing, suggesting input problems)? "
            "For completed jobs, identify outliers in quality metrics. Recommend "
            "which sequences to prioritize for downstream analysis based on "
            "prediction confidence, biophysical properties, and overall quality. "
            "Flag any sequences that should be re-run with different settings."
        ),
        "Jobs": (
            "Review this job's artifacts and output quality. Does the job have "
            "all expected output files (structures, scores, summaries)? Are the "
            "prediction confidence metrics sufficient for downstream use? "
            "Recommend the best next workflow step: evaluate, refine, design, "
            "or re-run with different parameters. Flag any missing artifacts "
            "or quality concerns that should be addressed first."
        ),
    }
    # Find best matching question
    auto_q = "Provide a scientific interpretation of these results and recommend next steps."
    for key, q in questions.items():
        if key.lower() in page_name.lower():
            auto_q = q
            break

    # Pick best expert for this page type
    expert_map = {
        "Prediction": "Machine Learning Specialist",
        "Evaluation": "Liam",
        "Compare": "Computational Biologist",
        "Mutation": "Protein Engineer",
        "MPNN": "Machine Learning Specialist",
        "Refine": "Digital Recep",
        "Evolution": "Protein Engineer",
        "MSA": "Computational Biologist",
        "Design": "Protein Engineer",
        "Batch": "Computational Biologist",
        "Jobs": "Computational Biologist",
    }
    expert = "Structural Biologist"
    for key, exp in expert_map.items():
        if key.lower() in page_name.lower():
            expert = exp
            break

    _exp_icon = AGENT_ICONS.get(expert, "🔬")
    _exp_color = AGENT_COLORS.get(expert, "#6366f1")
    with st.expander(f"{_exp_icon} Scientific Analysis — {expert}", expanded=False):
        st.markdown(
            f'<div style="font-size:.78rem;color:{_exp_color};margin-bottom:8px">'
            f'{_html.escape(AGENT_ROLES.get(expert, ""))}</div>',
            unsafe_allow_html=True,
        )
        col_btn, col_deep, col_clr = st.columns([2, 2, 1])
        with col_btn:
            clicked = st.button(
                f"🧠 Analyze with {expert}",
                key=f"{key_prefix}_btn",
                use_container_width=True,
                type="primary",
            )
        with col_deep:
            deep_mode = st.checkbox(
                "🔍 Deep analysis",
                key=f"{key_prefix}_deep",
                help="Step-by-step reasoning, ~2× longer output",
            )
        with col_clr:
            if st.button("↺", key=f"{key_prefix}_clr", use_container_width=True, help="Clear result"):
                st.session_state.pop(reply_key, None)
                st.rerun()

        if clicked:
            max_tok = 1000 if deep_mode else 600
            q = auto_q
            if deep_mode:
                q = "Think step by step. Structure your response with numbered sections.\n\n" + q

            # ── Auto-detect and run tools from session state ───────────
            tool_ctx_extra = ""
            try:
                from protein_design_hub.web.agent_tools import (
                    run_sequence_biophysics, run_structure_analysis,
                )
                _bio_agents = {"Protein Engineer", "Biophysicist"}
                _struct_agents = {"Structural Biologist", "Digital Recep", "Liam",
                                  "Computational Biologist"}

                if expert in _bio_agents:
                    # Gather (name, sequence) tuples from common session state keys
                    seqs: list = []
                    for _sk in ["design_sequences", "mpnn_sequences", "evolved_sequences",
                                "mutation_sequences", "msa_sequences"]:
                        _v = st.session_state.get(_sk)
                        if isinstance(_v, list):
                            for item in _v:
                                if isinstance(item, (tuple, list)) and len(item) >= 2:
                                    seqs.append((str(item[0]), str(item[1])))
                                elif isinstance(item, dict):
                                    n = item.get("name", "seq")
                                    s = item.get("sequence") or item.get("seq", "")
                                    if s:
                                        seqs.append((n, s))
                    if seqs:
                        with st.spinner("Running biophysical analysis…"):
                            _r = run_sequence_biophysics(seqs[:20])
                            if _r.success:
                                tool_ctx_extra = "\n\nBiophysical analysis:\n" + _r.context_text

                elif expert in _struct_agents:
                    # Look for a PDB in the active job directory
                    import glob as _glob
                    job_dir = (st.session_state.get("active_job_dir") or
                               st.session_state.get("mutagenesis_job_dir") or "")
                    if job_dir:
                        pdbs = _glob.glob(f"{job_dir}/**/*.pdb", recursive=True)[:1]
                        if pdbs:
                            with st.spinner("Analyzing structure…"):
                                _r = run_structure_analysis(Path(pdbs[0]))
                                if _r.success:
                                    tool_ctx_extra = "\n\nStructure analysis:\n" + _r.context_text
            except Exception:
                pass

            full_context = context + tool_ctx_extra
            with st.spinner(f"Consulting {expert}…"):
                reply = ask_agent_advice(
                    question=q,
                    agent_name=expert,
                    context=full_context,
                    max_tokens=max_tok,
                )
            st.session_state[reply_key] = {"agent": expert, "text": reply}

        stored = st.session_state.get(reply_key)
        if stored:
            if stored["text"].startswith("[Error]"):
                st.error(stored["text"])
            else:
                _render_agent_response_card(stored["agent"], stored["text"])


def render_ml_stats_panel(
    records: list,
    numeric_keys: Optional[List[str]] = None,
    target_key: Optional[str] = None,
    page_name: str = "Analysis",
    key_prefix: str = "ml_stats",
    expanded: bool = False,
) -> None:
    """
    Combined statistical analysis + ML expert interpretation panel.

    Renders:
    - Descriptive stats, correlation matrix, feature importance, distributions
      (via ``stats_panel.render_stats_from_records``)
    - AI interpretation by the Machine Learning Specialist + cross-page context
    """
    if not records:
        return

    # ── Stats visuals ─────────────────────────────────────────────────
    try:
        from protein_design_hub.web.stats_panel import render_stats_from_records
        render_stats_from_records(
            records,
            numeric_keys=numeric_keys,
            target_key=target_key,
            title=f"{page_name} — Statistical Analysis",
            key_prefix=key_prefix,
            expanded=expanded,
        )
    except Exception as e:
        st.caption(f"Stats panel unavailable: {e}")

    # ── ML expert analysis button ─────────────────────────────────────
    reply_key = f"_ml_stats_reply_{key_prefix}"
    tool_report_key = f"_ml_stats_tools_{key_prefix}"
    _ml_color = AGENT_COLORS["Machine Learning Specialist"]
    with st.expander(f"🤖 ML Specialist — {page_name}", expanded=False):
        st.markdown(
            f'<div style="font-size:.78rem;color:{_ml_color};margin-bottom:8px">'
            f'Feature engineering · regression · statistical patterns · anomaly detection</div>',
            unsafe_allow_html=True,
        )
        col_btn, col_deep, col_clr = st.columns([2, 2, 1])
        with col_btn:
            clicked = st.button(
                "🧠 Run ML Analysis",
                key=f"{key_prefix}_ml_btn",
                use_container_width=True,
                type="primary",
            )
        with col_deep:
            deep_mode = st.checkbox(
                "🔍 Deep analysis",
                key=f"{key_prefix}_ml_deep",
                help="Extended reasoning with Lasso/Ridge feature selection context",
            )
        with col_clr:
            if st.button("↺", key=f"{key_prefix}_ml_clr", use_container_width=True, help="Clear"):
                st.session_state.pop(reply_key, None)
                st.session_state.pop(tool_report_key, None)
                st.rerun()

        if clicked:
            import pandas as pd
            import numpy as np
            try:
                df = pd.DataFrame(records)
                num_cols = numeric_keys or [c for c in df.select_dtypes(include=[np.number]).columns]

                # ── Step 1: Run ML tool suite (normality, outliers, PCA, regression) ──
                tool_report = None
                with st.spinner("Running ML tools (normality · outliers · PCA · regression)…"):
                    try:
                        from protein_design_hub.web.agent_tools import run_ml_tool_suite
                        tool_report = run_ml_tool_suite(records, num_cols, target_key)
                        st.session_state[tool_report_key] = tool_report
                    except Exception:
                        pass

                # ── Step 2: Supplementary pandas-based context ─────────────────────
                stat_lines = [f"Dataset: {len(records)} samples, {len(num_cols)} numeric features"]
                for col in num_cols[:14]:
                    s = df[col].dropna()
                    if not s.empty:
                        try:
                            from scipy.stats import skew as _skew, kurtosis as _kurt
                            sk, ku = float(_skew(s)), float(_kurt(s))
                        except Exception:
                            sk, ku = float(s.skew()), 0.0
                        stat_lines.append(
                            f"  {col}: n={len(s)}, mean={s.mean():.4g}, std={s.std():.4g}, "
                            f"min={s.min():.4g}, max={s.max():.4g}, "
                            f"skew={sk:+.3f}, kurt={ku:+.3f}"
                        )

                if len(num_cols) >= 2:
                    corr = df[num_cols].corr()
                    pairs = []
                    for i, c1 in enumerate(num_cols):
                        for j, c2 in enumerate(num_cols):
                            if j <= i:
                                continue
                            pairs.append((c1, c2, corr.loc[c1, c2]))
                    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    stat_lines.append("Top correlations (Pearson r):")
                    for c1, c2, r in pairs[:8]:
                        direction = "↑" if r > 0 else "↓"
                        stat_lines.append(f"  {c1} ↔ {c2}: r={r:+.3f} {direction}")

                if target_key and target_key in num_cols:
                    stat_lines.append(f"Regression target: {target_key}")

                # ── Step 3: Combine tool context + supplementary stats for LLM ──────
                stats_context = "\n".join(stat_lines)
                tool_ctx = (tool_report.context_string
                            if tool_report and tool_report.any_success else "")
                if tool_ctx:
                    full_stats = tool_ctx + "\n\nSupplementary stats:\n" + stats_context
                else:
                    full_stats = stats_context

                cross_ctx = _get_cross_page_context()
                full_context = full_stats
                if cross_ctx and "No cross-page" not in cross_ctx:
                    full_context += "\n\n" + cross_ctx

                # ── Step 4: LLM interpretation of computed results ─────────────────
                max_tok = 1100 if deep_mode else 750
                question = (
                    f"You are analyzing {page_name} results from a protein design workflow. "
                    f"The dataset has {len(records)} samples and {len(num_cols)} numeric features. "
                    f"Tool suite results are provided above (normality tests, outlier detection, "
                    f"PCA, Lasso/Ridge feature importance, and regression).\n\n"
                    "Provide a structured ML analysis covering:\n"
                    "1. **Distributional patterns** — flag high skew (|skew|>1) or heavy tails "
                    "(|kurt|>2); recommend transformations for non-normal features\n"
                    "2. **Correlation structure** — identify strongly correlated pairs (|r|>0.7) "
                    "suggesting mechanistic linkage or multicollinearity risk\n"
                    "3. **Feature relevance** — interpret Lasso-selected features and Mutual Info "
                    "scores; explain biological or structural significance\n"
                    "4. **Anomalies and outliers** — interpret Isolation Forest anomaly rate and "
                    "Z-score outlier counts; flag any suspicious samples\n"
                    "5. **Regression quality** — evaluate R² and CV R²; flag overfitting "
                    "(OLS R² >> CV R²) and interpret Lasso regularisation strength\n"
                    "6. **Recommendations** — 2–3 concrete next steps for analysis or experiment"
                )
                if deep_mode:
                    question = "Think step by step. Use numbered headers for each section.\n\n" + question

                with st.spinner("ML Specialist interpreting tool results…"):
                    reply = ask_agent_advice(
                        question=question,
                        agent_name="Machine Learning Specialist",
                        context=full_context,
                        max_tokens=max_tok,
                    )
                st.session_state[reply_key] = reply
            except Exception as e:
                st.session_state[reply_key] = f"[Error] Could not build statistical context: {e}"

        # Render stored tool computation results (persists across reruns)
        stored_report = st.session_state.get(tool_report_key)
        if stored_report:
            try:
                from protein_design_hub.web.agent_tools import render_agent_tool_report
                st.markdown("**Tool Computation Results**")
                render_agent_tool_report(stored_report)
            except Exception:
                pass

        stored = st.session_state.get(reply_key)
        if stored:
            if str(stored).startswith("[Error]"):
                st.error(stored)
            else:
                _render_agent_response_card("Machine Learning Specialist", stored)


# ── Agent identity map ────────────────────────────────────────────────

AGENT_ICONS: Dict[str, str] = {
    "Principal Investigator": "👑",
    "Structural Biologist": "🔬",
    "Computational Biologist": "💻",
    "Machine Learning Specialist": "🤖",
    "Protein Engineer": "🔧",
    "Biophysicist": "⚡",
    "Digital Recep": "🛠️",
    "Liam": "📊",
    "Immunologist": "🧬",
    "Scientific Critic": "🎯",
}

AGENT_COLORS: Dict[str, str] = {
    "Principal Investigator": "#fbbf24",
    "Structural Biologist": "#60a5fa",
    "Computational Biologist": "#34d399",
    "Machine Learning Specialist": "#818cf8",
    "Protein Engineer": "#fb923c",
    "Biophysicist": "#f472b6",
    "Digital Recep": "#a78bfa",
    "Liam": "#22c55e",
    "Immunologist": "#67e8f9",
    "Scientific Critic": "#f87171",
}

AGENT_ROLES: Dict[str, str] = {
    "Principal Investigator": "Strategic oversight · Experimental design",
    "Structural Biologist": "Structure quality · pLDDT · fold assessment",
    "Computational Biologist": "Sequence analysis · MSA · evolution",
    "Machine Learning Specialist": "Feature analysis · regression · statistics",
    "Protein Engineer": "Stability · mutations · biophysics",
    "Biophysicist": "Energy · thermodynamics · dynamics",
    "Digital Recep": "Refinement · OST · quality metrics",
    "Liam": "Evaluation metrics · scoring · benchmarks",
    "Immunologist": "Epitopes · immunogenicity · antibodies",
    "Scientific Critic": "Critical review · limitations · rigour",
}


def _render_agent_response_card(agent_name: str, text: str) -> None:
    """Render a professional styled response card with agent identity header."""
    icon = AGENT_ICONS.get(agent_name, "🧬")
    color = AGENT_COLORS.get(agent_name, "#6366f1")
    role = AGENT_ROLES.get(agent_name, "Domain Expert")
    esc_name = _html.escape(agent_name)
    esc_role = _html.escape(role)

    st.markdown(
        f"""
<div style="
    border-left: 4px solid {color};
    background: linear-gradient(135deg, rgba(30,36,51,0.95), rgba(20,24,40,0.98));
    border-radius: 0 12px 12px 0;
    padding: 16px 20px 14px 18px;
    margin: 12px 0 8px 0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.35);
">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
    <span style="font-size:1.4rem;line-height:1">{icon}</span>
    <div>
      <div style="font-weight:700;color:{color};font-size:0.95rem;letter-spacing:.01em">{esc_name}</div>
      <div style="font-size:0.72rem;color:#6b7280;margin-top:1px">{esc_role}</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(text)


@contextmanager
def _temporary_llm_override(provider: str = "", model: str = ""):
    """Temporarily override active LLM provider/model for a single operation."""
    if not provider and not model:
        yield
        return

    settings = None
    previous = None
    try:
        from protein_design_hub.core.config import get_settings, normalize_ollama_model_name
        from protein_design_hub.agents.meeting import reset_llm_client

        settings = get_settings()
        previous = (
            settings.llm.provider,
            settings.llm.base_url,
            settings.llm.model,
            settings.llm.api_key,
        )

        if provider and provider != settings.llm.provider:
            settings.llm.provider = provider
            settings.llm.base_url = ""
            settings.llm.api_key = ""
            settings.llm.model = ""
        if model:
            if settings.llm.provider == "ollama":
                model = normalize_ollama_model_name(model)
            settings.llm.model = model

        reset_llm_client()
        _cached_llm_status.clear()
        list_available_models.clear()
        yield
    finally:
        if settings is not None and previous is not None:
            from protein_design_hub.agents.meeting import reset_llm_client

            (
                settings.llm.provider,
                settings.llm.base_url,
                settings.llm.model,
                settings.llm.api_key,
            ) = previous
            reset_llm_client()
            _cached_llm_status.clear()
            list_available_models.clear()


def render_all_experts_panel(
    title: str,
    agenda: str,
    context: str = "",
    questions: tuple[str, ...] = (),
    key_prefix: str = "all_experts",
    expanded: bool = False,
    provider_override: str = "",
    model_override: str = "",
    save_dir: Optional[Path] = None,
) -> None:
    """Run a team meeting with all experts and render summary + transcript."""
    llm_online = llm_available()
    if not llm_online:
        st.info(
            "Current LLM backend looks offline. You can still run with an alternate provider "
            "using the override controls below."
        )

    summary_key = f"{key_prefix}_summary"
    transcript_key = f"{key_prefix}_transcript"
    running_key = f"{key_prefix}_running"

    with st.expander(title, expanded=expanded):
        st.markdown(
            '<div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">'
            + "".join(
                f'<span style="background:rgba(30,36,51,0.9);border:1px solid {AGENT_COLORS.get(a,"#6366f1")}33;'
                f'color:{AGENT_COLORS.get(a,"#6366f1")};border-radius:20px;padding:2px 10px;font-size:.7rem">'
                f'{AGENT_ICONS.get(a,"🧬")} {_html.escape(a)}</span>'
                for a in ["Principal Investigator", "Structural Biologist", "Computational Biologist",
                          "Machine Learning Specialist", "Protein Engineer", "Biophysicist",
                          "Digital Recep", "Immunologist", "Scientific Critic"]
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        effective_provider = provider_override
        effective_model = model_override

        # Built-in backend override for every all-expert panel.
        # Page-specific overrides (if passed) take precedence.
        if not provider_override and not model_override:
            ov_key = f"{key_prefix}_ov_enable"
            mode_key = f"{key_prefix}_ov_mode"
            model_key = f"{key_prefix}_ov_model"
            custom_provider_key = f"{key_prefix}_ov_custom_provider"

            ov_enabled = st.checkbox(
                "Use alternate backend for this expert panel",
                value=bool(st.session_state.get(ov_key, False)),
                key=ov_key,
            )

            if ov_enabled:
                try:
                    from protein_design_hub.core.config import LLM_PROVIDER_PRESETS

                    provider_options = ["current", "ollama", "deepseek"] + [
                        p for p in LLM_PROVIDER_PRESETS.keys()
                        if p not in {"ollama", "deepseek"}
                    ] + ["custom"]
                    provider_default_model = {
                        provider: preset[1] for provider, preset in LLM_PROVIDER_PRESETS.items()
                    }
                except Exception:
                    provider_options = ["current", "ollama", "deepseek", "custom"]
                    provider_default_model = {
                        "ollama": "qwen2.5:14b",
                        "deepseek": "deepseek-chat",
                    }

                current_mode = st.session_state.get(mode_key, "current")
                if current_mode not in provider_options:
                    current_mode = "current"

                mode = st.selectbox(
                    "Provider override",
                    options=provider_options,
                    index=provider_options.index(current_mode),
                    format_func=lambda x: {
                        "current": "Current configured provider",
                        "ollama": "Ollama (local: qwen2.5:14b / deepseek-r1:14b)",
                        "deepseek": "DeepSeek Cloud (requires API key)",
                        "custom": "Custom provider/model",
                    }.get(x, x),
                    key=mode_key,
                )

                if mode == "custom":
                    custom_provider = st.text_input(
                        "Custom provider ID",
                        value=st.session_state.get(custom_provider_key, ""),
                        key=custom_provider_key,
                    ).strip()
                    model = st.text_input(
                        "Custom model ID (optional)",
                        value=st.session_state.get(model_key, ""),
                        key=model_key,
                    ).strip()
                    effective_provider = custom_provider
                    effective_model = model
                elif mode == "current":
                    model = st.text_input(
                        "Model override (optional)",
                        value=st.session_state.get(model_key, ""),
                        key=model_key,
                        help="Leave empty to keep configured model.",
                    ).strip()
                    effective_provider = ""
                    effective_model = model
                elif mode == "ollama":
                    from protein_design_hub.core.config import (
                        normalize_ollama_model_name,
                        OLLAMA_RECOMMENDED_MODELS,
                    )
                    default_model = normalize_ollama_model_name(
                        st.session_state.get(model_key, "qwen2.5:14b") or "qwen2.5:14b"
                    )
                    if default_model != st.session_state.get(model_key, ""):
                        st.session_state[model_key] = default_model
                    rec_ids = [m[0] for m in OLLAMA_RECOMMENDED_MODELS]
                    rec_labels = {m[0]: f"{m[0]} — {m[1]}" for m in OLLAMA_RECOMMENDED_MODELS}
                    if default_model not in rec_ids:
                        rec_ids = [default_model] + rec_ids
                        rec_labels[default_model] = default_model
                    sel_idx = rec_ids.index(default_model) if default_model in rec_ids else 0
                    model = st.selectbox(
                        "Ollama model",
                        options=rec_ids,
                        index=sel_idx,
                        format_func=lambda m: rec_labels.get(m, m),
                        key=model_key,
                        help="qwen2.5:14b is fast; deepseek-r1:14b provides deeper reasoning.",
                    )
                    effective_provider = "ollama"
                    effective_model = normalize_ollama_model_name(model or "qwen2.5:14b")
                elif mode == "deepseek":
                    model = st.text_input(
                        "DeepSeek model (optional)",
                        value=st.session_state.get(model_key, "deepseek-chat") or "deepseek-chat",
                        key=model_key,
                    ).strip()
                    effective_provider = "deepseek"
                    effective_model = model or "deepseek-chat"
                else:
                    suggested = provider_default_model.get(mode, "")
                    model = st.text_input(
                        "Model (optional)",
                        value=st.session_state.get(model_key, suggested) or suggested,
                        key=model_key,
                    ).strip()
                    effective_provider = mode
                    effective_model = model
            else:
                effective_provider = ""
                effective_model = ""

        elif provider_override or model_override:
            st.caption(
                f"Backend override in effect: provider=`{provider_override or 'current'}`, "
                f"model=`{model_override or 'configured default'}`"
            )

        cfg = _get_llm_cfg()
        cfg_provider = cfg.provider if cfg else "unknown"
        cfg_model = cfg.model if cfg else "unknown"
        effective_provider_label = effective_provider or cfg_provider
        effective_model_label = effective_model or cfg_model
        st.caption(
            f"Effective backend: provider=`{effective_provider_label}`, "
            f"model=`{effective_model_label}`"
        )

        run_disabled = (not llm_online) and (not effective_provider and not effective_model)

        c1, c2 = st.columns([1, 1])
        with c1:
            run_btn = st.button(
                "🧠 Run All-Expert Review",
                key=f"{key_prefix}_run",
                type="primary",
                disabled=run_disabled,
            )
        with c2:
            if st.button("🗑 Clear", key=f"{key_prefix}_clr"):
                st.session_state.pop(summary_key, None)
                st.session_state.pop(transcript_key, None)
                st.session_state.pop(running_key, None)
                st.rerun()

        if run_disabled:
            st.caption(
                "Run is disabled because the current backend is offline and no alternate "
                "provider/model override is set."
            )

        if run_btn and not st.session_state.get(running_key):
            st.session_state[running_key] = True
            try:
                from protein_design_hub.agents.meeting import run_meeting
                from protein_design_hub.agents.scientists import (
                    PRINCIPAL_INVESTIGATOR,
                    ALL_EXPERTS_TEAM_MEMBERS,
                )
                from protein_design_hub.core.config import get_settings

                settings = get_settings()
                if save_dir is not None:
                    sd = Path(save_dir)
                else:
                    scoped_job = (st.session_state.get("active_job_dir") or "").strip()
                    if not scoped_job:
                        scoped_job = (st.session_state.get("mutagenesis_job_dir") or "").strip()
                    if scoped_job:
                        sd = Path(scoped_job) / "meetings"
                    else:
                        sd = Path("./outputs/meetings")
                sd.mkdir(parents=True, exist_ok=True)
                sn = f"{key_prefix}_{int(time.time())}"
                with st.spinner("Running all-expert meeting... this may take a few minutes."):
                    with _temporary_llm_override(effective_provider, effective_model):
                        summary = run_meeting(
                            meeting_type="team",
                            agenda=agenda,
                            agenda_questions=questions,
                            contexts=(context,) if context else (),
                            save_dir=sd,
                            save_name=sn,
                            team_lead=PRINCIPAL_INVESTIGATOR,
                            team_members=ALL_EXPERTS_TEAM_MEMBERS,
                            num_rounds=settings.llm.num_rounds,
                            return_summary=True,
                        )
                st.session_state[summary_key] = summary
                tp = sd / f"{sn}.json"
                if tp.exists():
                    with open(tp) as f:
                        st.session_state[transcript_key] = json.load(f)
            except Exception as e:
                st.error(f"All-expert review failed: {e}")
                if "Connection" in str(e) or "refused" in str(e):
                    st.warning("Make sure your LLM backend is running (e.g. `ollama serve`).")
            finally:
                st.session_state[running_key] = False
                st.rerun()

        if st.session_state.get(summary_key):
            st.markdown("#### Summary")
            st.markdown(st.session_state[summary_key])

        if st.session_state.get(transcript_key):
            with st.expander("📜 Transcript", expanded=False):
                for turn in st.session_state[transcript_key]:
                    an = turn.get("agent", "Unknown")
                    msg = turn.get("message", "")
                    st.markdown(f"**{an}:** {msg}")


# ── Multi-turn chatbot ───────────────────────────────────────────────

def _chat_llm_call(
    agent_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 700,
) -> str:
    """Multi-turn LLM call preserving full conversation history."""
    cfg = _get_llm_cfg()
    if cfg is None:
        return "[Error] LLM not configured. Go to Agents > LLM Status to set up."

    agents = _resolve_agent_map()
    agent = agents.get(agent_name)
    sys_msg = agent.system_message if agent else {
        "role": "system",
        "content": f"You are an expert {agent_name}.",
    }

    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
        ensure_ollama_gpu(cfg.provider, cfg.model)
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[sys_msg] + messages,
            temperature=cfg.temperature,
            max_tokens=max_tokens,
            **ollama_extra_body(cfg.provider),
        )
        ensure_ollama_gpu(cfg.provider, cfg.model)
        return resp.choices[0].message.content or "(empty response)"
    except Exception as e:
        return f"[Error] LLM call failed: {e}"


# ── Chat CSS ──────────────────────────────────────────────────────────

_CHAT_CSS = """
<style>
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 12px 4px;
    scroll-behavior: smooth;
}
.chat-msg {
    display: flex;
    gap: 10px;
    margin-bottom: 14px;
    animation: chatFadeIn 0.3s ease;
}
@keyframes chatFadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.chat-msg-user {
    flex-direction: row-reverse;
}
.chat-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.chat-avatar-user {
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.35);
}
.chat-avatar-agent {
    background: rgba(168, 85, 247, 0.18);
    border: 1px solid rgba(168, 85, 247, 0.35);
}
.chat-bubble {
    max-width: 80%;
    padding: 10px 15px;
    border-radius: 14px;
    font-size: 0.88rem;
    line-height: 1.55;
    white-space: pre-wrap;
    word-break: break-word;
}
.chat-bubble-user {
    background: rgba(99, 102, 241, 0.13);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-bottom-right-radius: 4px;
    color: var(--pdhub-text, #e2e8f0);
}
.chat-bubble-agent {
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-bottom-left-radius: 4px;
    color: var(--pdhub-text, #e2e8f0);
}
.chat-name {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 3px;
    opacity: 0.7;
}
.chat-thinking {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    font-size: 0.82rem;
    color: #a78bfa;
    font-style: italic;
}
.chat-dot {
    width: 6px; height: 6px;
    background: #a78bfa;
    border-radius: 50%;
    animation: chatBounce 1.2s ease infinite;
}
.chat-dot:nth-child(2) { animation-delay: 0.2s; }
.chat-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes chatBounce {
    0%, 80%, 100% { transform: translateY(0); }
    40% { transform: translateY(-6px); }
}
.chat-suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
</style>
"""


def render_agent_chatbot(key_prefix: str = "chatbot") -> None:
    """Full chatbot-style interface for conversing with any scientist agent.

    Maintains conversation history in ``st.session_state`` so the chat
    persists across Streamlit reruns.

    Args:
        key_prefix: Unique prefix to avoid widget key collisions when
            the chatbot is rendered on multiple pages.
    """
    # Inject chat CSS once
    st.markdown(_CHAT_CSS, unsafe_allow_html=True)

    # ── session state keys ────────────────────────────────────────────
    hist_key = f"_chat_history_{key_prefix}"
    agent_key = f"_chat_agent_{key_prefix}"

    if hist_key not in st.session_state:
        st.session_state[hist_key] = []
    if agent_key not in st.session_state:
        st.session_state[agent_key] = "Structural Biologist"

    history: List[Dict[str, str]] = st.session_state[hist_key]

    # ── agent selector + controls ─────────────────────────────────────
    col_agent, col_clear = st.columns([3, 1])
    with col_agent:
        selected = st.selectbox(
            "Chat with expert",
            AGENT_OPTIONS,
            index=AGENT_OPTIONS.index(st.session_state[agent_key])
            if st.session_state[agent_key] in AGENT_OPTIONS else 0,
            key=f"{key_prefix}_sel",
            label_visibility="collapsed",
        )
        # If the agent changed, note it
        if selected != st.session_state[agent_key]:
            st.session_state[agent_key] = selected
    with col_clear:
        if st.button("🗑 New Chat", key=f"{key_prefix}_clr", use_container_width=True):
            st.session_state[hist_key] = []
            st.rerun()

    agent_name = st.session_state[agent_key]
    agent_icon = AGENT_ICONS.get(agent_name, "🧪")

    # ── agent info bar ────────────────────────────────────────────────
    agents_map = _resolve_agent_map()
    agent_obj = agents_map.get(agent_name)
    expertise_short = ""
    if agent_obj:
        expertise_short = agent_obj.expertise[:120]
        if len(agent_obj.expertise) > 120:
            expertise_short += "..."
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;padding:8px 14px;'
        f'background:rgba(168,85,247,0.06);border:1px solid rgba(168,85,247,0.18);'
        f'border-radius:10px;margin-bottom:10px">'
        f'<span style="font-size:1.4rem">{agent_icon}</span>'
        f'<div>'
        f'<div style="font-weight:700;font-size:.9rem;color:var(--pdhub-accent,#a78bfa)">{_html.escape(agent_name)}</div>'
        f'<div style="font-size:.75rem;color:var(--pdhub-text-secondary,#94a3b8)">'
        f'{_html.escape(expertise_short)}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── render chat history ───────────────────────────────────────────
    if not history:
        st.markdown(
            f'<div style="text-align:center;padding:30px 10px;color:var(--pdhub-text-muted,#64748b)">'
            f'<div style="font-size:2.5rem;margin-bottom:8px">{agent_icon}</div>'
            f'<div style="font-size:.95rem;font-weight:600;margin-bottom:4px">'
            f'Chat with {_html.escape(agent_name)}</div>'
            f'<div style="font-size:.82rem">Ask about protein structures, '
            f'prediction strategies, evaluation metrics, or design approaches.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Suggestion chips
        st.caption("💡 Quick questions to get started:")
        _suggestions = [
            "What predictor should I use for a 200-residue monomer?",
            "How do I interpret a pLDDT of 75?",
            "My clash score is 35 - should I refine?",
            "Suggest mutations to improve thermostability",
        ]
        sug_cols = st.columns(2)
        for i, sug in enumerate(_suggestions):
            with sug_cols[i % 2]:
                if st.button(
                    sug,
                    key=f"{key_prefix}_sug_{i}",
                    use_container_width=True,
                ):
                    # Add user message and get response
                    history.append({"role": "user", "content": sug})
                    with st.spinner(f"{agent_name} is thinking..."):
                        reply = _chat_llm_call(agent_name, history)
                    history.append({"role": "assistant", "content": reply})
                    st.session_state[hist_key] = history
                    st.rerun()
    else:
        # Build full chat HTML
        chat_html_parts = ['<div class="chat-container">']
        for msg in history:
            is_user = msg["role"] == "user"
            cls = "chat-msg-user" if is_user else ""
            av_cls = "chat-avatar-user" if is_user else "chat-avatar-agent"
            bub_cls = "chat-bubble-user" if is_user else "chat-bubble-agent"
            icon = "👤" if is_user else agent_icon
            name = "You" if is_user else agent_name
            raw = msg["content"]
            if is_user:
                text = _html.escape(raw)
            else:
                # Convert basic markdown to HTML for agent responses
                text = _html.escape(raw)
                # Bold: **text** → <b>text</b>
                import re as _re
                text = _re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
                # Italic: *text* → <i>text</i> (but not inside bold)
                text = _re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
                # Inline code: `text` → <code>text</code>
                text = _re.sub(r'`([^`]+)`', r'<code style="background:rgba(99,102,241,0.15);padding:1px 5px;border-radius:4px;font-size:.82em">\1</code>', text)
                # Bullet lists: lines starting with - or *
                text = _re.sub(r'^[\-\*]\s+', '• ', text, flags=_re.MULTILINE)
                # Numbered lists: preserve
                # Line breaks
                text = text.replace('\n', '<br>')
            chat_html_parts.append(
                f'<div class="chat-msg {cls}">'
                f'<div class="chat-avatar {av_cls}">{icon}</div>'
                f'<div class="chat-bubble {bub_cls}">'
                f'<div class="chat-name">{_html.escape(name)}</div>'
                f'{text}'
                f'</div></div>'
            )
        chat_html_parts.append('</div>')
        st.markdown("".join(chat_html_parts), unsafe_allow_html=True)

    # ── input area ────────────────────────────────────────────────────
    with st.form(key=f"{key_prefix}_form", clear_on_submit=True):
        col_inp, col_send = st.columns([5, 1])
        with col_inp:
            user_input = st.text_input(
                "Message",
                placeholder=f"Ask {agent_name} anything about protein design...",
                label_visibility="collapsed",
                key=f"{key_prefix}_inp",
            )
        with col_send:
            send = st.form_submit_button(
                "Send",
                type="primary",
                use_container_width=True,
            )

    if send and user_input and user_input.strip():
        msg = user_input.strip()
        history.append({"role": "user", "content": msg})
        with st.spinner(f"{agent_name} is thinking..."):
            reply = _chat_llm_call(agent_name, history)
        history.append({"role": "assistant", "content": reply})
        st.session_state[hist_key] = history
        st.rerun()

    # ── footer info ───────────────────────────────────────────────────
    n_turns = len([m for m in history if m["role"] == "user"])
    st.caption(
        f"{n_turns} message{'s' if n_turns != 1 else ''} "
        f"| Chatting with **{agent_name}** "
        f"| Switch agents above to get different perspectives"
    )
