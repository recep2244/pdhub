"""Shared agent-integration helpers for Streamlit pages.

Provides lightweight wrappers so every page can offer:
  * Quick LLM advice (single-turn call with a domain expert)
  * Multi-turn chatbot conversation with any agent
  * Agent status badge (cached, non-blocking)
"""

from __future__ import annotations

import html as _html
import time
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

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
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[sys_msg, {"role": "user", "content": user_content}],
            temperature=cfg.temperature,
            max_tokens=max_tokens,
        )
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
) -> None:
    """Collapsible 'Ask AI Agent' panel. Stores responses in session state."""
    # Session state key for persisting the reply across reruns
    reply_key = f"_agent_reply_{key_prefix}"

    with st.expander("🤖 Ask AI Agent for Advice", expanded=False):
        col_q, col_e = st.columns([3, 1])
        with col_q:
            question = st.text_area(
                "Your question",
                value=default_question,
                height=80,
                placeholder="e.g. How can I improve the pLDDT score?",
                key=f"{key_prefix}_q",
            )
        with col_e:
            agent_choice = st.selectbox(
                "Ask expert",
                AGENT_OPTIONS,
                index=AGENT_OPTIONS.index(expert) if expert in AGENT_OPTIONS else 0,
                key=f"{key_prefix}_expert",
            )

        col_btn, col_clear = st.columns([2, 1])
        with col_btn:
            clicked = st.button(
                "💬 Get Advice",
                key=f"{key_prefix}_btn",
                use_container_width=True,
                type="primary",
            )
        with col_clear:
            if st.button("Clear", key=f"{key_prefix}_clr", use_container_width=True):
                st.session_state.pop(reply_key, None)
                st.rerun()

        if clicked:
            if not question or not question.strip():
                st.warning("Please type a question first.")
            else:
                with st.spinner(f"Asking {agent_choice}..."):
                    reply = ask_agent_advice(
                        question=question.strip(),
                        agent_name=agent_choice,
                        context=page_context,
                    )
                st.session_state[reply_key] = {"agent": agent_choice, "text": reply}

        # Show stored reply (persists across reruns)
        stored = st.session_state.get(reply_key)
        if stored:
            if stored["text"].startswith("[Error]"):
                st.error(stored["text"])
                st.info("Make sure Ollama is running (`ollama serve`) or check LLM config on the Agents page.")
            else:
                st.markdown(f"**{stored['agent']}** says:")
                st.markdown(stored["text"])

        st.caption("Powered by the Agent Pipeline. Configure LLM on the Agents page.")


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
    }
    expert = "Structural Biologist"
    for key, exp in expert_map.items():
        if key.lower() in page_name.lower():
            expert = exp
            break

    with st.expander(f"🔬 AI Scientific Analysis ({expert})", expanded=False):
        col_btn, col_clr = st.columns([3, 1])
        with col_btn:
            clicked = st.button(
                f"🧠 Analyze with {expert}",
                key=f"{key_prefix}_btn",
                use_container_width=True,
                type="primary",
            )
        with col_clr:
            if st.button("Clear", key=f"{key_prefix}_clr", use_container_width=True):
                st.session_state.pop(reply_key, None)
                st.rerun()

        if clicked:
            with st.spinner(f"Consulting {expert}..."):
                reply = ask_agent_advice(
                    question=auto_q,
                    agent_name=expert,
                    context=context,
                    max_tokens=600,
                )
            st.session_state[reply_key] = {"agent": expert, "text": reply}

        stored = st.session_state.get(reply_key)
        if stored:
            if stored["text"].startswith("[Error]"):
                st.error(stored["text"])
            else:
                st.markdown(f"**{stored['agent']}:**")
                st.markdown(stored["text"])


# ── Agent icon lookup ─────────────────────────────────────────────────

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
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[sys_msg] + messages,
            temperature=cfg.temperature,
            max_tokens=max_tokens,
        )
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
