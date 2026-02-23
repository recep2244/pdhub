"""Global shared results context — cross-page agent access.

Every page writes its completed results here via ``set_page_results()``.
Agent panels call ``get_all_context_summary()`` to get a single rich
context string covering ALL pages so the LLM has full workflow visibility.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

_STORE_KEY = "_pdhub_shared_results"


# ---------------------------------------------------------------------------
# Core store
# ---------------------------------------------------------------------------

def _store() -> Dict[str, Any]:
    if _STORE_KEY not in st.session_state:
        st.session_state[_STORE_KEY] = {}
    return st.session_state[_STORE_KEY]


def set_page_results(page: str, results: Dict[str, Any]) -> None:
    """Write/update a page's results in the global store."""
    _store()[page] = {
        **{k: v for k, v in results.items() if not k.startswith("_")},
        "_page": page,
        "_timestamp": datetime.now().isoformat(),
    }


def get_page_results(page: str) -> Optional[Dict[str, Any]]:
    """Retrieve a specific page's latest results (or None)."""
    return _store().get(page)


def get_all_results() -> Dict[str, Dict[str, Any]]:
    """Return the full store."""
    return dict(_store())


def clear_page_results(page: str) -> None:
    _store().pop(page, None)


# ---------------------------------------------------------------------------
# Summary for LLM context
# ---------------------------------------------------------------------------

def get_all_context_summary(max_items_per_page: int = 20) -> str:
    """
    Build a compact cross-page summary string suitable for LLM consumption.
    Each page's numeric/string results are listed; lists are summarised.
    """
    store = _store()
    if not store:
        return "No cross-page results available yet."

    lines: List[str] = ["=== Protein Design Hub — Cross-Page Results ==="]

    _PRIORITY = [
        "Predict", "Evaluate", "Compare", "MPNN", "MutationScanner",
        "Evolution", "MSA", "Batch", "Agents",
    ]
    ordered = sorted(
        store.keys(),
        key=lambda p: (_PRIORITY.index(p) if p in _PRIORITY else 99, p),
    )

    for page in ordered:
        data = store[page]
        ts = data.get("_timestamp", "")[:16]
        lines.append(f"\n── {page} (updated {ts}) ──")
        count = 0
        for k, v in data.items():
            if k.startswith("_"):
                continue
            if count >= max_items_per_page:
                lines.append(f"  ... (truncated)")
                break
            if isinstance(v, (int, float)):
                lines.append(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
            elif isinstance(v, str):
                short = v[:120].replace("\n", " ")
                lines.append(f"  {k}: {short}")
            elif isinstance(v, bool):
                lines.append(f"  {k}: {v}")
            elif isinstance(v, list):
                lines.append(f"  {k}: [{len(v)} items] {_summarise_list(v)}")
            elif isinstance(v, dict):
                lines.append(f"  {k}: {{{len(v)} keys}}")
            count += 1

    return "\n".join(lines)


def _summarise_list(lst: list, n: int = 3) -> str:
    """Return a brief preview of list contents."""
    preview = []
    for item in lst[:n]:
        if isinstance(item, (int, float, str)):
            preview.append(str(item)[:40])
        elif isinstance(item, dict):
            preview.append("{" + ", ".join(f"{k}={v}" for k, v in list(item.items())[:2]) + "}")
    suffix = f"... +{len(lst)-n} more" if len(lst) > n else ""
    return ", ".join(preview) + (" " + suffix if suffix else "")


# ---------------------------------------------------------------------------
# Workflow status helper
# ---------------------------------------------------------------------------

def get_workflow_status() -> Dict[str, bool]:
    """Return which workflow steps have results."""
    store = _store()
    return {
        "predict": "Predict" in store,
        "evaluate": "Evaluate" in store,
        "compare": "Compare" in store,
        "mpnn": "MPNN" in store,
        "mutation_scan": "MutationScanner" in store,
        "evolution": "Evolution" in store,
        "msa": "MSA" in store,
        "batch": "Batch" in store,
    }


def render_workflow_status_bar() -> None:
    """Show a compact breadcrumb of completed workflow steps."""
    status = get_workflow_status()
    icons = {
        "predict": "🔮", "evaluate": "📊", "compare": "⚖️",
        "mpnn": "🎯", "mutation_scan": "🔬", "evolution": "🧬",
        "msa": "🧬", "batch": "📦",
    }
    labels = {
        "predict": "Predict", "evaluate": "Evaluate", "compare": "Compare",
        "mpnn": "MPNN", "mutation_scan": "Mutation Scan", "evolution": "Evolution",
        "msa": "MSA", "batch": "Batch",
    }
    parts = []
    for key, done in status.items():
        icon = icons[key]
        label = labels[key]
        if done:
            parts.append(f'<span style="color:#22c55e;font-weight:600;">{icon} {label}</span>')
        else:
            parts.append(f'<span style="color:#6b7280;">{icon} {label}</span>')

    if any(status.values()):
        st.markdown(
            '<div style="font-size:0.8rem;padding:6px 0;border-bottom:1px solid rgba(100,100,100,0.2);margin-bottom:8px;">'
            + "  ·  ".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )
