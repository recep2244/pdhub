"""Protein Design Hub — application entry / router.

Uses st.navigation + st.Page for explicit, AT-correct multi-page routing (the
page scripts live in app_pages/ so they are NOT auto-discovered/duplicated). The
built-in nav is hidden; the custom goal-driven sidebar (sidebar_nav) remains the
visible navigation. set_page_config is owned here, once.
"""

import sys
from pathlib import Path

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import streamlit as st

st.set_page_config(
    page_title="Protein Design Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

_P = "app_pages"  # page scripts dir (relative to this entry)


def _page(script, title, icon, default=False):
    return st.Page(f"{_P}/{script}", title=title, icon=icon, default=default)

# Goal-driven IA (blueprint decision #1). Built-in nav hidden → custom sidebar_nav shows.
nav = st.navigation(
    {
        "Launchpad": [
            _page("home.py", "Home", "🏠", default=True),
        ],
        "Design Tracks": [
            _page("14_binder.py", "Binder Design", "🔗"),
            _page("12_antibody.py", "Antibody", "🧫"),
            _page("15_plant.py", "Plant / Wheat", "🌾"),
            _page("10_mutation_scanner.py", "Mutagenesis", "🧬"),
        ],
        "Modeling": [
            _page("1_predict.py", "Predict", "🔮"),
            _page("2_evaluate.py", "Evaluate", "📊"),
            _page("3_compare.py", "Compare", "⚖️"),
        ],
        "Design Lab": [
            _page("0_design.py", "Editor", "✏️"),
            _page("8_mpnn.py", "MPNN Lab", "🎯"),
            _page("4_evolution.py", "Evolution", "📈"),
            _page("7_msa.py", "MSA", "🧬"),
        ],
        "AI & Tools": [
            _page("11_agents.py", "Agents", "🤖"),
            _page("5_batch.py", "Batch", "📦"),
            _page("9_jobs.py", "Jobs", "📁"),
            _page("6_settings.py", "Settings", "⚙️"),
            _page("13_guide.py", "Guide", "📖"),
        ],
    },
    position="hidden",
)

nav.run()
