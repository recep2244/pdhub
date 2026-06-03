"""Out-of-process services for the Protein Design Hub.

These modules run independently of Streamlit (no UI imports) and communicate
with the web app only through the durable SQLite job store in
:mod:`protein_design_hub.core.job_store`.
"""

from __future__ import annotations

__all__ = ["worker"]
