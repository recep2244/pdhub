"""Pytest configuration for the protein_design_hub test suite.

Scope: this file ONLY registers the 5-tier marker scheme (mirrors the
``markers`` list in ``[tool.pytest.ini_options]`` in ``pyproject.toml``) so the
markers resolve even when pytest is invoked without reading the project config.
No fixtures, no collection hooks, no deselection — a bare ``pytest`` run still
collects and executes every test.

Tiers
-----
unit         Pure-Python, no heavy deps/network/binaries (must pass with
             torch/GPU/external tools absent).
integration  Multi-module in-process integration, no GPU/network.
gpu          Requires a CUDA GPU and/or torch weights (skipped in CI).
external     Requires network or an external binary/service (skipped in CI).
e2e          Full end-to-end flow (web smoke + pipeline).
"""

from __future__ import annotations

# (marker name, one-line description) — keep in sync with pyproject.toml.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("unit", "Tier 1 — pure-Python unit test, no heavy deps/network/binaries."),
    ("integration", "Tier 2 — multi-module in-process integration, no GPU/network."),
    ("gpu", "Tier 3 — requires a CUDA GPU and/or torch weights (skipped in CI)."),
    ("external", "Tier 4 — requires network or external binary/service (skipped in CI)."),
    ("e2e", "Tier 5 — full end-to-end flow (web smoke + pipeline)."),
)


def pytest_configure(config) -> None:  # noqa: ANN001 (pytest passes its Config)
    """Register the 5-tier markers so ``-m`` selection works and ``--strict-markers`` is satisfied."""
    for name, description in _MARKERS:
        config.addinivalue_line("markers", f"{name}: {description}")
