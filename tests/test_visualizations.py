"""Regression tests for web/visualizations figure builders.

The page web-smoke never supplies real PAE/contact data, so figure-construction
bugs (e.g. an invalid plotly colorbar property) slipped through. These build
each figure with representative inputs to guarantee they don't raise.
"""

import numpy as np
import pytest

viz = pytest.importorskip("protein_design_hub.web.visualizations")


def test_pae_heatmap_builds():
    fig = viz.create_pae_heatmap([[float(i + j) for j in range(12)] for i in range(12)])
    assert fig is not None and len(fig.data) == 1


def test_contact_map_builds_model_only_and_comparison():
    m = np.random.RandomState(0).rand(15, 15) * 12
    assert viz.create_contact_map(m) is not None
    assert viz.create_contact_map(m, reference_contacts=m.T) is not None


def test_plddt_plot_builds():
    fig = viz.create_plddt_plot([60.0 + i % 40 for i in range(30)], chain_breaks=[15])
    assert fig is not None
