"""Additive descriptive_direction + p_value_display (no method changes)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.exp1 import _descriptive_direction_cell, _p_value_display
from jevbench.metrics import hard_correctness, soft_correctness


def test_p_value_display_when_no_null_extreme() -> None:
    assert _p_value_display(0.0, n_null_pval=2000, n_null_extreme=0) == "p < 1/2001"
    assert _p_value_display(0.04, n_null_pval=2000, n_null_extreme=40).startswith("p = ")


def test_descriptive_direction_cell_shapes() -> None:
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=50)
    labels = probs.argmax(axis=1)
    dist = rng.dirichlet(np.ones(3), size=50)
    soft = soft_correctness(probs, dist)
    hard = hard_correctness(probs, labels)
    cell = _descriptive_direction_cell(probs, soft, labels)
    assert cell["n"] == 50
    assert len(cell["bins"]) == 10
    assert cell["overconfidence"] == pytest.approx(
        cell["mean_top_label_confidence"] - cell["mean_correctness"]
    )
    cell_h = _descriptive_direction_cell(probs, hard, labels)
    assert cell_h["n"] == 50
