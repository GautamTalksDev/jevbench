"""Pass-sensitivity helpers (post-hoc; not a registered endpoint)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.exp1_sensitivity import PASSES, SENSITIVITY_LABEL, _mean_across_passes


def test_sensitivity_label_constant() -> None:
    assert "NOT PREREGISTERED" in SENSITIVITY_LABEL
    assert len(PASSES) == 10


def test_mean_across_passes_averages_probs() -> None:
    rows = []
    for p in range(10):
        e = 0.1 * p
        rows.append(
            {
                "item_id": "a",
                "client": "jev",
                "role": "primary",
                "pass": p,
                "tier": "easy",
                "label": "entailment",
                "label_dist": [0.8, 0.1, 0.1],
                "probabilities": {
                    "entailment": e,
                    "neutral": (1.0 - e) / 2,
                    "contradiction": (1.0 - e) / 2,
                },
            }
        )
    out = _mean_across_passes(
        rows, "jev", probs_key="probabilities", item_ids={"a"}
    )
    assert len(out) == 1
    mean_e = float(np.mean([0.1 * p for p in range(10)]))
    assert out[0]["probabilities"]["entailment"] == pytest.approx(mean_e)
