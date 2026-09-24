"""Amendment 9 — verdict rule and divergence helpers."""

from __future__ import annotations

import numpy as np

from jevbench.metrics import (
    jensen_shannon_divergence,
    total_variation_distance,
    uniform_baseline_divergence,
)
from jevbench.verdict import decide_verdict, p_holds_one_sided


def test_decide_verdict_tracks() -> None:
    v = decide_verdict(
        corrected_delta_ece=0.10, p_value=0.01, p_holds_reject_ge_effect=0.5
    )
    assert v["verdict"] == "tracks"


def test_decide_verdict_holds() -> None:
    v = decide_verdict(
        corrected_delta_ece=0.01, p_value=0.4, p_holds_reject_ge_effect=0.02
    )
    assert v["verdict"] == "holds"


def test_decide_verdict_inconclusive() -> None:
    v = decide_verdict(
        corrected_delta_ece=0.05, p_value=0.2, p_holds_reject_ge_effect=0.4
    )
    assert v["verdict"] == "inconclusive"


def test_p_holds_one_sided() -> None:
    sims = np.array([0.08, 0.09, 0.10, 0.11])
    assert p_holds_one_sided(0.05, sims) == 0.0
    assert p_holds_one_sided(0.12, sims) == 1.0


def test_jsd_tvd_uniform() -> None:
    p = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    q = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    assert float(jensen_shannon_divergence(p, q).mean()) == 0.0
    assert float(total_variation_distance(p, q).mean()) == 0.0
    uni = uniform_baseline_divergence(p)
    assert uni["jsd_mean"] > 0.0
