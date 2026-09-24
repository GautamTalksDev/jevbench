"""Secondary analysis smoke tests (Amendment 10)."""

from __future__ import annotations

import numpy as np

from jevbench.secondary import (
    ambiguity_detection_table,
    auroc_vs_entropy,
    cross_primitive_jsd,
    repeat_instability,
    temperature_scaling_cv,
    top_uncertainty,
)


def test_top_uncertainty_and_auroc() -> None:
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3]])
    unc = top_uncertainty(probs)
    assert unc[0] < unc[1]
    ent = np.array([0.2, 1.4])
    assert auroc_vs_entropy(unc, ent) == 1.0


def test_cross_jsd_and_instability() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    j = cross_primitive_jsd(a, b)
    assert j[0] == 0.0 and j[1] > 0.0
    inst = repeat_instability([a, b])
    assert 0.0 <= inst["flip_rate"] <= 1.0


def test_ambiguity_table_and_temperature() -> None:
    rng = np.random.default_rng(0)
    n = 40
    ent = rng.uniform(0.2, 1.5, size=n)
    sig1 = ent + rng.normal(0, 0.1, size=n)
    sig2 = rng.normal(0, 1, size=n)
    sig3 = ent + rng.normal(0, 0.5, size=n)
    table = ambiguity_detection_table(
        entropy=ent, top_unc=sig1, cross_jsd=sig2, instability=sig3
    )
    assert "one_minus_top" in table["signals"]
    probs = rng.dirichlet(np.ones(3), size=n)
    soft = probs.max(axis=1) * 0.8
    human = rng.dirichlet(np.ones(3), size=n)
    strata = np.array(["easy", "hard"] * (n // 2))
    cv = temperature_scaling_cv(probs, soft, human, strata, seed=1)
    assert cv["mean_temperature"] > 0
