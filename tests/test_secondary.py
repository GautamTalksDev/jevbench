"""Secondary analysis tests (Amendment 10 / Step 15) on synthetic data."""

from __future__ import annotations

import numpy as np

from jevbench.secondary import (
    ambiguity_detection_table,
    auroc_binary,
    auroc_vs_entropy,
    choice_margin,
    cross_entropy_to_humans,
    cross_primitive_jsd,
    fit_temperature,
    partial_spearman,
    per_item_repeat_signals,
    repeat_instability,
    spearman_vs_entropy,
    stability_reconciliation_table,
    temperature_scale_probs,
    temperature_scaling_cv,
    top_uncertainty,
)


def test_top_uncertainty_and_auroc() -> None:
    probs = np.array([[0.9, 0.05, 0.05], [0.4, 0.3, 0.3]])
    unc = top_uncertainty(probs)
    assert unc[0] < unc[1]
    ent = np.array([0.2, 1.4])
    assert auroc_vs_entropy(unc, ent) == 1.0
    assert auroc_binary(unc, np.array([0, 1])) == 1.0


def test_cross_jsd_and_instability() -> None:
    a = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    b = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    j = cross_primitive_jsd(a, b)
    assert j[0] == 0.0 and j[1] > 0.0
    inst = repeat_instability([a, b])
    assert 0.0 <= inst["flip_rate"] <= 1.0
    per = per_item_repeat_signals([a, b])
    assert per["flip_rate"][0] == 0.0
    assert per["flip_rate"][1] > 0.0


def test_s1_known_ordering() -> None:
    """Signal perfectly tracking entropy → AUROC 1 and Spearman ~1."""
    n = 60
    ent = np.linspace(0.1, 1.5, n)
    hard = (ent >= np.median(ent)).astype(int)
    # Perfect signal = entropy itself
    sig = ent.copy()
    noise = np.random.default_rng(0).normal(0, 1, size=n)
    table = ambiguity_detection_table(
        entropy=ent,
        hard_indicator=hard,
        top_unc=sig,
        cross_jsd=noise,
        flip_rate=sig,
        mean_tvd=sig,
        n_boot=50,
        seed=1,
    )
    assert table["signals"]["one_minus_top"]["auroc_hard_vs_easy"] == 1.0
    assert table["signals"]["one_minus_top"]["spearman_vs_entropy"]["rho"] > 0.99
    assert (
        table["signals"]["cross_primitive_jsd"]["spearman_vs_entropy"]["rho"]
        < table["signals"]["one_minus_top"]["spearman_vs_entropy"]["rho"]
    )


def test_s2_partial_and_margin() -> None:
    rng = np.random.default_rng(2)
    n = 80
    margin = rng.uniform(0.0, 1.0, size=n)
    # flip anti-correlates with margin; entropy = margin + noise
    flip = 1.0 - margin + rng.normal(0, 0.05, size=n)
    ent = margin + rng.normal(0, 0.2, size=n)
    out = stability_reconciliation_table(
        flip_rate=flip, entropy=ent, margin=margin, n_boot=40, seed=3
    )
    assert out["spearman_flip_margin"]["rho"] < 0
    assert "partial_spearman_flip_entropy_given_margin" in out


def test_choice_margin_exact() -> None:
    m = choice_margin(np.array([[0.7, 0.2, 0.1], [0.5, 0.5, 0.0]]))
    assert abs(m[0] - 0.5) < 1e-12
    assert abs(m[1] - 0.0) < 1e-12


def test_s2_partial_spearman_known() -> None:
    # x independent of y given z: x = z + e1, y = z + e2 with large noise
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    x = z + rng.normal(size=200)
    y = z + rng.normal(size=200)
    full = spearman_vs_entropy(x, y)["rho"]
    part = partial_spearman(x, y, z)
    # Controlling for z removes most of the x–y association
    assert abs(part["rho"]) < abs(full)
    assert abs(part["rho"]) < 0.25


def test_s3_temperature_recovers_sharp_distribution() -> None:
    """Overconfident one-hot vs softer humans → T > 1 lowers CE."""
    rng = np.random.default_rng(4)
    n = 80
    human = rng.dirichlet(np.ones(3) * 2.0, size=n)
    peaked = np.zeros_like(human)
    peaked[np.arange(n), human.argmax(axis=1)] = 1.0
    probs = 0.9 * peaked + 0.1 * human
    probs = probs / probs.sum(axis=1, keepdims=True)
    t = fit_temperature(probs, human)
    assert t > 1.0  # soften
    ce0 = cross_entropy_to_humans(probs, human)
    ce1 = cross_entropy_to_humans(temperature_scale_probs(probs, t), human)
    assert ce1 <= ce0 + 1e-9

    soft = probs.max(axis=1)
    strata = np.array(["easy", "hard"] * (n // 2))
    labels = human.argmax(axis=1)
    cv = temperature_scaling_cv(
        probs, soft, human, strata, labels=labels, seed=5
    )
    assert cv["mean_temperature"] > 0
    assert "before" in cv["folds"][0] and "after" in cv["folds"][0]
    assert "soft_ece_hard" in cv["before_mean"]
