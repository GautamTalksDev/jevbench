"""PROMPT O — Beta–Binomial soft null (offline, no Jev data)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.power import (
    simulate_soft_beta_binomial_stratum,
    soft_delta_ece_percentile,
    specimen_top_prob_pool,
    _soft_ece_uniform,
)


def test_specimen_top_prob_pool_shape() -> None:
    pool = specimen_top_prob_pool()
    assert pool.shape == (1500,)
    assert 0.34 <= float(pool.min()) <= float(pool.max()) <= 0.995


def test_soft_null_mean_near_p() -> None:
    rng = np.random.default_rng(0)
    pool = specimen_top_prob_pool()
    draw = simulate_soft_beta_binomial_stratum(
        5000, rng, kappa=20.0, p_pool=pool, conf_shift=0.0
    )
    # E[share|p]≈p — residual mean near 0
    resid = draw.soft_correct - draw.true_p
    assert abs(float(resid.mean())) < 0.01
    assert abs(float((draw.conf - draw.true_p).mean())) < 1e-9


def test_proof_of_cause_degenerate_ece_zero() -> None:
    """κ→∞, no binomial ⇒ share = conf = p per item ⇒ ECE≡0, CI never excludes 0."""
    rng = np.random.default_rng(1)
    pool = specimen_top_prob_pool()
    excl = 0
    for _ in range(40):
        trial = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        hard = simulate_soft_beta_binomial_stratum(
            200,
            trial,
            kappa=1.0,
            p_pool=pool,
            infinite_kappa=True,
            apply_binomial=False,
        )
        easy = simulate_soft_beta_binomial_stratum(
            200,
            trial,
            kappa=1.0,
            p_pool=pool,
            infinite_kappa=True,
            apply_binomial=False,
        )
        assert np.allclose(hard.conf, hard.soft_correct)
        assert _soft_ece_uniform(hard.conf, hard.soft_correct) == pytest.approx(0.0)
        iv = soft_delta_ece_percentile(
            hard.conf,
            hard.soft_correct,
            easy.conf,
            easy.soft_correct,
            n_boot=200,
            rng=trial,
        )
        excl += int(iv["excludes_zero"])
        assert iv["point"] == pytest.approx(0.0)
    assert excl == 0
