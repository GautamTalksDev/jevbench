"""PROMPT P — structural ECE bias correction (offline, no Jev data)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.power import (
    expected_ece_under_calibration,
    soft_delta_ece_corrected,
    soft_delta_ece_percentile,
    simulate_soft_beta_binomial_stratum,
    specimen_top_prob_pools,
    _soft_ece_uniform,
)


def test_expected_ece_positive_under_calibration() -> None:
    rng = np.random.default_rng(0)
    _, p_hard = specimen_top_prob_pools()
    conf = rng.choice(p_hard, size=200, replace=True)
    e0 = expected_ece_under_calibration(conf, kappa=20.0, n_sims=300, rng=rng)
    assert e0 > 0.0


def test_corrected_near_zero_under_null() -> None:
    rng = np.random.default_rng(1)
    p_easy, p_hard = specimen_top_prob_pools()
    hard = simulate_soft_beta_binomial_stratum(
        300, rng, kappa=20.0, p_pool=p_hard, conf_shift=0.0
    )
    easy = simulate_soft_beta_binomial_stratum(
        300, rng, kappa=20.0, p_pool=p_easy, conf_shift=0.0
    )
    iv = soft_delta_ece_corrected(
        hard.conf,
        hard.soft_correct,
        easy.conf,
        easy.soft_correct,
        kappa=20.0,
        n_boot=200,
        n_e0=300,
        n_null_pval=300,
        rng=rng,
    )
    assert "raw_delta_ece" in iv and "corrected_delta_ece" in iv
    # Correction should pull the point toward 0 relative to structural bias0.
    assert abs(iv["corrected_delta_ece"]) <= abs(iv["raw_delta_ece"]) + abs(
        iv["bias0_delta_ece"]
    ) + 1e-9


def test_percentile_exposes_tail_flags() -> None:
    rng = np.random.default_rng(2)
    p_easy, p_hard = specimen_top_prob_pools()
    hard = simulate_soft_beta_binomial_stratum(
        200, rng, kappa=20.0, p_pool=p_hard, conf_shift=0.15
    )
    easy = simulate_soft_beta_binomial_stratum(
        200, rng, kappa=20.0, p_pool=p_easy, conf_shift=0.0
    )
    iv = soft_delta_ece_percentile(
        hard.conf,
        hard.soft_correct,
        easy.conf,
        easy.soft_correct,
        n_boot=300,
        rng=rng,
    )
    assert "excludes_zero_upper" in iv and "excludes_zero_lower" in iv
    assert iv["excludes_zero"] == (iv["excludes_zero_upper"] or iv["excludes_zero_lower"])


def test_ece_zero_when_share_equals_conf() -> None:
    conf = np.linspace(0.4, 0.95, 50)
    assert _soft_ece_uniform(conf, conf) == pytest.approx(0.0)
