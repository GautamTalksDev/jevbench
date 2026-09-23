"""Simulator calibration + smoke power (offline, no network)."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.metrics import expected_calibration_error
from jevbench.power import (
    calibrate_simulator,
    confidence_for_target,
    cross_check_vs_delta_ece,
    run_power_analysis,
    simulate_stratum,
)


def test_simulator_hits_target_ece_at_large_n():
    rows = calibrate_simulator(n=50_000, seed=0)  # 50k is enough for tol=0.005
    for r in rows:
        assert r.abs_error <= 0.005


def test_simulate_stratum_accuracy_and_ece():
    rng = np.random.default_rng(1)
    probs, labels = simulate_stratum(2000, accuracy=0.63, miscalibration=0.14, rng=rng)
    assert abs(labels.mean() - 0.63) < 0.002
    ece = expected_calibration_error(probs, labels).ece
    assert ece == pytest.approx(0.14, abs=0.002)
    assert confidence_for_target(0.63, 0.14) == pytest.approx(0.77)


def test_simulate_stratum_noise_reduces_separation():
    """Noise sources must move realised ΔECE away from the clean 0.09."""
    rng = np.random.default_rng(7)
    clean_deltas = []
    noisy_deltas = []
    for _ in range(40):
        ph, yh = simulate_stratum(200, 0.63, 0.14, rng)
        pe, ye = simulate_stratum(200, 0.92, 0.05, rng)
        clean_deltas.append(
            expected_calibration_error(ph, yh).ece
            - expected_calibration_error(pe, ye).ece
        )
        ph2, yh2 = simulate_stratum(
            200,
            0.63,
            0.14,
            rng,
            label_noise=0.05,
            difficulty_sd=0.10,
            tier_leakage=0.05,
            other_accuracy=0.92,
            other_miscalibration=0.05,
        )
        pe2, ye2 = simulate_stratum(
            200,
            0.92,
            0.05,
            rng,
            label_noise=0.05,
            difficulty_sd=0.10,
            tier_leakage=0.05,
            other_accuracy=0.63,
            other_miscalibration=0.14,
        )
        noisy_deltas.append(
            expected_calibration_error(ph2, yh2).ece
            - expected_calibration_error(pe2, ye2).ece
        )
    # Clean is near-deterministic at 0.09; noisy has lower mean |gap| or more var
    assert abs(float(np.mean(clean_deltas)) - 0.09) < 0.01
    assert float(np.std(noisy_deltas)) > float(np.std(clean_deltas))


def test_cross_check_fast_vs_delta_ece():
    cross = cross_check_vs_delta_ece(n=400, seed=2)
    assert cross["point_match"]
    assert cross["intervals_overlap"]


def test_power_analysis_quick(tmp_path):
    payload = run_power_analysis(
        out_dir=tmp_path,
        n_trials=20,
        n_boot=200,
        seed=3,
        write_chart=True,
        run_surfaces=False,
        run_coverage=False,
    )
    assert (tmp_path / "power_analysis.json").is_file()
    assert payload["mean_null_fpr"] < 0.20  # loose for tiny trial count
    assert payload["simulator_calibration"]
    rates = [r["rate_exclude_zero"] for r in payload["power_curve"]]
    assert max(rates) > 0.3
    assert "noise" in payload
