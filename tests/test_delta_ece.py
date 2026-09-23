"""ΔECE endpoint, equal-n guard, null band, top-label ECE pin."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.metrics import (
    _for_ece,
    delta_ece,
    ece_both_strategies,
    expected_calibration_error,
    null_delta_ece_band,
    subsample_to_equal_n,
)


def _calibrated(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.05, 0.95, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    return probs, labels


def test_binary_top_label_ece_definition():
    """Pinned: conf = max(p, 1-p), correct = 1[pred == y] — not classic P(y=1)."""
    probs = np.array([0.2, 0.8, 0.1, 0.9])
    labels = np.array([0, 1, 1, 1])
    conf, correct = _for_ece(probs, labels)
    np.testing.assert_allclose(conf, [0.8, 0.8, 0.9, 0.9])
    np.testing.assert_allclose(correct, [1.0, 1.0, 0.0, 1.0])

    # Classic P(y=1) reliability would use conf=probs and outcome=labels —
    # those differ on the first item (0.2 vs 0.8).
    assert not np.allclose(conf, probs)


def test_ece_uses_mean_confidence_not_midpoint():
    # All mass in (0.7, 0.8] at conf≈0.75; midpoint would be 0.75 too here,
    # so force asymmetric: confs at 0.71 only in that bin.
    probs = np.full(20, 0.71)  # pred=1, conf=0.71
    labels = np.ones(20, dtype=int)  # all correct → gap = |1 - 0.71|
    ece = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    # Only one occupied bin; ECE = |1 - 0.71| = 0.29
    assert ece.ece == pytest.approx(0.29, abs=1e-9)
    occupied = [b for b in ece.bins if b.count > 0]
    assert len(occupied) == 1
    assert occupied[0].mean_confidence == pytest.approx(0.71)
    # Midpoint of [0.7, 0.8] is 0.75 — we must NOT use that
    assert occupied[0].mean_confidence != pytest.approx(0.75)


def test_empty_bins_contribute_zero():
    probs = np.full(30, 0.95)
    labels = np.ones(30, dtype=int)
    ece = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    empty = [b for b in ece.bins if b.count == 0]
    assert empty
    assert all(b.fraction == 0.0 for b in empty)
    # ECE equals the single occupied bin's abs gap
    occ = [b for b in ece.bins if b.count > 0][0]
    assert ece.ece == pytest.approx(occ.abs_gap)


def test_ece_both_strategies():
    probs, labels = _calibrated(400, seed=11)
    u, q = ece_both_strategies(probs, labels, n_bins=10)
    assert u.strategy == "uniform"
    assert q.strategy == "quantile"
    assert u.n == q.n == 400


def test_delta_ece_rejects_unequal_n():
    ph, yh = _calibrated(100, seed=1)
    pe, ye = _calibrated(80, seed=2)
    with pytest.raises(ValueError, match="unequal n|sample size|bias"):
        delta_ece(ph, yh, pe, ye, seed=0, n_boot=50)


def test_delta_ece_allow_unequal_sets_flag():
    ph, yh = _calibrated(100, seed=1)
    pe, ye = _calibrated(80, seed=2)
    res = delta_ece(
        ph, yh, pe, ye, seed=0, n_boot=100, allow_unequal_n=True
    )
    assert res.equal_n is False
    assert res.n_hard == 100
    assert res.n_easy == 80
    assert "equal_n" in res.to_dict()


def test_subsample_to_equal_n_deterministic():
    pa, ya = _calibrated(120, seed=3)
    pb, yb = _calibrated(80, seed=4)
    ids_a = [f"a-{i}" for i in range(120)]
    ids_b = [f"b-{i}" for i in range(80)]
    s1 = subsample_to_equal_n(pa, ya, ids_a, pb, yb, ids_b, seed=99)
    s2 = subsample_to_equal_n(pa, ya, ids_a, pb, yb, ids_b, seed=99)
    assert s1.target_n == 80
    assert len(s1.item_ids_a) == len(s1.item_ids_b) == 80
    assert len(s1.discarded_item_ids) == 40
    assert all(i.startswith("a-") for i in s1.discarded_item_ids)
    assert s1.item_ids_a == s2.item_ids_a
    assert s1.discarded_item_ids == s2.discarded_item_ids

    # After subsample, delta_ece accepts without override
    res = delta_ece(
        s1.probs_a,
        s1.labels_a,
        s1.probs_b,
        s1.labels_b,
        seed=7,
        n_boot=200,
    )
    assert res.equal_n is True
    assert res.n_hard == res.n_easy == 80
    assert res.ece_hard.bin_counts  # occupancy present
    assert res.ece_easy.bin_counts


def test_null_delta_ece_indistinguishable_from_zero():
    """Part 0 control: perfect calibration, equal n → ΔECE ≈ 0 / CI covers 0.

    Two perfectly calibrated strata at a realistic per-stratum n. The point
    estimate of ΔECE must sit near zero; a stratified bootstrap CI on one
    draw must cross zero. If this fails, the pipeline invents a false positive.
    """
    n = 150
    n_bins = 10
    band = null_delta_ece_band(n, seed=20260921, n_sims=200, n_bins=n_bins)
    # Mean of null ΔECE should be near 0 relative to its scale
    assert abs(band.mean) < 0.05
    assert band.p025 < 0.0 < band.p975

    # One explicit delta_ece call on calibrated equal-n strata
    ph, yh = _calibrated(n, seed=10)
    pe, ye = _calibrated(n, seed=11)
    res = delta_ece(ph, yh, pe, ye, seed=42, n_boot=2_000, n_bins=n_bins)
    assert res.equal_n
    assert res.crosses_zero(), (
        f"null ΔECE CI should cross zero; got [{res.ci_low}, {res.ci_high}] "
        f"point={res.point}"
    )
    # Observed |point| should not wildly exceed the null band half-width
    assert abs(res.point) < max(abs(band.p025), abs(band.p975)) + 0.05


def test_delta_ece_detects_real_miscalibration_gap():
    """Easy: calibrated; hard: constant overconfident 0.95 vs 50% accuracy."""
    n = 200
    pe, ye = _calibrated(n, seed=20)
    ph = np.full(n, 0.95)
    yh = np.concatenate([np.ones(n // 2), np.zeros(n - n // 2)]).astype(int)
    res = delta_ece(ph, yh, pe, ye, seed=3, n_boot=1_000, n_bins=10)
    assert res.point > 0.2
    assert res.ci_low > 0.0  # should clear zero under strong miscalibration


def test_delta_ece_seed_recorded():
    pe, ye = _calibrated(50, seed=1)
    ph, yh = _calibrated(50, seed=2)
    res = delta_ece(ph, yh, pe, ye, seed=12345, n_boot=100)
    assert res.seed == 12345
    assert res.n_boot == 100
