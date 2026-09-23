"""Agreement suite: our wrappers vs netcal / sklearn.

Appendix material for the paper. Synthetic calibrated + miscalibrated fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

from jevbench.metrics import (
    _for_ece,
    brier_score,
    expected_calibration_error,
    netcal_ece,
)

netcal = pytest.importorskip("netcal", reason="netcal not installed")
from netcal.metrics import ECE as NetcalECE  # noqa: E402


def _calibrated(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.02, 0.98, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    return probs, labels


def _miscalibrated(n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    probs = np.full(n, 0.95)
    labels = np.concatenate([np.ones(n // 5, dtype=int), np.zeros(n - n // 5, dtype=int)])
    return probs, labels


def test_ece_agrees_with_netcal_on_toplabel_inputs():
    """Binning arithmetic: feed netcal the same (conf, correct) we use."""
    probs, labels = _calibrated()
    ours = expected_calibration_error(probs, labels, n_bins=15, strategy="uniform")
    theirs = netcal_ece(probs, labels, n_bins=15)
    assert ours.ece == pytest.approx(theirs, abs=1e-9)


def test_ece_agrees_with_netcal_miscalibrated():
    probs, labels = _miscalibrated()
    ours = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    theirs = netcal_ece(probs, labels, n_bins=10)
    assert ours.ece == pytest.approx(theirs, abs=1e-9)
    assert ours.ece > 0.3


def test_netcal_raw_binary_is_classic_not_toplabel():
    """Document the variant fork — do not silently adopt either.

    ``NetcalECE.measure(p, y)`` on raw binary probability of the positive
    class is classic reliability of P(y=1). Our primary endpoint is
    TOP-LABEL ECE: conf = max(p, 1-p), correct = 1[pred == y].

    Fixture: half the mass at p=0.1 (all y=0), half at p=0.9 (all y=0).
    Top-label collapses both into conf≈0.9 with 50% accuracy → ECE≈0.4.
    Classic keeps two bins (0.1 and 0.9) with different gaps → different ECE.
    """
    probs = np.concatenate([np.full(500, 0.1), np.full(500, 0.9)])
    labels = np.zeros(1000, dtype=int)

    ours = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    classic = float(NetcalECE(10).measure(probs, labels.astype(float)))
    conf, correct = _for_ece(probs, labels)
    toplabel_via_netcal = float(NetcalECE(10).measure(conf, correct))

    assert ours.ece == pytest.approx(toplabel_via_netcal, abs=1e-9)
    assert abs(ours.ece - classic) > 0.05
    assert ours.ece == pytest.approx(0.4, abs=0.02)


def test_brier_agrees_with_sklearn_to_1e9():
    probs, labels = _calibrated(800, seed=3)
    assert brier_score(probs, labels) == pytest.approx(
        brier_score_loss(labels, probs), abs=1e-9
    )


def test_calibrated_fixture_behaves():
    probs, labels = _calibrated()
    ece = expected_calibration_error(probs, labels, n_bins=20).ece
    assert ece < 0.03


def test_miscalibrated_fixture_behaves():
    probs, labels = _miscalibrated()
    ece = expected_calibration_error(probs, labels, n_bins=10).ece
    assert ece > 0.3
