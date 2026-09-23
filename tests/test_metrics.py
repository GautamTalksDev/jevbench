"""Heavy unit tests for jevbench.metrics — pure functions, no I/O."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from jevbench.metrics import (
    as_confidence,
    auroc,
    automation_at_accuracy,
    brier_score,
    calibration_by_tier,
    ece_with_occupancy,
    expected_calibration_error,
    mcnemar,
    negation_sum_distribution,
    paired_bootstrap,
    primitive_disagreement_rate,
    reliability_curve,
    risk_coverage_curve,
    selective_from_confidence,
)


def test_confidence_array_rejected_by_calibration():
    probs = np.array([0.9, 0.8, 0.1])
    labels = np.array([1, 1, 0])
    conf = as_confidence([0.9, 0.9, 0.9])
    with pytest.raises(TypeError, match="SHARPNESS"):
        expected_calibration_error(conf, labels)
    with pytest.raises(TypeError, match="SHARPNESS"):
        expected_calibration_error(probs, labels, confidence=conf)
    with pytest.raises(TypeError, match="SHARPNESS"):
        brier_score(conf, labels)
    with pytest.raises(TypeError, match="SHARPNESS"):
        automation_at_accuracy(conf, labels)


def test_ece_returns_occupancy_not_bare_float():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 200)
    labels = (rng.uniform(0, 1, 200) < probs).astype(int)
    result = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    assert not isinstance(result, float)
    assert hasattr(result, "ece")
    assert hasattr(result, "bin_counts")
    assert sum(result.bin_counts) == 200
    assert len(result.bins) == 10
    assert all(b.count == result.bin_counts[b.index] for b in result.bins)
    # Alias
    assert ece_with_occupancy(probs, labels).ece == pytest.approx(result.ece)


def test_ece_saturated_softmax_occupancy_visible():
    """50/60 in one bin must show up in occupancy — the failure mode we criticise."""
    probs = np.concatenate([np.full(50, 0.99), np.linspace(0.1, 0.5, 10)])
    labels = np.concatenate([np.ones(50, dtype=int), np.zeros(10, dtype=int)])
    result = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    assert result.max_bin_fraction >= 50 / 60 - 1e-9
    assert max(result.bin_counts) >= 50


def test_perfectly_calibrated_fixture_low_ece():
    rng = np.random.default_rng(1)
    # Draw probs then Bernoulli — asymptotically calibrated
    probs = rng.uniform(0.05, 0.95, 5000)
    labels = (rng.uniform(0, 1, 5000) < probs).astype(int)
    result = expected_calibration_error(probs, labels, n_bins=15, strategy="uniform")
    assert result.ece < 0.03


def test_deliberately_miscalibrated_fixture_high_ece():
    # Always predict 0.9; labels are 50/50 → large gap in that bin
    probs = np.full(1000, 0.9)
    labels = np.concatenate([np.ones(500, dtype=int), np.zeros(500, dtype=int)])
    result = expected_calibration_error(probs, labels, n_bins=10, strategy="uniform")
    assert result.ece > 0.3
    assert result.mce > 0.3


def test_uniform_and_quantile_both_reported():
    rng = np.random.default_rng(2)
    probs = rng.beta(5, 2, 500)  # skewed toward 1
    labels = (rng.uniform(0, 1, 500) < probs).astype(int)
    u = expected_calibration_error(probs, labels, strategy="uniform")
    q = expected_calibration_error(probs, labels, strategy="quantile")
    assert u.strategy == "uniform"
    assert q.strategy == "quantile"
    # Quantile bins should be more balanced in count
    u_counts = [c for c in u.bin_counts if c > 0]
    q_counts = [c for c in q.bin_counts if c > 0]
    assert np.std(q_counts) <= np.std(u_counts) + 5  # soft check


def test_brier_matches_sklearn():
    probs = np.array([0.1, 0.4, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert brier_score(probs, labels) == pytest.approx(brier_score_loss(labels, probs))


def test_reliability_curve_aligns_with_ece_bins():
    probs = np.linspace(0.05, 0.95, 100)
    labels = (probs > 0.5).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=5)
    curve = reliability_curve(probs, labels, n_bins=5)
    assert curve.bin_counts == ece.bin_counts
    assert curve.bin_edges == ece.bin_edges
    np.testing.assert_allclose(
        curve.bin_accuracy, ece.bin_accuracy, equal_nan=True
    )
    np.testing.assert_allclose(
        curve.bin_mean_confidence, ece.bin_mean_confidence, equal_nan=True
    )


def test_calibration_by_tier_flagship_shape():
    # Easy: calibrated + accurate; hard: miscalibrated + less accurate
    easy_p = np.full(200, 0.9)
    easy_y = np.concatenate([np.ones(180), np.zeros(20)]).astype(int)
    hard_p = np.full(200, 0.9)
    hard_y = np.concatenate([np.ones(100), np.zeros(100)]).astype(int)
    result = calibration_by_tier(
        {"easy": easy_p, "hard": hard_p},
        {"easy": easy_y, "hard": hard_y},
        tier_order=["easy", "hard"],
    )
    assert result.tier_names == ("easy", "hard")
    assert result.accuracies[0] > result.accuracies[1]
    assert result.ece_uniform[1] > result.ece_uniform[0]
    # Negative slope: as accuracy rises, ECE falls
    assert result.slope_uniform < 0
    assert result.tiers[0].ece_uniform.bin_counts


def test_auroc_docstring_warns_not_calibration():
    doc = auroc.__doc__ or ""
    assert "NOT calibration" in doc or "not calibration" in doc.lower()
    assert "0.9" in doc
    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert auroc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_automation_at_accuracy():
    # High predicted-class confidence, all decisions correct → full coverage
    probs = np.array([0.95, 0.9, 0.85, 0.2, 0.15, 0.1])
    labels = np.array([1, 1, 1, 0, 0, 0])
    result = automation_at_accuracy(probs, labels, target_accuracy=0.90)
    assert result.feasible
    assert result.coverage == pytest.approx(1.0)
    assert result.accuracy_on_automated == pytest.approx(1.0)

    # Errors sit near the decision boundary (low predicted-class confidence)
    probs2 = np.array([0.95, 0.9, 0.85, 0.55, 0.52, 0.51])
    labels2 = np.array([1, 1, 1, 0, 0, 0])  # mid probs predict 1 → wrong
    result2 = automation_at_accuracy(probs2, labels2, target_accuracy=0.90)
    assert result2.feasible
    assert result2.n_automated == 3
    assert result2.accuracy_on_automated >= 0.90
    assert result2.coverage == pytest.approx(0.5)


def test_risk_coverage_and_aurc():
    probs = np.linspace(0.1, 0.9, 50)
    labels = (probs > 0.5).astype(int)
    curve = risk_coverage_curve(probs, labels)
    assert len(curve.coverage) == len(curve.risk)
    assert curve.aurc >= 0.0
    assert np.isfinite(curve.aurc)


def test_selective_from_confidence_allowed():
    conf = as_confidence([0.99, 0.95, 0.5, 0.4])
    correct = np.array([1, 1, 0, 0])
    result = selective_from_confidence(conf, correct, target_accuracy=0.90)
    assert result.feasible
    assert result.n_automated == 2


def test_primitive_disagreement_and_negation():
    noul = np.array([0.2, 0.8, 0.6])
    choice = np.array([0.9, 0.8, 0.4])  # disagrees on first and third
    rate = primitive_disagreement_rate(noul, choice)
    assert rate == pytest.approx(2 / 3)

    dist = negation_sum_distribution([0.5, 0.7, 0.9], [0.5, 0.5, 0.29])
    assert dist.values[0] == pytest.approx(1.0)
    assert dist.values[2] == pytest.approx(1.19)
    assert dist.mean_abs_deviation > 0
    assert "1.19" in dist.example_note


def test_paired_bootstrap_and_mcnemar_return_intervals():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 200).astype(float)
    b = a.copy()
    b[:20] = 1 - b[:20]
    ci = paired_bootstrap(a, b, lambda x, y: float(x.mean() - y.mean()), n=500, seed=0)
    assert hasattr(ci, "low") and hasattr(ci, "high")
    assert ci.n_resamples == 500
    d = ci.to_dict()
    assert "crosses_zero" in d
    assert "interpretation" in d

    m = mcnemar(a.astype(bool), b.astype(bool), n_bootstrap=500, seed=0)
    assert "accuracy_delta" in m.to_dict()
    assert m.accuracy_delta.low <= m.accuracy_delta.point <= m.accuracy_delta.high


def test_multiclass_top_label_ece():
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )
    labels = np.array([0, 1, 2, 1])  # last is wrong
    result = expected_calibration_error(probs, labels, n_bins=4)
    assert result.n == 4
    assert sum(result.bin_counts) == 4
