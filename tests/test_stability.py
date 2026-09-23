"""Offline tests for EXP-6 stability metrics."""

from __future__ import annotations

import numpy as np
import pytest

from jevbench.stability import (
    JS_MAX,
    analyze_stability,
    build_flipped_fixture,
    build_identical_repeats,
    build_planted_boundary_fixture,
    detect_model_version_changes,
    jensen_shannon,
    total_variation,
    RepeatObservation,
)


def test_identical_repeats_zero_flip_and_divergences():
    obs = build_identical_repeats(n_items=15, n_repeats=10)
    report = analyze_stability(obs, seed=0)
    assert report.label["flip_rate"]["point"] == 0.0
    assert report.label["n_items_flipped"] == 0
    assert report.confidence["mean_sd"] == 0.0
    assert report.distribution["mean_of_mean_tv"] == 0.0
    assert report.distribution["mean_of_mean_js"] == 0.0
    assert report.distribution["mean_of_max_js"] == 0.0


def test_flipped_fixture_exact_flip_rate():
    obs = build_flipped_fixture(n_items=20, n_repeats=10, flip_fraction=0.5)
    report = analyze_stability(obs, seed=1)
    assert report.label["flip_rate"]["point"] == pytest.approx(0.5)
    assert report.label["n_items_flipped"] == 10


def test_js_divergence_properties():
    p = np.array([0.7, 0.2, 0.1])
    q = np.array([0.1, 0.2, 0.7])
    # Bounded
    assert 0.0 <= jensen_shannon(p, q) <= JS_MAX + 1e-9
    # Symmetric
    assert jensen_shannon(p, q) == pytest.approx(jensen_shannon(q, p), abs=1e-12)
    # Zero iff identical
    assert jensen_shannon(p, p) == pytest.approx(0.0, abs=1e-15)
    # Finite when a probability is exactly 0 (KL would choke on support mismatch
    # for raw KL(p||q); JS via mixture stays finite)
    p0 = np.array([1.0, 0.0, 0.0])
    q0 = np.array([0.0, 1.0, 0.0])
    js = jensen_shannon(p0, q0)
    assert np.isfinite(js)
    assert js == pytest.approx(JS_MAX, abs=1e-6)  # extreme pair → near ln2
    # TV basics
    assert total_variation(p, p) == 0.0
    assert total_variation(p0, q0) == pytest.approx(1.0)


def test_boundary_analysis_recovers_planted_relationship():
    obs = build_planted_boundary_fixture(n_near=40, n_far=40, n_repeats=10)
    report = analyze_stability(obs, decision_threshold=0.5, seed=2)
    assert report.boundary.supports_reconciliation
    assert report.boundary.flip_rate_overall == pytest.approx(0.5, abs=0.01)
    occupied = [b for b in report.boundary.bins if b.n_items > 0]
    assert len(occupied) >= 2
    assert occupied[0].flip_rate > occupied[-1].flip_rate
    # Confidence stable overall
    assert report.confidence["mean_sd"] == pytest.approx(0.0, abs=1e-12)
    assert "reconciliation" in report.boundary.statement.lower()


def test_model_version_change_surfaced():
    obs = build_identical_repeats(n_items=4, n_repeats=3, model="jev-1.13.0")
    # Inject a mid-run alias move on one observation
    mutated = list(obs)
    mutated[5] = RepeatObservation(
        item_id=mutated[5].item_id,
        repeat=mutated[5].repeat,
        decision=mutated[5].decision,
        probabilities=mutated[5].probabilities,
        confidence=mutated[5].confidence,
        tier=mutated[5].tier,
        serving_path=mutated[5].serving_path,
        resolved_model="jev-1.14.0",  # alias moved
        wall_ms=mutated[5].wall_ms,
        client="jev",
    )
    report = detect_model_version_changes(mutated)
    assert report.n_changes >= 1
    assert report.warning is not None
    assert "PUBLISHABLE" in report.warning
    assert "jev-1.14.0" in report.unique_models
