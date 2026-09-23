"""Acceptance tests for top-label ECE and ΔECE — closed-form, not smoke.

Every test here must pass before EXP-1 spends a dollar.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jevbench.metrics import (
    delta_ece,
    expected_calibration_error,
    netcal_ece,
)

REPO = Path(__file__).resolve().parents[1]
BIAS_FLOOR_PATH = REPO / "results" / "bias_floor.json"
N_BINS = 10


def _calibrated_uniform_half(
    n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """p ~ U(0.5, 1), y ~ Bern(p). Positive class = 1 → conf = p (top-label)."""
    rng = np.random.default_rng(seed)
    probs = rng.uniform(0.5, 1.0, n)
    labels = (rng.uniform(0, 1, n) < probs).astype(int)
    return probs, labels


# ---------------------------------------------------------------------------
# 1. DEGENERATE — exact
# ---------------------------------------------------------------------------


def test_degenerate_ece_exact_point_four():
    """All p=0.9, predicted class correct for exactly 50% → ECE == 0.4."""
    n = 1000
    probs = np.full(n, 0.9)
    # pred always 1; half correct ⇒ half labels 1, half 0
    labels = np.concatenate([np.ones(n // 2, dtype=int), np.zeros(n // 2, dtype=int)])
    result = expected_calibration_error(probs, labels, n_bins=N_BINS, strategy="uniform")

    occupied = [b for b in result.bins if b.count > 0]
    assert len(occupied) == 1
    assert occupied[0].mean_confidence == pytest.approx(0.9, abs=1e-15)
    assert occupied[0].accuracy == pytest.approx(0.5, abs=1e-15)
    assert result.ece == pytest.approx(0.4, abs=1e-12)
    assert abs(result.ece - 0.4) < 1e-12


# ---------------------------------------------------------------------------
# 2. PERFECT — exact
# ---------------------------------------------------------------------------


def test_perfect_ece_exactly_zero():
    """All p=1.0, all correct → ECE == 0.0 exactly."""
    n = 500
    probs = np.ones(n)
    labels = np.ones(n, dtype=int)
    result = expected_calibration_error(probs, labels, n_bins=N_BINS, strategy="uniform")
    assert result.ece == 0.0
    assert result.mce == 0.0


# ---------------------------------------------------------------------------
# 3. CALIBRATED ASYMPTOTIC
# ---------------------------------------------------------------------------


def test_calibrated_ece_decreases_with_n():
    """ECE shrinks toward 0 as n grows — but is NOT zero at finite n."""
    ece_1k = expected_calibration_error(
        *_calibrated_uniform_half(1_000, seed=1), n_bins=N_BINS
    ).ece
    ece_100k = expected_calibration_error(
        *_calibrated_uniform_half(100_000, seed=1), n_bins=N_BINS
    ).ece
    assert ece_100k < ece_1k
    assert ece_100k > 0.0  # finite-n bias — subject of test 5


# ---------------------------------------------------------------------------
# 4. NETCAL AGREEMENT
# ---------------------------------------------------------------------------


def test_netcal_agreement_identical_inputs():
    pytest.importorskip("netcal")
    rng = np.random.default_rng(2026)
    probs = rng.uniform(0.05, 0.95, 3000)
    labels = (rng.uniform(0, 1, 3000) < probs).astype(int)

    ours = expected_calibration_error(probs, labels, n_bins=N_BINS, strategy="uniform")
    theirs = netcal_ece(probs, labels, n_bins=N_BINS)
    diff = abs(ours.ece - theirs)

    # Deliberate resolution of float residuals: both paths implement the same
    # top-label ECE (mean conf_bar, empty bins = 0). Sub-1e-12 disagreement is
    # IEEE reduction-order noise between our Python loop and netcal's. A real
    # variant fork (midpoint conf_bar, classic P(y=1), different digitize)
    # shows up at 1e-3 or larger — fail hard in that case, do not widen.
    if diff > 1e-12:
        pytest.fail(
            f"ECE disagreement with netcal on identical top-label inputs.\n"
            f"  ours   = {ours.ece!r}\n"
            f"  netcal = {theirs!r}\n"
            f"  |diff| = {diff!r}\n"
            f"Suspected variant difference: bin edge / digitize convention, "
            f"or midpoint-vs-mean conf_bar. Resolve deliberately; "
            f"do not widen tolerance. (Raw netcal.measure(p, y) is classic "
            f"P(y=1) reliability — we feed netcal our (conf, correct) arrays.)"
        )
    assert diff <= 1e-12


# ---------------------------------------------------------------------------
# 5. THE BIAS FLOOR
# ---------------------------------------------------------------------------


def test_bias_floor_ci_contains_zero_and_writes_table():
    """Perfectly calibrated equal-n strata → ΔECE 95% CI contains zero.

    Also sweep n and write results/bias_floor.json (null band for the paper).
    """
    n = 500
    ph, yh = _calibrated_uniform_half(n, seed=10)
    pe, ye = _calibrated_uniform_half(n, seed=11)
    res = delta_ece(
        ph, yh, pe, ye, seed=42, n_boot=2_000, n_bins=N_BINS, strategy="uniform"
    )
    assert res.equal_n
    assert res.ci_low < 0.0 < res.ci_high, (
        f"null ΔECE CI must contain zero; got [{res.ci_low}, {res.ci_high}] "
        f"point={res.point}"
    )

    # Prefer operating n from power_analysis.json when present (Prompt F4)
    operating_n: int | None = None
    power_path = REPO / "results" / "power_analysis.json"
    if power_path.is_file():
        try:
            operating_n = json.loads(power_path.read_text(encoding="utf-8")).get(
                "chosen_n_per_stratum"
            )
        except (json.JSONDecodeError, OSError):
            operating_n = None
    if operating_n is None:
        operating_n = 200  # fallback until power analysis is re-run

    ns = sorted({50, 100, 200, 500, 1000, 2000, int(operating_n)})
    n_reps = 200
    rows: list[dict] = []
    master = np.random.default_rng(20_260_921)
    for n_i in ns:
        eces: list[float] = []
        for _ in range(n_reps):
            seed_i = int(master.integers(0, 2**31 - 1))
            p, y = _calibrated_uniform_half(n_i, seed=seed_i)
            eces.append(
                float(
                    expected_calibration_error(
                        p, y, n_bins=N_BINS, strategy="uniform"
                    ).ece
                )
            )
        arr = np.asarray(eces, dtype=float)
        rows.append(
            {
                "n": n_i,
                "n_bins": N_BINS,
                "strategy": "uniform",
                "n_reps": n_reps,
                "mean_ece": float(arr.mean()),
                "std_ece": float(arr.std(ddof=1)),
                "p025_ece": float(np.quantile(arr, 0.025)),
                "p975_ece": float(np.quantile(arr, 0.975)),
                "is_operating_n": n_i == int(operating_n),
            }
        )

    op_row = next(r for r in rows if r["is_operating_n"])
    effect = 0.09
    ratio = effect / op_row["mean_ece"] if op_row["mean_ece"] > 0 else float("nan")
    methods_sentence = (
        f"At n={operating_n} per stratum with {N_BINS} bins, a perfectly "
        f"calibrated model yields mean ECE {op_row['mean_ece']:.4f}; our "
        f"pre-specified effect size of 0.09 is {ratio:.2f}x that floor."
    )

    payload = {
        "schema": "jevbench.bias_floor.v2",
        "note": (
            "Mean top-label ECE of a perfectly calibrated binary sample "
            "(p~U(0.5,1), y~Bern(p)) at each n. This is the finite-sample "
            "bias floor: observed ΔECE must clear the corresponding band "
            "to mean anything. Equal n per stratum is mandatory. Equal-n "
            "cancels much of the BIAS between strata but not the VARIANCE."
        ),
        "seed": 20_260_921,
        "generator": "p ~ Uniform(0.5, 1.0); y ~ Bernoulli(p)",
        "operating_n": int(operating_n),
        "pre_specified_effect_size": effect,
        "methods_sentence": methods_sentence,
        "rows": rows,
    }
    BIAS_FLOOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    BIAS_FLOOR_PATH.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paper_fig = REPO / "paper" / "figures" / "bias_floor.json"
    if paper_fig.parent.is_dir():
        paper_fig.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    # Monotone-ish: larger n → smaller mean ECE (bias shrinks)
    means = [r["mean_ece"] for r in rows]
    assert means[-1] < means[0]
    assert BIAS_FLOOR_PATH.is_file()
    assert "At n=" in methods_sentence

    print("\nNull ECE bias floor (perfectly calibrated):")
    print(f"{'n':>6}  {'mean ECE':>10}  {'p2.5':>10}  {'p97.5':>10}")
    for r in rows:
        mark = " *" if r["is_operating_n"] else ""
        print(
            f"{r['n']:6d}  {r['mean_ece']:10.4f}  {r['p025_ece']:10.4f}  "
            f"{r['p975_ece']:10.4f}{mark}"
        )
    print(f"\nMethods (verbatim):\n{methods_sentence}")


# ---------------------------------------------------------------------------
# 6. UNEQUAL-N GUARD
# ---------------------------------------------------------------------------


def test_unequal_n_guard():
    ph, yh = _calibrated_uniform_half(100, seed=1)
    pe, ye = _calibrated_uniform_half(80, seed=2)
    with pytest.raises(ValueError, match="unequal n|sample size|bias"):
        delta_ece(ph, yh, pe, ye, seed=0, n_boot=50)

    res = delta_ece(
        ph, yh, pe, ye, seed=0, n_boot=50, allow_unequal_n=True
    )
    assert res.equal_n is False
    assert res.to_dict()["equal_n"] is False


# ---------------------------------------------------------------------------
# 7. BIAS DIRECTION — documents the trap
# ---------------------------------------------------------------------------


def test_unequal_n_bias_skews_delta_ece_positive():
    """DOCUMENTATION AS MUCH AS VERIFICATION.

    Two perfectly calibrated strata with n_hard=100, n_easy=1000 and
    allow_unequal_n=True. ECE bias is larger in the smaller sample, so
    ΔECE = ECE(hard) − ECE(easy) skews POSITIVE even though neither
    stratum is miscalibrated. This is exactly the false positive Part 0
    warns about — the result you hope to find, for the wrong reason.
    """
    n_hard, n_easy = 100, 1000
    n_draws = 40
    deltas: list[float] = []
    rng = np.random.default_rng(7)
    for _ in range(n_draws):
        sh = int(rng.integers(0, 2**31 - 1))
        se = int(rng.integers(0, 2**31 - 1))
        ph, yh = _calibrated_uniform_half(n_hard, seed=sh)
        pe, ye = _calibrated_uniform_half(n_easy, seed=se)
        # Point estimate only (n_boot=1 keeps this cheap; we care about the
        # point's sampling distribution under unequal n).
        res = delta_ece(
            ph,
            yh,
            pe,
            ye,
            seed=int(rng.integers(0, 2**31 - 1)),
            n_boot=1,
            n_bins=N_BINS,
            allow_unequal_n=True,
        )
        assert res.equal_n is False
        deltas.append(res.point)

    mean_delta = float(np.mean(deltas))
    assert mean_delta > 0.0, (
        f"Expected positive bias in ΔECE under unequal n "
        f"(n_hard={n_hard}, n_easy={n_easy}); got mean={mean_delta}"
    )


# ---------------------------------------------------------------------------
# 8. DETERMINISM
# ---------------------------------------------------------------------------


def test_delta_ece_determinism_and_overlapping_seeds():
    n = 200
    ph, yh = _calibrated_uniform_half(n, seed=30)
    pe, ye = _calibrated_uniform_half(n, seed=31)

    a = delta_ece(ph, yh, pe, ye, seed=99, n_boot=500, n_bins=N_BINS)
    b = delta_ece(ph, yh, pe, ye, seed=99, n_boot=500, n_bins=N_BINS)
    # Same seed ⇒ bit-for-bit identical on the estimand + interval
    assert a.point == b.point
    assert a.ci_low == b.ci_low
    assert a.ci_high == b.ci_high
    assert a.ece_hard.ece == b.ece_hard.ece
    assert a.ece_easy.ece == b.ece_easy.ece
    assert a.ece_hard.bin_counts == b.ece_hard.bin_counts
    assert a.ece_easy.bin_counts == b.ece_easy.bin_counts
    assert a.seed == b.seed == 99

    c = delta_ece(ph, yh, pe, ye, seed=100, n_boot=500, n_bins=N_BINS)
    # Different seed ⇒ different resamples…
    assert (c.ci_low, c.ci_high, c.point) != (a.ci_low, a.ci_high, a.point)
    # …but intervals still overlap (both target the same estimand)
    assert a.ci_low <= c.ci_high and c.ci_low <= a.ci_high
