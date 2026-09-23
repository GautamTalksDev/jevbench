"""Power analysis for the EXP-1 ΔECE endpoint.

Effect size 0.09 is anchored to the published literature, not chosen by us:
two published Jev benchmarks report roughly ECE 0.05–0.07 at ~92% accuracy
and ~0.154 at ~63% accuracy — a gap of about 0.09. The effect size is
therefore the magnitude that would actually explain the disagreement in the
existing literature. That sentence is the justification a reviewer will look for.

No network. No API key. Run before any paid call — this sets the budget.

Prompt F hardening
------------------
F1 adds label_noise / difficulty_sd / tier_leakage so the power curve is not
a near-deterministic step. Required n is chosen at the pessimistic corner
(label_noise=0.05, difficulty_sd=0.10), not the clean simulator.
F2 measures null FPR under percentile and BCa at 5000 trials.
F3 budgets via cost.estimate() with per-model input/output rates.
F4 records the bias floor at the chosen operating n.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from jevbench.cost import estimate as cost_estimate
from jevbench.metrics import (
    _bca_endpoints,
    delta_ece,
    expected_calibration_error,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

# Literature-anchored effect (see module docstring).
TARGET_DELTA_ECE = 0.09
EASY_ACCURACY = 0.92
EASY_ECE = 0.05
HARD_ACCURACY = 0.63
HARD_ECE = 0.14  # 0.14 - 0.05 = 0.09

# F1: finer grid through the transition band
N_GRID = (100, 110, 125, 140, 160, 180, 200, 250, 300, 500, 750, 1000)
LABEL_NOISE_SWEEP = (0.0, 0.03, 0.05)
DIFFICULTY_SD_SWEEP = (0.0, 0.05, 0.10)
TIER_LEAKAGE_SWEEP = (0.0, 0.05)

# Pessimistic decision corner (Prompt F decision rule)
PESSIMISTIC_LABEL_NOISE = 0.05
PESSIMISTIC_DIFFICULTY_SD = 0.10
PESSIMISTIC_TIER_LEAKAGE = 0.0  # decision rule names the other two only

N_TRIALS_DEFAULT = 500
N_TRIALS_COVERAGE = 5_000
N_BOOT_POWER = 2_000  # power only; real analysis uses 10_000
N_BOOT_REAL = 10_000
POWER_THRESHOLD = 0.80
CALIBRATE_N = 200_000
CALIBRATE_TARGETS = (0.0, 0.02, 0.05, 0.09, 0.15)
CALIBRATE_TOL = 0.005
CALIBRATE_ACCURACY = 0.75

PRICE_SNAPSHOT = "2026-09-19"
BASELINE_MODEL = "openai/gpt-4o-mini"
REPEATS_DEFAULT = 3

# Soft ceiling: if pessimistic n exceeds this, say so and stop underpowered.
LABEL_BUDGET_CEILING = 750  # per stratum; ChaosNLI already supplies the 100 annotations


@dataclass(frozen=True)
class SimulatorCalibrationRow:
    target_ece: float
    measured_ece: float
    abs_error: float
    n: int
    accuracy: float


def confidence_for_target(accuracy: float, miscalibration: float) -> float:
    """Choose constant top-label confidence c with |accuracy - c| = miscalibration.

    Prefers over-confident c = accuracy + miscalibration when feasible in
    [0.5, 1]; otherwise under-confident c = accuracy - miscalibration.
    """
    if miscalibration < 0:
        raise ValueError("miscalibration (target ECE) must be >= 0")
    if not 0.5 <= accuracy <= 1.0:
        raise ValueError(
            "simulate_stratum requires accuracy in [0.5, 1] so top-label "
            f"confidence can sit in [0.5, 1]; got accuracy={accuracy}"
        )
    c_hi = accuracy + miscalibration
    c_lo = accuracy - miscalibration
    if 0.5 <= c_hi <= 1.0:
        return float(c_hi)
    if 0.5 <= c_lo <= 1.0:
        return float(c_lo)
    raise ValueError(
        f"cannot realise ECE={miscalibration} at accuracy={accuracy}: "
        f"neither {c_hi} nor {c_lo} lies in [0.5, 1]"
    )


def _beta_mom(mean: float, sd: float) -> tuple[float, float]:
    """Method-of-moments Beta(α, β) matching mean and sd on (0, 1)."""
    mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
    var = float(sd) ** 2
    max_var = mean * (1.0 - mean) * 0.99
    var = min(max(var, 1e-12), max_var)
    common = mean * (1.0 - mean) / var - 1.0
    alpha = max(mean * common, 1e-3)
    beta = max((1.0 - mean) * common, 1e-3)
    return float(alpha), float(beta)


def simulate_stratum(
    n: int,
    accuracy: float,
    miscalibration: float,
    rng: np.random.Generator,
    *,
    label_noise: float = 0.0,
    difficulty_sd: float = 0.0,
    tier_leakage: float = 0.0,
    other_accuracy: float | None = None,
    other_miscalibration: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate one binary stratum with known accuracy and target top-label ECE.

    Clean path (all noise = 0)
    -------------------------
    All items share confidence ``c = confidence_for_target(...)``. Exactly
    ``round(n * accuracy)`` labels are positive so finite-n accuracy matches.
    ECE → |accuracy - c| as n → ∞.

    Noise sources (Prompt F1)
    -------------------------
    label_noise :
        With this probability flip the ground-truth label (annotator error).
    difficulty_sd :
        Draw per-item accuracy from a Beta with the stratum mean and this sd,
        then sample correctness from it (within-tier heterogeneity).
    tier_leakage :
        With this probability draw the item from the OTHER stratum's
        (accuracy, miscalibration) — models tier-misassignment. Requires
        ``other_accuracy`` / ``other_miscalibration``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    for name, v in (
        ("label_noise", label_noise),
        ("difficulty_sd", difficulty_sd),
        ("tier_leakage", tier_leakage),
    ):
        if not 0.0 <= float(v) <= 1.0 and name != "difficulty_sd":
            raise ValueError(f"{name} must be in [0, 1], got {v}")
        if name == "difficulty_sd" and float(v) < 0.0:
            raise ValueError(f"difficulty_sd must be >= 0, got {v}")

    clean = (
        float(label_noise) == 0.0
        and float(difficulty_sd) == 0.0
        and float(tier_leakage) == 0.0
    )
    if clean:
        c = confidence_for_target(accuracy, miscalibration)
        n_correct = int(round(n * accuracy))
        n_correct = min(max(n_correct, 0), n)
        labels = np.zeros(n, dtype=int)
        if n_correct:
            idx = rng.choice(n, size=n_correct, replace=False)
            labels[idx] = 1
        probs = np.full(n, c, dtype=float)
        return probs, labels

    if tier_leakage > 0.0 and (
        other_accuracy is None or other_miscalibration is None
    ):
        raise ValueError(
            "tier_leakage > 0 requires other_accuracy and other_miscalibration"
        )

    probs = np.empty(n, dtype=float)
    labels = np.empty(n, dtype=int)
    for i in range(n):
        if tier_leakage > 0.0 and rng.random() < tier_leakage:
            acc_m = float(other_accuracy)  # type: ignore[arg-type]
            mis = float(other_miscalibration)  # type: ignore[arg-type]
        else:
            acc_m = float(accuracy)
            mis = float(miscalibration)
        c = confidence_for_target(acc_m, mis)
        if difficulty_sd > 0.0:
            a, b = _beta_mom(acc_m, difficulty_sd)
            a_i = float(rng.beta(a, b))
        else:
            a_i = acc_m
        y = 1 if rng.random() < a_i else 0
        if label_noise > 0.0 and rng.random() < label_noise:
            y = 1 - y
        probs[i] = c
        labels[i] = y
    return probs, labels


def calibrate_simulator(
    *,
    n: int = CALIBRATE_N,
    targets: tuple[float, ...] = CALIBRATE_TARGETS,
    accuracy: float = CALIBRATE_ACCURACY,
    tol: float = CALIBRATE_TOL,
    seed: int = 0,
) -> list[SimulatorCalibrationRow]:
    """Verify requesting target ECE t yields measured ECE ≈ t at large n."""
    rng = np.random.default_rng(seed)
    rows: list[SimulatorCalibrationRow] = []
    for t in targets:
        probs, labels = simulate_stratum(n, accuracy, t, rng)
        measured = float(
            expected_calibration_error(probs, labels, n_bins=10, strategy="uniform").ece
        )
        err = abs(measured - t)
        rows.append(
            SimulatorCalibrationRow(
                target_ece=t,
                measured_ece=measured,
                abs_error=err,
                n=n,
                accuracy=accuracy,
            )
        )
        if err > tol:
            raise AssertionError(
                f"simulator missed target ECE: requested {t}, measured {measured} "
                f"(|err|={err} > tol={tol}) at n={n}. A simulator that does not "
                f"hit its target invalidates every power number computed from it."
            )
    return rows


def _ece_constant_conf(accuracy: float, conf: float) -> float:
    return abs(float(accuracy) - float(conf))


def _interval_excludes_zero(low: float, high: float) -> bool:
    return high < 0.0 or low > 0.0


def _ece_scalar_uniform(scores: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Top-label ECE scalar (uniform bins, mean conf_bar) — hot path."""
    n = len(scores)
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.searchsorted(edges[1:-1], scores, side="right")
    counts = np.bincount(idx, minlength=n_bins)
    sum_s = np.bincount(idx, weights=scores, minlength=n_bins)
    sum_c = np.bincount(idx, weights=correct, minlength=n_bins)
    ece = 0.0
    for b in range(n_bins):
        c = int(counts[b])
        if c == 0:
            continue
        ece += (c / n) * abs(sum_s[b] / c - sum_c[b] / c)
    return float(ece)


def _conf_and_correct(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-label conf / correct for 1-D binary probs (matches metrics._for_ece)."""
    pred = (probs >= 0.5).astype(int)
    conf = np.maximum(probs, 1.0 - probs)
    correct = (pred == labels.astype(int)).astype(float)
    return conf, correct


def delta_ece_bootstrap_intervals(
    probs_hard: np.ndarray,
    labels_hard: np.ndarray,
    probs_easy: np.ndarray,
    labels_easy: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    n_bins: int = 10,
    alpha: float = 0.05,
    compute_bca: bool = True,
) -> dict[str, Any]:
    """Percentile (+ optional BCa) ΔECE intervals (fast scalar ECE path).

    Jackknife deletes one item at a time from the pooled hard∪easy set.
    Skip BCa on power curves (``compute_bca=False``); enable for coverage.
    """
    sh, ch = _conf_and_correct(probs_hard, labels_hard)
    se, ce = _conf_and_correct(probs_easy, labels_easy)
    point = _ece_scalar_uniform(sh, ch, n_bins) - _ece_scalar_uniform(se, ce, n_bins)

    rng = np.random.default_rng(seed)
    n_h, n_e = len(sh), len(se)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ih = rng.integers(0, n_h, size=n_h)
        ie = rng.integers(0, n_e, size=n_e)
        samples[b] = _ece_scalar_uniform(sh[ih], ch[ih], n_bins) - _ece_scalar_uniform(
            se[ie], ce[ie], n_bins
        )

    p_low = float(np.quantile(samples, alpha / 2.0))
    p_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    excl_p = _interval_excludes_zero(p_low, p_high)

    if not compute_bca:
        return {
            "point": point,
            "percentile": (p_low, p_high),
            "bca": (float("nan"), float("nan")),
            "z0": float("nan"),
            "acceleration": float("nan"),
            "excludes_zero_percentile": excl_p,
            "excludes_zero_bca": excl_p,  # unused when BCa off
        }

    jack = np.empty(n_h + n_e, dtype=float)
    for i in range(n_h):
        mask = np.ones(n_h, dtype=bool)
        mask[i] = False
        jack[i] = _ece_scalar_uniform(sh[mask], ch[mask], n_bins) - _ece_scalar_uniform(
            se, ce, n_bins
        )
    for j in range(n_e):
        mask = np.ones(n_e, dtype=bool)
        mask[j] = False
        jack[n_h + j] = _ece_scalar_uniform(sh, ch, n_bins) - _ece_scalar_uniform(
            se[mask], ce[mask], n_bins
        )

    b_low, b_high, z0, accel = _bca_endpoints(point, samples, jack, alpha=alpha)
    return {
        "point": point,
        "percentile": (p_low, p_high),
        "bca": (b_low, b_high),
        "z0": z0,
        "acceleration": accel,
        "excludes_zero_percentile": excl_p,
        "excludes_zero_bca": _interval_excludes_zero(b_low, b_high),
    }


def delta_ece_constant_conf_interval(
    *,
    n: int,
    accuracy_hard: float,
    conf_hard: float,
    accuracy_easy: float,
    conf_easy: float,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Stratified two-sample bootstrap CI for ΔECE under constant-conf strata.

    Used for power only when strata are clean (exact counts, constant conf).
    """
    rng = np.random.default_rng(seed)
    n_h = int(round(n * accuracy_hard))
    n_e = int(round(n * accuracy_easy))
    p_h = n_h / n
    p_e = n_e / n
    acc_h = rng.binomial(n, p_h, size=n_boot) / n
    acc_e = rng.binomial(n, p_e, size=n_boot) / n
    deltas = np.abs(acc_h - conf_hard) - np.abs(acc_e - conf_easy)
    low = float(np.quantile(deltas, 0.025))
    high = float(np.quantile(deltas, 0.975))
    point = _ece_constant_conf(p_h, conf_hard) - _ece_constant_conf(p_e, conf_easy)
    return point, low, high


def _constant_conf_resample_interval(
    labels_hard: np.ndarray,
    conf_hard: float,
    labels_easy: np.ndarray,
    conf_easy: float,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Bootstrap ΔECE when each stratum has a single confidence value.

    Resamples the realised label vectors (valid under constant conf). Vectorised.
    """
    yh = np.asarray(labels_hard, dtype=float)
    ye = np.asarray(labels_easy, dtype=float)
    n_h, n_e = len(yh), len(ye)
    point = abs(float(yh.mean()) - conf_hard) - abs(float(ye.mean()) - conf_easy)
    rng = np.random.default_rng(seed)
    ih = rng.integers(0, n_h, size=(n_boot, n_h))
    ie = rng.integers(0, n_e, size=(n_boot, n_e))
    acc_h = yh[ih].mean(axis=1)
    acc_e = ye[ie].mean(axis=1)
    deltas = np.abs(acc_h - conf_hard) - np.abs(acc_e - conf_easy)
    return point, float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))


def simulate_calibrated_stratum(
    n: int,
    accuracy: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Perfectly calibrated heterogeneous stratum (asymptotic ECE → 0)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if not 0.5 <= accuracy <= 1.0:
        raise ValueError(f"accuracy must be in [0.5, 1], got {accuracy}")
    half = min(accuracy - 0.5, 1.0 - accuracy)
    half = max(half, 1e-6)
    lo, hi = accuracy - half, accuracy + half
    probs = rng.uniform(lo, hi, n)
    labels = (rng.uniform(0.0, 1.0, n) < probs).astype(int)
    return probs, labels


def _power_worker(
    n: int,
    n_trials: int,
    n_boot: int,
    seed: int,
    label_noise: float,
    difficulty_sd: float,
    tier_leakage: float,
    null: bool,
    compute_bca: bool = False,
) -> dict[str, Any]:
    """One (n, noise) cell — designed for ProcessPoolExecutor."""
    rng = np.random.default_rng(seed)
    excl_p = 0
    excl_b = 0
    points: list[float] = []
    for _ in range(n_trials):
        trial_seed = int(rng.integers(0, 2**31 - 1))
        sim = np.random.default_rng(trial_seed)
        if null:
            ph, yh = simulate_calibrated_stratum(n, HARD_ACCURACY, sim)
            pe, ye = simulate_calibrated_stratum(n, EASY_ACCURACY, sim)
        else:
            ph, yh = simulate_stratum(
                n,
                HARD_ACCURACY,
                HARD_ECE,
                sim,
                label_noise=label_noise,
                difficulty_sd=difficulty_sd,
                tier_leakage=tier_leakage,
                other_accuracy=EASY_ACCURACY,
                other_miscalibration=EASY_ECE,
            )
            pe, ye = simulate_stratum(
                n,
                EASY_ACCURACY,
                EASY_ECE,
                sim,
                label_noise=label_noise,
                difficulty_sd=difficulty_sd,
                tier_leakage=tier_leakage,
                other_accuracy=HARD_ACCURACY,
                other_miscalibration=HARD_ECE,
            )
        # Constant-confidence fast path (exact counts OR noise with leakage=0).
        # Within-stratum conf is homogeneous whenever tier_leakage=0, so ECE
        # collapses to |acc - c| and bootstrap is a binary-mean resample.
        if (
            not null
            and tier_leakage == 0.0
            and float(np.ptp(ph)) < 1e-15
            and float(np.ptp(pe)) < 1e-15
        ):
            point, low, high = _constant_conf_resample_interval(
                yh.astype(float),
                float(ph[0]),
                ye.astype(float),
                float(pe[0]),
                n_boot=n_boot,
                seed=trial_seed + 1,
            )
            points.append(point)
            if _interval_excludes_zero(low, high):
                excl_p += 1
                excl_b += 1
            continue

        if (
            not null
            and label_noise == 0.0
            and difficulty_sd == 0.0
            and tier_leakage == 0.0
        ):
            point, low, high = delta_ece_constant_conf_interval(
                n=n,
                accuracy_hard=float(yh.mean()),
                conf_hard=float(ph[0]),
                accuracy_easy=float(ye.mean()),
                conf_easy=float(pe[0]),
                n_boot=n_boot,
                seed=trial_seed + 1,
            )
            points.append(point)
            if _interval_excludes_zero(low, high):
                excl_p += 1
                excl_b += 1
            continue

        iv = delta_ece_bootstrap_intervals(
            ph,
            yh,
            pe,
            ye,
            n_boot=n_boot,
            seed=trial_seed + 1,
            compute_bca=compute_bca,
        )
        points.append(float(iv["point"]))
        if iv["excludes_zero_percentile"]:
            excl_p += 1
        if compute_bca and iv["excludes_zero_bca"]:
            excl_b += 1
        elif not compute_bca and iv["excludes_zero_percentile"]:
            excl_b += 1

    return {
        "n_per_stratum": n,
        "n_trials": n_trials,
        "n_boot": n_boot,
        "label_noise": label_noise,
        "difficulty_sd": difficulty_sd,
        "tier_leakage": tier_leakage,
        "null": null,
        "rate_exclude_zero": excl_p / n_trials,
        "rate_exclude_zero_percentile": excl_p / n_trials,
        "rate_exclude_zero_bca": excl_b / n_trials,
        "mean_point_delta_ece": float(np.mean(points)) if points else float("nan"),
        "compute_bca": compute_bca,
    }


def run_power_curve(
    *,
    n_grid: tuple[int, ...] = N_GRID,
    n_trials: int = N_TRIALS_DEFAULT,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260921,
    null: bool = False,
    label_noise: float = 0.0,
    difficulty_sd: float = 0.0,
    tier_leakage: float = 0.0,
    max_workers: int = 8,
    compute_bca: bool = False,
) -> dict[str, Any]:
    """Estimate power (or null FPR) vs n-per-stratum."""
    with ProcessPoolExecutor(max_workers=min(max_workers, len(n_grid))) as pool:
        futs = {
            pool.submit(
                _power_worker,
                n,
                n_trials,
                n_boot,
                seed + i * 10_009,
                label_noise,
                difficulty_sd,
                tier_leakage,
                null,
                compute_bca,
            ): n
            for i, n in enumerate(n_grid)
        }
        by_n: dict[int, dict[str, Any]] = {}
        for fut in as_completed(futs):
            row = fut.result()
            by_n[int(row["n_per_stratum"])] = row
    curve = [by_n[n] for n in n_grid]
    return {
        "null": null,
        "n_trials": n_trials,
        "n_boot": n_boot,
        "seed": seed,
        "label_noise": label_noise,
        "difficulty_sd": difficulty_sd,
        "tier_leakage": tier_leakage,
        "curve": curve,
        "stratum_model": (
            "heterogeneous_calibrated"
            if null
            else "noisy_target_ece"
            if (label_noise or difficulty_sd or tier_leakage)
            else "constant_conf_target_ece"
        ),
    }


def run_power_surface(
    *,
    axis: Literal["label_noise", "difficulty_sd"],
    n_grid: tuple[int, ...] = N_GRID,
    n_trials: int = N_TRIALS_DEFAULT,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260921,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Power surface over (n, noise_param). Other axis fixed at pessimistic."""
    if axis == "label_noise":
        values = LABEL_NOISE_SWEEP
        fixed = {
            "difficulty_sd": PESSIMISTIC_DIFFICULTY_SD,
            "tier_leakage": PESSIMISTIC_TIER_LEAKAGE,
        }
    else:
        values = DIFFICULTY_SD_SWEEP
        fixed = {
            "label_noise": PESSIMISTIC_LABEL_NOISE,
            "tier_leakage": PESSIMISTIC_TIER_LEAKAGE,
        }

    cells: list[dict[str, Any]] = []
    jobs: list[tuple] = []
    for vi, v in enumerate(values):
        for ni, n in enumerate(n_grid):
            job_i = vi * 1000 + ni
            if axis == "label_noise":
                jobs.append(
                    (
                        n,
                        n_trials,
                        n_boot,
                        seed + job_i * 10_009,
                        float(v),
                        fixed["difficulty_sd"],
                        fixed["tier_leakage"],
                        False,
                        False,
                    )
                )
            else:
                jobs.append(
                    (
                        n,
                        n_trials,
                        n_boot,
                        seed + job_i * 10_009,
                        fixed["label_noise"],
                        float(v),
                        fixed["tier_leakage"],
                        False,
                        False,
                    )
                )

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_power_worker, *args) for args in jobs]
        for fut in as_completed(futs):
            cells.append(fut.result())

    if axis == "label_noise":
        cells.sort(key=lambda r: (r["label_noise"], r["n_per_stratum"]))
    else:
        cells.sort(key=lambda r: (r["difficulty_sd"], r["n_per_stratum"]))

    return {
        "axis": axis,
        "n_grid": list(n_grid),
        "values": list(values),
        "fixed": fixed,
        "n_trials": n_trials,
        "n_boot": n_boot,
        "seed": seed,
        "cells": cells,
    }


def cross_check_vs_delta_ece(*, n: int = 300, seed: int = 0) -> dict[str, Any]:
    """One-trial agreement: fast constant-conf CI vs full ``delta_ece``."""
    rng = np.random.default_rng(seed)
    ph, yh = simulate_stratum(n, HARD_ACCURACY, HARD_ECE, rng)
    pe, ye = simulate_stratum(n, EASY_ACCURACY, EASY_ECE, rng)
    full = delta_ece(ph, yh, pe, ye, seed=seed + 99, n_boot=2_000)
    point, low, high = delta_ece_constant_conf_interval(
        n=n,
        accuracy_hard=float(yh.mean()),
        conf_hard=float(ph[0]),
        accuracy_easy=float(ye.mean()),
        conf_easy=float(pe[0]),
        n_boot=2_000,
        seed=seed + 99,
    )
    return {
        "n": n,
        "delta_ece_point": full.point,
        "delta_ece_ci": [full.ci_low, full.ci_high],
        "delta_ece_ci_bca": [full.ci_bca_low, full.ci_bca_high],
        "fast_point": point,
        "fast_ci": [low, high],
        "point_match": abs(full.point - point) < 1e-12,
        "intervals_overlap": full.ci_low <= high and low <= full.ci_high,
    }


def choose_n(
    power_curve: list[dict[str, Any]], threshold: float = POWER_THRESHOLD
) -> int | None:
    for row in power_curve:
        if row["rate_exclude_zero"] >= threshold:
            return int(row["n_per_stratum"])
    return None


def _is_step_function(curve: list[dict[str, Any]], *, jump_tol: float = 0.85) -> bool:
    """Detect a near 0→1 jump across adjacent n (simulator bug signal)."""
    rates = [float(r["rate_exclude_zero"]) for r in curve]
    for a, b in zip(rates, rates[1:]):
        if a < 0.15 and b > jump_tol:
            return True
    return False


def estimate_budget(n_per_stratum: int, *, repeats: int = REPEATS_DEFAULT) -> dict[str, Any]:
    """Label count + cost.estimate() split (Jev / baseline / total)."""
    total_items = 2 * n_per_stratum
    split = cost_estimate(
        snapshot_date=PRICE_SNAPSHOT,
        n_items=total_items,
        repeats=repeats,
        baseline_model=BASELINE_MODEL,
        pilot_source="default_pilot_means_20call",
    )
    d = split.to_dict()
    return {
        "n_per_stratum": n_per_stratum,
        "total_items_to_label": total_items,
        "repeats": repeats,
        "api_calls_jev_plus_adapter": total_items * repeats * 2,
        "pricing_snapshot": PRICE_SNAPSHOT,
        "baseline_model": BASELINE_MODEL,
        "estimated_jev_usd": d["jev_usd"],
        "estimated_baseline_adapter_usd": d["baseline_usd"],
        "estimated_total_usd": d["total_usd"],
        "split_line": d["split_line"],
        "jev_input_tokens": d["jev_input_tokens"],
        "baseline_input_tokens": d["baseline_input_tokens"],
        "baseline_output_tokens": d["baseline_output_tokens"],
        "pilot_source": d["pilot_source"],
        "note": (
            "Baseline dominates spend. Token means are from DEFAULT_PILOT "
            "(structured-output overhead ~500 in / ~80 out). Replace with a "
            "measured 20-call pilot before claiming a budget in the README. "
            "Jev output tokens are free and never billed."
        ),
        "cost_split": d,
    }


def _coverage_shard(
    n_per_stratum: int,
    n_t: int,
    n_boot: int,
    seed: int,
    shard_i: int,
) -> dict[str, int]:
    """Module-level worker for coverage (must pickle)."""
    row = _power_worker(
        n_per_stratum,
        n_t,
        n_boot,
        seed + shard_i * 10_009,
        0.0,
        0.0,
        0.0,
        True,
        True,
    )
    return {
        "excl_p": int(round(row["rate_exclude_zero_percentile"] * n_t)),
        "excl_b": int(round(row["rate_exclude_zero_bca"] * n_t)),
        "n": n_t,
    }


def run_coverage_study(
    *,
    n_per_stratum: int,
    n_trials: int = N_TRIALS_COVERAGE,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260922,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Null FPR for percentile and BCa at fixed n (Prompt F2)."""
    n_workers = min(max_workers, max(1, n_trials // 50))
    base = n_trials // n_workers
    rem = n_trials % n_workers
    shards = [base + (1 if i < rem else 0) for i in range(n_workers)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futs = [
            pool.submit(
                _coverage_shard, n_per_stratum, shards[i], n_boot, seed, i
            )
            for i in range(n_workers)
            if shards[i] > 0
        ]
        excl_p = excl_b = total = 0
        for fut in as_completed(futs):
            r = fut.result()
            excl_p += r["excl_p"]
            excl_b += r["excl_b"]
            total += r["n"]

    def _fpr_block(excl: int, method: str) -> dict[str, Any]:
        fpr = excl / total
        se = float(np.sqrt(fpr * (1.0 - fpr) / total))
        ci = [max(0.0, fpr - 1.96 * se), min(1.0, fpr + 1.96 * se)]
        return {
            "method": method,
            "trials": total,
            "empirical_fpr": fpr,
            "empirical_fpr_se": se,
            "empirical_fpr_ci95": ci,
            "empirical_coverage": 1.0 - fpr,
            "nominal_alpha": 0.05,
            "near_nominal": abs(fpr - 0.05) <= 0.01,
        }

    percentile = _fpr_block(excl_p, "percentile")
    bca = _fpr_block(excl_b, "bca")

    # Primary selection
    if bca["near_nominal"] and not percentile["near_nominal"]:
        primary = "bca"
        reason = (
            "BCa empirical FPR is within 0.01 of nominal 0.05 while percentile "
            "is not; BCa corrects bootstrap bias/skew for ECE."
        )
    elif percentile["near_nominal"] and bca["near_nominal"]:
        primary = "bca"
        reason = (
            "Both intervals are near nominal; BCa is primary because ECE is "
            "positively biased and boundary-constrained."
        )
    elif abs(bca["empirical_fpr"] - 0.05) < abs(percentile["empirical_fpr"] - 0.05):
        primary = "bca"
        reason = (
            "Neither reaches nominal; BCa is closer to 0.05. Report empirical "
            f"coverage {bca['empirical_coverage']:.1%} — do not claim 95%."
        )
    else:
        primary = "percentile"
        reason = (
            "Neither reaches nominal; percentile is closer. Report empirical "
            f"coverage {percentile['empirical_coverage']:.1%} — do not claim 95%."
        )

    return {
        "schema": "jevbench.coverage.v1",
        "n_per_stratum": n_per_stratum,
        "n_boot": n_boot,
        "seed": seed,
        "null_model": "heterogeneous_calibrated",
        "percentile": percentile,
        "bca": bca,
        "primary_method": primary,
        "primary_reason": reason,
        "claim_95": (
            (primary == "bca" and bca["near_nominal"])
            or (primary == "percentile" and percentile["near_nominal"])
        ),
    }


def render_power_curve_chart(
    *,
    power_curve: list[dict[str, Any]],
    null_curve: list[dict[str, Any]],
    chosen_n: int | None,
    out_dir: Path,
    title_suffix: str = "",
) -> dict[str, Path]:
    from jevbench.charts import make_context

    ctx = make_context(
        run_id="power-analysis",
        resolved_model="simulator",
        out_dir=out_dir,
        extra_footer=(
            "offline power · target ΔECE=0.09 · pessimistic noise corner"
        ),
    )
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    brand = ctx.style.brand
    ns = [r["n_per_stratum"] for r in power_curve]
    pows = [r["rate_exclude_zero"] for r in power_curve]
    fprs = [r["rate_exclude_zero"] for r in null_curve]

    ax.plot(
        ns,
        pows,
        color=ctx.style.series("jev")["color"],
        marker="o",
        markersize=12,
        linewidth=2.5,
        label="power (ΔECE=0.09, pessimistic noise)",
    )
    ax.plot(
        ns,
        fprs,
        color=ctx.style.series("adapter")["color"],
        marker="s",
        markersize=10,
        linestyle="--",
        linewidth=2.5,
        label="false positive rate (null)",
    )
    ax.axhline(
        POWER_THRESHOLD,
        color=brand["muted"],
        linestyle=":",
        linewidth=2.0,
        label=f"power target ({POWER_THRESHOLD:.0%})",
    )
    ax.axhline(
        0.05,
        color=ctx.style.tokens["reference"]["color"],
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
    )
    if chosen_n is not None:
        ax.axvline(
            chosen_n,
            color=ctx.style.series("jev")["color"],
            linestyle="-.",
            linewidth=2.0,
            label=f"chosen n = {chosen_n}",
        )
    ax.set_xlabel("n per stratum")
    ax.set_ylabel("Fraction of 95% CIs excluding zero")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Power to detect ΔECE = 0.09" + title_suffix)
    ax.legend(loc="best", frameon=True)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, "power_curve")


def run_power_analysis(
    *,
    out_dir: Path | None = None,
    n_trials: int = N_TRIALS_DEFAULT,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260921,
    write_chart: bool = True,
    run_surfaces: bool = True,
    run_coverage: bool = True,
    coverage_trials: int = N_TRIALS_COVERAGE,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Full offline power analysis → results/power_analysis.json (+ coverage)."""
    t0 = time.perf_counter()
    out_dir = out_dir or RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = calibrate_simulator(seed=seed)
    cross = cross_check_vs_delta_ece(n=300, seed=seed)
    if not cross["point_match"] or not cross["intervals_overlap"]:
        raise RuntimeError(
            f"fast power bootstrap disagrees with delta_ece: {cross}. "
            "Stop — fix the estimator/simulator before spending."
        )

    # Decision curve at pessimistic corner
    power = run_power_curve(
        n_trials=n_trials,
        n_boot=n_boot,
        seed=seed,
        null=False,
        label_noise=PESSIMISTIC_LABEL_NOISE,
        difficulty_sd=PESSIMISTIC_DIFFICULTY_SD,
        tier_leakage=PESSIMISTIC_TIER_LEAKAGE,
        max_workers=max_workers,
    )
    if _is_step_function(power["curve"]):
        raise RuntimeError(
            "Power curve is still a step function after adding label_noise, "
            "difficulty_sd, and tier_leakage. Simulator bug — stop and find it. "
            f"rates={[r['rate_exclude_zero'] for r in power['curve']]}"
        )

    null = run_power_curve(
        n_trials=n_trials,
        n_boot=n_boot,
        seed=seed + 1,
        null=True,
        max_workers=max_workers,
    )
    chosen = choose_n(power["curve"])
    fpr_at_chosen = None
    mean_fpr = float(np.mean([r["rate_exclude_zero"] for r in null["curve"]]))
    if chosen is not None:
        for row in null["curve"]:
            if row["n_per_stratum"] == chosen:
                fpr_at_chosen = row["rate_exclude_zero"]
                break

    underpowered = False
    underpowered_note = None
    if chosen is None:
        underpowered = True
        underpowered_note = (
            "No n in the grid reached power >= 0.80 under the pessimistic "
            "corner. Do not proceed underpowered — pre-specify a larger "
            "detectable effect size or extend labelling capacity."
        )
    elif chosen > LABEL_BUDGET_CEILING:
        underpowered = True
        underpowered_note = (
            f"Required n={chosen} per stratum exceeds the labelling ceiling "
            f"({LABEL_BUDGET_CEILING}). Say so plainly and pre-specify a larger "
            "detectable effect size instead of proceeding underpowered."
        )

    surfaces: dict[str, Any] = {}
    if run_surfaces and n_trials >= 100:
        surfaces["label_noise"] = run_power_surface(
            axis="label_noise",
            n_trials=n_trials,
            n_boot=n_boot,
            seed=seed + 100,
            max_workers=max_workers,
        )
        surfaces["difficulty_sd"] = run_power_surface(
            axis="difficulty_sd",
            n_trials=n_trials,
            n_boot=n_boot,
            seed=seed + 200,
            max_workers=max_workers,
        )

    coverage: dict[str, Any] | None = None
    if run_coverage and n_trials >= 100:
        cov_n = chosen if chosen is not None else 200
        coverage = run_coverage_study(
            n_per_stratum=cov_n,
            n_trials=coverage_trials,
            n_boot=n_boot,
            seed=seed + 300,
            max_workers=max_workers,
        )
        cov_path = out_dir / "coverage.json"
        cov_path.write_text(
            json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    if mean_fpr > 0.10 and n_trials >= 200:
        raise RuntimeError(
            f"Empirical false positive rate is materially above 0.05 "
            f"(mean FPR across n-grid = {mean_fpr:.3f}). The method is broken "
            f"— no amount of data fixes it. Stop and fix the estimator before "
            f"spending anything."
        )

    budget = estimate_budget(chosen) if chosen is not None else None
    chart_paths: dict[str, str] = {}
    paths: dict[str, Path] = {}
    if write_chart:
        paths = render_power_curve_chart(
            power_curve=power["curve"],
            null_curve=null["curve"],
            chosen_n=chosen,
            out_dir=out_dir,
            title_suffix=(
                f" (noise ln={PESSIMISTIC_LABEL_NOISE}, "
                f"sd={PESSIMISTIC_DIFFICULTY_SD})"
            ),
        )

        def _rel(p: Path) -> str:
            try:
                return str(p.relative_to(REPO))
            except ValueError:
                return str(p)

        chart_paths = {k: _rel(v) for k, v in paths.items()}

    payload: dict[str, Any] = {
        "schema": "jevbench.power_analysis.v2",
        "effect_size_justification": (
            "Two published Jev benchmarks report roughly ECE 0.05–0.07 at ~92% "
            "accuracy and ~0.154 at ~63% accuracy — a gap of about 0.09. The "
            "effect size is therefore anchored to the magnitude that would "
            "actually explain the disagreement in the existing literature, not "
            "chosen by us."
        ),
        "target_delta_ece": TARGET_DELTA_ECE,
        "easy": {"accuracy": EASY_ACCURACY, "target_ece": EASY_ECE},
        "hard": {"accuracy": HARD_ACCURACY, "target_ece": HARD_ECE},
        "noise": {
            "label_noise_sweep": list(LABEL_NOISE_SWEEP),
            "difficulty_sd_sweep": list(DIFFICULTY_SD_SWEEP),
            "tier_leakage_sweep": list(TIER_LEAKAGE_SWEEP),
            "pessimistic": {
                "label_noise": PESSIMISTIC_LABEL_NOISE,
                "difficulty_sd": PESSIMISTIC_DIFFICULTY_SD,
                "tier_leakage": PESSIMISTIC_TIER_LEAKAGE,
            },
            "decision_rule": (
                "Required n = smallest n with power >= 0.80 under "
                "label_noise=0.05 AND difficulty_sd=0.10 (pessimistic corner)."
            ),
        },
        "n_boot_power": n_boot,
        "n_boot_real_analysis": N_BOOT_REAL,
        "n_trials": n_trials,
        "seed": seed,
        "simulator_calibration": [asdict(r) for r in calib],
        "cross_check_vs_delta_ece": cross,
        "power_curve": power["curve"],
        "null_curve": null["curve"],
        "power_surfaces": surfaces,
        "mean_null_fpr": mean_fpr,
        "fpr_at_chosen_n": fpr_at_chosen,
        "chosen_n_per_stratum": chosen,
        "power_threshold": POWER_THRESHOLD,
        "underpowered": underpowered,
        "underpowered_note": underpowered_note,
        "label_budget_ceiling": LABEL_BUDGET_CEILING,
        "coverage": coverage,
        "budget": budget,
        "charts": chart_paths,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }

    out_json = out_dir / "power_analysis.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if write_chart and "1080p" in paths:
        stable = out_dir / "power_curve.png"
        stable.write_bytes(paths["1080p"].read_bytes())
        try:
            payload["charts"]["power_curve_png"] = str(stable.relative_to(REPO))
        except ValueError:
            payload["charts"]["power_curve_png"] = str(stable)
        out_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    return payload


def format_recommendation(payload: dict[str, Any]) -> str:
    chosen = payload.get("chosen_n_per_stratum")
    budget = payload.get("budget") or {}
    fpr = payload.get("mean_null_fpr")
    cov = payload.get("coverage") or {}
    if payload.get("underpowered"):
        return (
            f"DECISION: UNDERPOWERED. {payload.get('underpowered_note')} "
            f"Mean null FPR={fpr}."
        )
    if chosen is None:
        return (
            "DECISION: no n in the grid reached power "
            f">={POWER_THRESHOLD:.0%}. Extend N_GRID or revisit the design. "
            f"Mean null FPR={fpr:.3f}."
        )
    split = budget.get("split_line", "")
    cov_line = ""
    if cov:
        prim = cov.get("primary_method")
        block = cov.get(prim, {})
        cov_line = (
            f" Coverage primary={prim} FPR={block.get('empirical_fpr')} "
            f"(empirical coverage {block.get('empirical_coverage')}); "
            f"claim_95={cov.get('claim_95')}."
        )
    return (
        f"DECISION: label n={chosen} per stratum "
        f"({budget.get('total_items_to_label')} items total) before any paid run. "
        f"Estimated API calls (jev+adapter, {budget.get('repeats')} repeats)="
        f"{budget.get('api_calls_jev_plus_adapter')}. "
        f"{split}. "
        f"Mean null FPR={fpr:.3f}.{cov_line} "
        f"Use n_boot={N_BOOT_REAL} for the real analysis; report BOTH "
        f"percentile and BCa intervals."
    )


# ---------------------------------------------------------------------------
# Prompt J — entropy-dependent (asymmetric) label noise, both scoring rules
# ---------------------------------------------------------------------------

NLI_ENTROPY_MAX = float(np.log2(3))
ASYMMETRIC_NOISE_SCALE = 0.5


def noise_rate_from_entropy(
    entropy: ArrayLike,
    *,
    scale: float = ASYMMETRIC_NOISE_SCALE,
) -> np.ndarray:
    """Label-flip probability as a function of item entropy.

    ``rate = scale * clip(entropy / log2(3), 0, 1)``. Low-entropy items
    (the easy stratum) are barely flipped. High-entropy items (the hard
    stratum) approach ``scale``. The noise is asymmetric across strata
    because the strata are defined by entropy.
    """
    if not 0.0 <= float(scale) <= 1.0:
        raise ValueError(f"scale must be in [0, 1], got {scale}")
    h = np.asarray(entropy, dtype=float)
    return float(scale) * np.clip(h / NLI_ENTROPY_MAX, 0.0, 1.0)


def majority_share_from_entropy(entropy: ArrayLike) -> np.ndarray:
    """Annotator mass on the majority class, monotone in entropy.

    Entropy 0 → share 1. Entropy ``log2(3)`` → share 0.5. Used only to
    build the soft-correctness target; the hard label is a separate flip.
    """
    t = np.clip(np.asarray(entropy, dtype=float) / NLI_ENTROPY_MAX, 0.0, 1.0)
    return 1.0 - 0.5 * t


@dataclass(frozen=True)
class AsymmetricDraw:
    """One stratum under entropy-dependent label noise."""

    probs: np.ndarray
    labels_hard: np.ndarray
    soft_correct: np.ndarray
    entropy: np.ndarray
    noise_rate: np.ndarray


def simulate_asymmetric_stratum(
    n: int,
    accuracy: float,
    miscalibration: float,
    rng: np.random.Generator,
    *,
    entropy_range: tuple[float, float],
    noise_scale: float = ASYMMETRIC_NOISE_SCALE,
) -> AsymmetricDraw:
    """Constant-confidence stratum plus asymmetric label noise.

    The model predicts class 1 at confidence ``c = confidence_for_target``.
    A latent label equals that class on ``round(n * accuracy)`` items.
    Each item draws an entropy uniformly from ``entropy_range`` and:

    * hard label — the latent label, flipped with probability
      :func:`noise_rate_from_entropy` (the asymmetric noise);
    * soft correctness — annotator mass on the predicted class, from
      :func:`majority_share_from_entropy`. Soft correctness does not use
      the flipped hard label.

    Because the flip rate is higher where entropy is higher, hard scoring
    puts label noise in the hard stratum only, which inflates ΔECE in the
    direction of the hypothesis. Soft scoring is the primary rule.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    lo, hi = float(entropy_range[0]), float(entropy_range[1])
    if not 0.0 <= lo <= hi <= NLI_ENTROPY_MAX + 1e-9:
        raise ValueError(f"entropy_range {entropy_range} outside [0, log2(3)]")
    c = confidence_for_target(accuracy, miscalibration)
    n_latent = int(round(n * accuracy))
    n_latent = min(max(n_latent, 0), n)
    latent = np.zeros(n, dtype=int)
    if n_latent:
        latent[rng.choice(n, size=n_latent, replace=False)] = 1
    entropy = rng.uniform(lo, hi, n)
    rate = noise_rate_from_entropy(entropy, scale=noise_scale)
    flip = rng.random(n) < rate
    labels = latent.copy()
    labels[flip] = 1 - labels[flip]
    share = majority_share_from_entropy(entropy)
    # Predicted class is 1 (c >= 0.5 by confidence_for_target's range).
    soft = np.where(latent == 1, share, 1.0 - share)
    return AsymmetricDraw(
        probs=np.full(n, c, dtype=float),
        labels_hard=labels,
        soft_correct=soft.astype(float),
        entropy=entropy.astype(float),
        noise_rate=rate.astype(float),
    )


def _constant_conf_delta(
    correct_h: np.ndarray,
    conf_h: float,
    correct_e: np.ndarray,
    conf_e: float,
    *,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, float | bool]:
    """Percentile CI for ΔECE when every item in a stratum shares one confidence.

    All mass sits in one bin, so ECE = |mean(correctness) − confidence|.
    Works for 0/1 hard labels and for soft correctness in [0, 1].
    """
    point = abs(float(correct_h.mean()) - conf_h) - abs(float(correct_e.mean()) - conf_e)
    n_h, n_e = len(correct_h), len(correct_e)
    ih = rng.integers(0, n_h, size=(n_boot, n_h))
    ie = rng.integers(0, n_e, size=(n_boot, n_e))
    deltas = np.abs(correct_h[ih].mean(axis=1) - conf_h) - np.abs(
        correct_e[ie].mean(axis=1) - conf_e
    )
    low = float(np.quantile(deltas, 0.025))
    high = float(np.quantile(deltas, 0.975))
    return {
        "point": float(point),
        "ci_low": low,
        "ci_high": high,
        "excludes_zero": bool(high < 0.0 or low > 0.0),
    }


def _entropy_ranges_from_lock(repo: Path | None = None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Easy/hard entropy windows from the committed quantile lock."""
    root = repo or REPO
    path = root / "datasets" / "chaosnli" / "thresholds.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; lock ChaosNLI thresholds before the power rerun"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    q_easy = float(payload["q_easy"])
    q_hard = float(payload["q_hard"])
    return (0.0, q_easy), (q_hard, NLI_ENTROPY_MAX)


def run_asymmetric_power(
    *,
    n: int = 750,
    n_trials: int = 400,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260923,
    noise_scale: float = ASYMMETRIC_NOISE_SCALE,
    entropy_easy: tuple[float, float] | None = None,
    entropy_hard: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Power at one n under asymmetric label noise, hard scoring and soft scoring.

    Alternative: latent ΔECE target 0.09 (same literature anchor as the rest
    of this module), then entropy-dependent flips. Null: both strata
    calibrated to their latent labels (miscalibration 0) with the same
    asymmetric noise, so any positive ΔECE is the scoring artifact.
    """
    if entropy_easy is None or entropy_hard is None:
        entropy_easy, entropy_hard = _entropy_ranges_from_lock()
    rng = np.random.default_rng(seed)
    cells: dict[str, Any] = {}
    for setting, mis_h, mis_e in (
        ("alternative", HARD_ECE, EASY_ECE),
        ("null", 0.0, 0.0),
    ):
        for scoring in ("hard", "soft"):
            points: list[float] = []
            excl = 0
            for _ in range(n_trials):
                trial = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                hard = simulate_asymmetric_stratum(
                    n, HARD_ACCURACY, mis_h, trial,
                    entropy_range=entropy_hard, noise_scale=noise_scale,
                )
                easy = simulate_asymmetric_stratum(
                    n, EASY_ACCURACY, mis_e, trial,
                    entropy_range=entropy_easy, noise_scale=noise_scale,
                )
                ch = hard.labels_hard.astype(float) if scoring == "hard" else hard.soft_correct
                ce = easy.labels_hard.astype(float) if scoring == "hard" else easy.soft_correct
                iv = _constant_conf_delta(
                    ch, float(hard.probs[0]), ce, float(easy.probs[0]),
                    n_boot=n_boot, rng=trial,
                )
                points.append(float(iv["point"]))
                excl += int(iv["excludes_zero"])
            cells[f"{setting}_{scoring}"] = {
                "setting": setting,
                "scoring": scoring,
                "n_per_stratum": n,
                "n_trials": n_trials,
                "n_boot": n_boot,
                "ci_method": "percentile",
                "rate_exclude_zero": excl / n_trials,
                "mean_point_delta_ece": float(np.mean(points)),
            }
    hard_alt = cells["alternative_hard"]["rate_exclude_zero"]
    soft_alt = cells["alternative_soft"]["rate_exclude_zero"]
    hard_null = cells["null_hard"]["mean_point_delta_ece"]
    soft_null = cells["null_soft"]["mean_point_delta_ece"]
    return {
        "schema": "jevbench.asymmetric_power.v1",
        "n_per_stratum": n,
        "n_trials": n_trials,
        "n_boot": n_boot,
        "seed": seed,
        "noise_scale": noise_scale,
        "noise_rate": "scale * clip(entropy / log2(3), 0, 1)",
        "entropy_easy": list(entropy_easy),
        "entropy_hard": list(entropy_hard),
        "target_delta_ece": TARGET_DELTA_ECE,
        "cells": cells,
        "power_hard": hard_alt,
        "power_soft": soft_alt,
        "null_mean_delta_ece_hard": hard_null,
        "null_mean_delta_ece_soft": soft_null,
        "note": (
            "Hard scoring puts label noise in the hard stratum only, which "
            "inflates ΔECE in the direction of the hypothesis. Compare "
            "null_mean_delta_ece_hard with null_mean_delta_ece_soft: the hard "
            "rule's null mean sits higher when the artifact is present. "
            "Soft scoring is primary."
        ),
    }


def _mean_interval_covers(
    x: np.ndarray,
    truth: float,
    *,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict[str, bool]:
    """Percentile and BCa CIs for the mean of one stratum. Coverage vs ``truth``."""
    n = len(x)
    point = float(x.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = x[idx].mean(axis=1)
    p_low = float(np.quantile(samples, alpha / 2.0))
    p_high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    jack = (float(x.sum()) - x) / (n - 1)
    b_low, b_high, _z0, _a = _bca_endpoints(point, samples, jack, alpha=alpha)
    return {
        "percentile": bool(p_low <= truth <= p_high),
        "bca": bool(b_low <= truth <= b_high),
    }


def run_stratum_mean_coverage(
    *,
    n: int = 750,
    n_trials: int = N_TRIALS_COVERAGE,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260923,
    accuracy: float = HARD_ACCURACY,
) -> dict[str, Any]:
    """5000-trial null coverage for the MEAN of one calibrated stratum.

    Same generator as the ΔECE null's hard stratum
    (:func:`simulate_calibrated_stratum`). Every probability is at least 0.5,
    so the predicted class is 1 and per-item correctness is the label.
    The population mean of that correctness is ``accuracy``.
    """
    cover_p = 0
    cover_b = 0
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t * 1_000_003)
        _probs, labels = simulate_calibrated_stratum(n, accuracy, rng)
        # Guard: this generator's support is [accuracy-half, accuracy+half]
        # with half <= accuracy-0.5, so probs >= 0.5 and pred is class 1.
        hit = _mean_interval_covers(
            labels.astype(float),
            accuracy,
            n_boot=n_boot,
            rng=rng,
        )
        cover_p += int(hit["percentile"])
        cover_b += int(hit["bca"])

    def _block(covers: int, method: str) -> dict[str, Any]:
        cov = covers / n_trials
        fpr = 1.0 - cov
        se = float(np.sqrt(fpr * (1.0 - fpr) / n_trials))
        return {
            "method": method,
            "statistic": "mean",
            "trials": n_trials,
            "empirical_coverage": cov,
            "empirical_fpr": fpr,
            "empirical_fpr_se": se,
            "nominal_alpha": 0.05,
            "near_nominal": abs(fpr - 0.05) <= 0.01,
        }

    return {
        "n": n,
        "n_boot": n_boot,
        "seed": seed,
        "truth": accuracy,
        "null_model": "heterogeneous_calibrated_stratum_mean",
        "percentile": _block(cover_p, "percentile"),
        "bca": _block(cover_b, "bca"),
    }


def explain_bca_gap(mean_cov: dict[str, Any], delta_cov: dict[str, Any]) -> dict[str, Any]:
    """Which story the coverage numbers support.

    Two candidates for BCa undercoverage on ΔECE:

    * ``bca_on_the_mean`` — BCa misses nominal coverage even for a stratum
      mean, so the problem is the interval method.
    * ``delta_ece_statistic`` — the mean is near nominal under both
      intervals, and ΔECE is not. The problem is the binned ΔECE functional.
    """
    m_bca = float(mean_cov["bca"]["empirical_coverage"])
    m_pct = float(mean_cov["percentile"]["empirical_coverage"])
    d_bca = float(delta_cov["bca"]["empirical_coverage"])
    d_pct = float(delta_cov["percentile"]["empirical_coverage"])
    mean_bca_ok = bool(mean_cov["bca"]["near_nominal"])
    delta_bca_ok = bool(delta_cov["bca"]["near_nominal"])
    mean_short = 0.95 - m_bca
    delta_short = 0.95 - d_bca
    if mean_bca_ok and not delta_bca_ok:
        which = "delta_ece_statistic"
        text = (
            f"The mean of a stratum is near nominal under BCa "
            f"(coverage {m_bca:.1%}, percentile {m_pct:.1%}), while ΔECE BCa "
            f"coverage is {d_bca:.1%} (percentile {d_pct:.1%}). The data support "
            "the ΔECE explanation: BCa on a simple mean is fine, and the "
            "shortfall is in the binned ΔECE statistic."
        )
    elif (not mean_bca_ok) and mean_short >= 0.5 * max(delta_short, 0.0) and mean_short > 0.01:
        which = "bca_on_the_mean"
        text = (
            f"BCa coverage of the stratum mean is {m_bca:.1%} "
            f"(percentile {m_pct:.1%}), short of 95% by {mean_short:.1%}. "
            f"ΔECE BCa coverage is {d_bca:.1%} (percentile {d_pct:.1%}). "
            "The data support the BCa explanation: the interval method "
            "undercovers a mean, so the ΔECE shortfall is not evidence of a "
            "problem that appears only for ΔECE."
        )
    else:
        which = "inconclusive"
        text = (
            f"Mean coverage percentile {m_pct:.1%} / BCa {m_bca:.1%}; "
            f"ΔECE coverage percentile {d_pct:.1%} / BCa {d_bca:.1%}. "
            "Neither explanation dominates. Report both coverages and do not "
            "claim a 95% interval."
        )
    return {
        "explanation": which,
        "text": text,
        "mean_coverage_percentile": m_pct,
        "mean_coverage_bca": m_bca,
        "delta_ece_coverage_percentile": d_pct,
        "delta_ece_coverage_bca": d_bca,
    }


def run_bca_diagnostic(
    *,
    n: int = 750,
    n_trials: int = N_TRIALS_COVERAGE,
    n_boot: int = N_BOOT_POWER,
    seed: int = 20260923,
    max_workers: int = 8,
) -> dict[str, Any]:
    """5000-trial null: coverage of a stratum mean and of ΔECE, percentile and BCa."""
    mean_cov = run_stratum_mean_coverage(
        n=n, n_trials=n_trials, n_boot=n_boot, seed=seed
    )
    delta_cov = run_coverage_study(
        n_per_stratum=n,
        n_trials=n_trials,
        n_boot=n_boot,
        seed=seed + 17,
        max_workers=max_workers,
    )
    explanation = explain_bca_gap(mean_cov, delta_cov)
    return {
        "schema": "jevbench.bca_diagnostic.v1",
        "n_per_stratum": n,
        "n_trials": n_trials,
        "n_boot": n_boot,
        "seed": seed,
        "mean": mean_cov,
        "delta_ece": {
            "percentile": delta_cov["percentile"],
            "bca": delta_cov["bca"],
            "null_model": delta_cov["null_model"],
        },
        "explanation": explanation,
    }
