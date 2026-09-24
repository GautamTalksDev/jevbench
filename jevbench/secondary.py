"""Amendment 10 secondary analyses (PROMPT S) — exploratory unless named primary.

S1 Ambiguity detection: which Jev uncertainty signal predicts human entropy?
S2 Stability reconciliation: flip rate vs human entropy (not only vs boundary).
S3 Temperature scaling: one-T fit on a stratified half, evaluate on the other.

These consume already-collected calls — no extra API spend. Primary endpoint
(soft ΔECE, Amendment 9 verdict) is unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from jevbench.metrics import (
    expected_calibration_error,
    jensen_shannon_divergence,
    total_variation_distance,
)

Arm = Literal["choice", "noul"]


def top_uncertainty(probs: ArrayLike) -> np.ndarray:
    """Signal (1): 1 − top probability."""
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        return np.asarray([1.0 - float(np.max(p))], dtype=float)
    return 1.0 - p.max(axis=1)


def cross_primitive_jsd(
    choice_probs: ArrayLike, noul_probs: ArrayLike
) -> np.ndarray:
    """Signal (2): JSD between Choice and normalised three-Noul, same call."""
    return jensen_shannon_divergence(choice_probs, noul_probs)


def repeat_instability(
    probs_by_repeat: list[ArrayLike],
    *,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Signal (3): flip rate + mean pairwise TVD across R repeats.

    ``probs_by_repeat[r]`` is (n_items, K) for repeat r. Items aligned.
    """
    if len(probs_by_repeat) < 2:
        return {"flip_rate": float("nan"), "mean_tvd": float("nan"), "n_repeats": 0}
    mats = [np.asarray(p, dtype=float) for p in probs_by_repeat]
    n = mats[0].shape[0]
    for m in mats:
        if m.shape[0] != n:
            raise ValueError("repeat matrices must share n_items")
    argmax = [m.argmax(axis=1) for m in mats]
    flips = 0
    pairs = 0
    for i in range(n):
        labels_i = {int(a[i]) for a in argmax}
        if len(labels_i) > 1:
            flips += 1
        pairs += 1
    tvds: list[float] = []
    for a in range(len(mats)):
        for b in range(a + 1, len(mats)):
            tvds.extend(total_variation_distance(mats[a], mats[b]).tolist())
    return {
        "flip_rate": flips / max(pairs, 1),
        "mean_tvd": float(np.mean(tvds)) if tvds else float("nan"),
        "n_repeats": len(mats),
        "n_items": n,
    }


def auroc_vs_entropy(signal: ArrayLike, entropy: ArrayLike) -> float:
    """AUROC treating high-entropy (above median) as the positive class."""
    s = np.asarray(signal, dtype=float)
    e = np.asarray(entropy, dtype=float)
    if s.size != e.size or s.size < 2:
        return float("nan")
    y = (e >= np.median(e)).astype(int)
    if y.min() == y.max():
        return float("nan")
    pos = s[y == 1]
    neg = s[y == 0]
    correct = 0.0
    for p in pos:
        correct += float(np.sum(p > neg) + 0.5 * np.sum(p == neg))
    return correct / (len(pos) * len(neg))


def spearman_vs_entropy(signal: ArrayLike, entropy: ArrayLike) -> dict[str, float]:
    s = np.asarray(signal, dtype=float)
    e = np.asarray(entropy, dtype=float)
    if s.size != e.size or s.size < 4:
        return {"rho": float("nan"), "p": float("nan")}
    rho, p = stats.spearmanr(s, e)
    return {"rho": float(rho), "p": float(p)}


def paired_bootstrap_compare(
    signal_a: ArrayLike,
    signal_b: ArrayLike,
    entropy: ArrayLike,
    *,
    metric: Literal["auroc", "spearman"] = "auroc",
    n_boot: int = 2000,
    seed: int = 20260924,
) -> dict[str, Any]:
    """Paired bootstrap on AUROC or Spearman difference (a − b)."""
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    e = np.asarray(entropy, dtype=float)
    n = a.size
    rng = np.random.default_rng(seed)

    def _score(sig: np.ndarray, ent: np.ndarray) -> float:
        if metric == "auroc":
            return auroc_vs_entropy(sig, ent)
        return spearman_vs_entropy(sig, ent)["rho"]

    obs = _score(a, e) - _score(b, e)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas.append(_score(a[idx], e[idx]) - _score(b[idx], e[idx]))
    arr = np.asarray(deltas, dtype=float)
    return {
        "metric": metric,
        "delta_obs": float(obs),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "n_boot": n_boot,
    }


def temperature_scale_probs(probs: ArrayLike, temperature: float) -> np.ndarray:
    """Apply T to logits recovered from probabilities (clip for stability)."""
    p = np.asarray(probs, dtype=float)
    p = np.clip(p, 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    log_p = np.log(p)
    scaled = log_p / max(float(temperature), 1e-6)
    scaled -= scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)


def fit_temperature(
    probs: ArrayLike,
    soft_correct: ArrayLike,
    *,
    grid: tuple[float, ...] | None = None,
) -> float:
    """Fit one temperature minimizing soft Brier on the train half."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(soft_correct, dtype=float)
    if grid is None:
        grid = tuple(float(x) for x in np.concatenate([
            np.linspace(0.5, 2.0, 16),
            np.linspace(2.0, 5.0, 8),
        ]))
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        q = temperature_scale_probs(p, t)
        # soft Brier: mean ||q − onehot-ish soft||^2; use top-label soft target
        # Here soft_correct is the annotator share on the model's argmax path
        # for ECE; for T-fit we minimize ECE on the train fold instead.
        ece_res = expected_calibration_error(
            q,
            np.zeros(len(y), dtype=int),  # unused when correctness provided
            n_bins=10,
            strategy="uniform",
            correctness=y,
        )
        ece = float(ece_res.ece)
        if ece < best_loss:
            best_loss = ece
            best_t = float(t)
    return best_t


def temperature_scaling_cv(
    probs: ArrayLike,
    soft_correct: ArrayLike,
    human_dist: ArrayLike,
    strata: ArrayLike,
    *,
    seed: int = 20260924,
) -> dict[str, Any]:
    """Fit T on a stratified random half; evaluate ECE + JSD on the other; swap; average."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(soft_correct, dtype=float)
    h = np.asarray(human_dist, dtype=float)
    s = np.asarray(strata)
    rng = np.random.default_rng(seed)
    n = p.shape[0]
    idx = np.arange(n)

    def _split() -> tuple[np.ndarray, np.ndarray]:
        train, test = [], []
        for lab in np.unique(s):
            members = idx[s == lab]
            rng.shuffle(members)
            mid = len(members) // 2
            train.extend(members[:mid].tolist())
            test.extend(members[mid:].tolist())
        return np.asarray(train), np.asarray(test)

    folds: list[dict[str, float]] = []
    for _ in range(2):
        tr, te = _split()
        t = fit_temperature(p[tr], y[tr])
        q = temperature_scale_probs(p[te], t)
        ece_res = expected_calibration_error(
            q,
            np.zeros(len(te), dtype=int),
            n_bins=10,
            strategy="uniform",
            correctness=y[te],
        )
        ece = float(ece_res.ece)
        jsd = float(np.mean(jensen_shannon_divergence(q, h[te])))
        folds.append({"temperature": t, "ece": float(ece), "jsd_mean": jsd})

    raw = expected_calibration_error(
        p,
        np.zeros(n, dtype=int),
        n_bins=10,
        strategy="uniform",
        correctness=y,
    )
    return {
        "folds": folds,
        "mean_temperature": float(np.mean([f["temperature"] for f in folds])),
        "mean_ece": float(np.mean([f["ece"] for f in folds])),
        "mean_jsd": float(np.mean([f["jsd_mean"] for f in folds])),
        "raw_ece": float(raw.ece),
    }


def ambiguity_detection_table(
    *,
    entropy: ArrayLike,
    top_unc: ArrayLike,
    cross_jsd: ArrayLike,
    instability: ArrayLike,
) -> dict[str, Any]:
    """S1 summary: AUROC + Spearman for each signal vs human entropy."""
    signals = {
        "one_minus_top": np.asarray(top_unc, dtype=float),
        "cross_primitive_jsd": np.asarray(cross_jsd, dtype=float),
        "repeat_instability": np.asarray(instability, dtype=float),
    }
    e = np.asarray(entropy, dtype=float)
    rows = {}
    for name, sig in signals.items():
        rows[name] = {
            "auroc": auroc_vs_entropy(sig, e),
            **spearman_vs_entropy(sig, e),
        }
    comparisons = {}
    names = list(signals)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            comparisons[f"{a}_minus_{b}"] = paired_bootstrap_compare(
                signals[a], signals[b], e, metric="auroc"
            )
    return {
        "schema": "jevbench.secondary.s1_ambiguity.v1",
        "signals": rows,
        "paired_auroc_deltas": comparisons,
        "exploratory": True,
        "primary_unchanged": True,
    }
