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
    soft_correctness,
)

Arm = Literal["choice", "noul"]

S1_SEED = 20260924
S1_N_BOOT = 10_000


def top_uncertainty(probs: ArrayLike) -> np.ndarray:
    """Signal (1): 1 − top probability."""
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        return np.asarray([1.0 - float(np.max(p))], dtype=float)
    return 1.0 - p.max(axis=1)


def choice_margin(probs: ArrayLike) -> np.ndarray:
    """top1 − top2 Choice probability (pass 0)."""
    p = np.asarray(probs, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def cross_primitive_jsd(
    choice_probs: ArrayLike, noul_probs: ArrayLike
) -> np.ndarray:
    """Signal (2): JSD between Choice and normalised three-Noul, same call."""
    return jensen_shannon_divergence(choice_probs, noul_probs)


def per_item_repeat_signals(
    probs_by_repeat: list[ArrayLike],
) -> dict[str, np.ndarray]:
    """Per-item (3a) pairwise flip rate and (3b) mean pairwise TVD across R."""
    if len(probs_by_repeat) < 2:
        n = int(np.asarray(probs_by_repeat[0]).shape[0]) if probs_by_repeat else 0
        nan = np.full(n, np.nan)
        return {"flip_rate": nan, "mean_tvd": nan, "n_repeats": np.zeros(n)}
    mats = [np.asarray(p, dtype=float) for p in probs_by_repeat]
    n = mats[0].shape[0]
    for m in mats:
        if m.shape[0] != n:
            raise ValueError("repeat matrices must share n_items")
    r = len(mats)
    flip = np.zeros(n, dtype=float)
    tvd = np.zeros(n, dtype=float)
    pairs = r * (r - 1) // 2
    for i in range(n):
        args = [int(m[i].argmax()) for m in mats]
        disagree = 0
        tvds: list[float] = []
        for a in range(r):
            for b in range(a + 1, r):
                if args[a] != args[b]:
                    disagree += 1
                tvds.append(float(0.5 * np.sum(np.abs(mats[a][i] - mats[b][i]))))
        flip[i] = disagree / pairs if pairs else 0.0
        tvd[i] = float(np.mean(tvds)) if tvds else float("nan")
    return {
        "flip_rate": flip,
        "mean_tvd": tvd,
        "n_repeats": np.full(n, r, dtype=float),
    }


def repeat_instability(
    probs_by_repeat: list[ArrayLike],
    *,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """Aggregate flip rate + mean pairwise TVD across R repeats (summary)."""
    del labels  # retained for API compat with older callers
    per = per_item_repeat_signals(probs_by_repeat)
    flip = per["flip_rate"]
    tvd = per["mean_tvd"]
    if flip.size == 0:
        return {"flip_rate": float("nan"), "mean_tvd": float("nan"), "n_repeats": 0}
    any_flip = (flip > 0).astype(float)
    return {
        "flip_rate": float(np.mean(any_flip)),
        "mean_tvd": float(np.nanmean(tvd)),
        "n_repeats": int(per["n_repeats"][0]) if flip.size else 0,
        "n_items": int(flip.size),
        "mean_pairwise_flip_rate": float(np.mean(flip)),
    }


def auroc_binary(signal: ArrayLike, positive: ArrayLike) -> float:
    """AUROC for a continuous signal vs binary labels (1 = positive)."""
    s = np.asarray(signal, dtype=float)
    y = np.asarray(positive, dtype=int)
    if s.size != y.size or s.size < 2:
        return float("nan")
    if y.min() == y.max():
        return float("nan")
    # Mann–Whitney / ROC via ranks (O(n log n)); avoids O(n⁺·n⁻) Python loops.
    order = np.argsort(s)
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # Average ranks for ties
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and sorted_s[j] == sorted_s[i]:
            j += 1
        if j > i + 1:
            avg = 0.5 * (i + 1 + j)
            ranks[order[i:j]] = avg
        i = j
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    sum_ranks_pos = float(np.sum(ranks[y == 1]))
    # AUC = (sum_ranks_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

def auroc_vs_entropy(signal: ArrayLike, entropy: ArrayLike) -> float:
    """AUROC treating high-entropy (above median) as the positive class."""
    e = np.asarray(entropy, dtype=float)
    y = (e >= np.median(e)).astype(int)
    return auroc_binary(signal, y)


def spearman_vs_entropy(signal: ArrayLike, entropy: ArrayLike) -> dict[str, float]:
    s = np.asarray(signal, dtype=float)
    e = np.asarray(entropy, dtype=float)
    if s.size != e.size or s.size < 4:
        return {"rho": float("nan"), "p": float("nan")}
    rho, p = stats.spearmanr(s, e)
    return {"rho": float(rho), "p": float(p)}


def bootstrap_metric_ci(
    signal: ArrayLike,
    target: ArrayLike,
    *,
    metric: Literal["auroc_binary", "auroc_entropy", "spearman"] = "spearman",
    n_boot: int = S1_N_BOOT,
    seed: int = S1_SEED,
) -> dict[str, float]:
    """Item-level bootstrap CI for AUROC or Spearman."""
    s = np.asarray(signal, dtype=float)
    t = np.asarray(target, dtype=float)
    n = s.size
    rng = np.random.default_rng(seed)

    def _score(a: np.ndarray, b: np.ndarray) -> float:
        if metric == "auroc_binary":
            return auroc_binary(a, b)
        if metric == "auroc_entropy":
            return auroc_vs_entropy(a, b)
        return spearman_vs_entropy(a, b)["rho"]

    obs = _score(s, t)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[i] = _score(s[idx], t[idx])
    return {
        "point": float(obs),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_boot": n_boot,
        "seed": seed,
    }


def paired_bootstrap_compare(
    signal_a: ArrayLike,
    signal_b: ArrayLike,
    target: ArrayLike,
    *,
    metric: Literal["auroc", "auroc_binary", "spearman"] = "auroc",
    n_boot: int = S1_N_BOOT,
    seed: int = S1_SEED,
) -> dict[str, Any]:
    """Paired bootstrap on AUROC or Spearman difference (a − b)."""
    a = np.asarray(signal_a, dtype=float)
    b = np.asarray(signal_b, dtype=float)
    e = np.asarray(target, dtype=float)
    n = a.size
    rng = np.random.default_rng(seed)

    def _score(sig: np.ndarray, ent: np.ndarray) -> float:
        if metric in ("auroc", "auroc_entropy"):
            return auroc_vs_entropy(sig, ent)
        if metric == "auroc_binary":
            return auroc_binary(sig, ent)
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
        "seed": seed,
    }


def partial_spearman(
    x: ArrayLike, y: ArrayLike, z: ArrayLike
) -> dict[str, float]:
    """Partial Spearman(x, y | z) via rank residualisation."""
    xv = np.asarray(x, dtype=float)
    yv = np.asarray(y, dtype=float)
    zv = np.asarray(z, dtype=float)
    if xv.size != yv.size or xv.size != zv.size or xv.size < 5:
        return {"rho": float("nan"), "p": float("nan")}
    rx = stats.rankdata(xv)
    ry = stats.rankdata(yv)
    rz = stats.rankdata(zv)
    bx = np.polyfit(rz, rx, 1)
    by = np.polyfit(rz, ry, 1)
    rx_res = rx - (bx[0] * rz + bx[1])
    ry_res = ry - (by[0] * rz + by[1])
    rho, p = stats.spearmanr(rx_res, ry_res)
    return {"rho": float(rho), "p": float(p)}


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


def cross_entropy_to_humans(probs: ArrayLike, human_dist: ArrayLike) -> float:
    """Mean cross-entropy of model probs vs human vote distribution."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    h = np.clip(np.asarray(human_dist, dtype=float), 1e-12, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    h = h / h.sum(axis=-1, keepdims=True)
    return float(-np.mean(np.sum(h * np.log(p), axis=-1)))


def fit_temperature(
    probs: ArrayLike,
    human_dist: ArrayLike,
    *,
    grid: tuple[float, ...] | None = None,
    t_lo: float = 0.05,
    t_hi: float = 100.0,
    n_grid: int = 200,
) -> float:
    """Fit one temperature minimizing cross-entropy to the human vote dist."""
    return float(
        fit_temperature_detailed(
            probs,
            human_dist,
            grid=grid,
            t_lo=t_lo,
            t_hi=t_hi,
            n_grid=n_grid,
        )["temperature"]
    )


def _golden_section_minimize(
    fn,
    a: float,
    b: float,
    *,
    tol: float = 1e-5,
    max_iter: int = 80,
) -> tuple[float, float]:
    """1-D golden-section search on [a, b]; returns (argmin, f(argmin))."""
    if not (a < b):
        raise ValueError(f"golden-section requires a < b; got [{a}, {b}]")
    phi = (1.0 + 5.0**0.5) / 2.0
    inv = 1.0 / phi
    c = b - (b - a) * inv
    d = a + (b - a) * inv
    fc = float(fn(c))
    fd = float(fn(d))
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * inv
            fc = float(fn(c))
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * inv
            fd = float(fn(d))
    # Evaluate midpoint of final bracket
    mid = 0.5 * (a + b)
    fmid = float(fn(mid))
    candidates = [(c, fc), (d, fd), (mid, fmid)]
    best_t, best_f = min(candidates, key=lambda x: x[1])
    return float(best_t), float(best_f)


def fit_temperature_detailed(
    probs: ArrayLike,
    human_dist: ArrayLike,
    *,
    grid: tuple[float, ...] | None = None,
    t_lo: float = 0.05,
    t_hi: float = 100.0,
    n_grid: int = 200,
) -> dict[str, Any]:
    """Log-spaced grid then golden-section refine; record bound flags."""
    p = np.asarray(probs, dtype=float)
    h = np.asarray(human_dist, dtype=float)

    def _ce(t: float) -> float:
        return cross_entropy_to_humans(temperature_scale_probs(p, t), h)

    if grid is None:
        grid_arr = np.geomspace(float(t_lo), float(t_hi), int(n_grid))
    else:
        grid_arr = np.asarray(list(grid), dtype=float)
        t_lo = float(grid_arr.min())
        t_hi = float(grid_arr.max())
        n_grid = int(grid_arr.size)

    losses = np.asarray([_ce(float(t)) for t in grid_arr], dtype=float)
    best_i = int(np.argmin(losses))
    ce_grid_min = float(losses[best_i])
    # Bracket around the best grid point for golden-section refine.
    left = float(grid_arr[max(0, best_i - 1)])
    right = float(grid_arr[min(len(grid_arr) - 1, best_i + 1)])
    if left == right:
        # Expand slightly within bounds if grid collapsed (single-point edge).
        left = max(t_lo, left / 1.05)
        right = min(t_hi, right * 1.05)
    t_star, ce_min = _golden_section_minimize(_ce, left, right)
    within = bool(t_star <= t_lo * 1.01 or t_star >= t_hi * 0.99)
    return {
        "temperature": float(t_star),
        "ce_min": float(ce_min),
        "ce_grid_min": ce_grid_min,
        "grid_best_t": float(grid_arr[best_i]),
        "t_lo": float(t_lo),
        "t_hi": float(t_hi),
        "n_grid": int(n_grid),
        "within_1pct_of_bound": within,
        "bound_side": (
            "lo"
            if t_star <= t_lo * 1.01
            else "hi"
            if t_star >= t_hi * 0.99
            else None
        ),
    }


def _stratum_soft_ece(
    probs: np.ndarray, human: np.ndarray, labels: np.ndarray
) -> float:
    correct = soft_correctness(probs, human)
    res = expected_calibration_error(
        probs,
        labels,
        n_bins=10,
        strategy="uniform",
        correctness=correct,
    )
    return float(res.ece)


def temperature_scaling_cv(
    probs: ArrayLike,
    soft_correct: ArrayLike,
    human_dist: ArrayLike,
    strata: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    seed: int = 20260924,
) -> dict[str, Any]:
    """Fit T on stratified half A (CE to humans); evaluate on B; swap; average.

    ``soft_correct`` is retained for API compatibility with older smoke tests
    but S3 fits T by cross-entropy to ``human_dist`` (Amendment 10.3 / Step 15 E).
    """
    del soft_correct
    p = np.asarray(probs, dtype=float)
    h = np.asarray(human_dist, dtype=float)
    s = np.asarray(strata)
    if labels is None:
        y_lab = np.zeros(p.shape[0], dtype=int)
    else:
        y_lab = np.asarray(labels, dtype=int)
    rng = np.random.default_rng(seed)
    n = p.shape[0]
    idx = np.arange(n)

    def _split() -> tuple[np.ndarray, np.ndarray]:
        train, test = [], []
        for lab in np.unique(s):
            members = idx[s == lab].copy()
            rng.shuffle(members)
            mid = len(members) // 2
            train.extend(members[:mid].tolist())
            test.extend(members[mid:].tolist())
        return np.asarray(train), np.asarray(test)

    def _eval(q_full: np.ndarray, te: np.ndarray) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, lab in (("hard", "hard"), ("easy", "easy")):
            mask = s[te] == lab
            if not np.any(mask):
                out[f"soft_ece_{name}"] = float("nan")
                out[f"jsd_{name}"] = float("nan")
                continue
            sub = te[mask]
            out[f"soft_ece_{name}"] = _stratum_soft_ece(
                q_full[sub], h[sub], y_lab[sub]
            )
            out[f"jsd_{name}"] = float(
                np.mean(jensen_shannon_divergence(q_full[sub], h[sub]))
            )
        out["delta_ece_raw"] = float(
            out.get("soft_ece_hard", float("nan"))
            - out.get("soft_ece_easy", float("nan"))
        )
        return out

    tr0, te0 = _split()
    folds: list[dict[str, Any]] = []
    for fold_i, (tr, te) in enumerate(((tr0, te0), (te0, tr0))):
        fit = fit_temperature_detailed(p[tr], h[tr])
        t = float(fit["temperature"])
        q_scaled = temperature_scale_probs(p, t)
        folds.append(
            {
                "fold": fold_i,
                "temperature": t,
                "within_1pct_of_bound": bool(fit["within_1pct_of_bound"]),
                "bound_side": fit["bound_side"],
                "ce_min": float(fit["ce_min"]),
                "t_lo": float(fit["t_lo"]),
                "t_hi": float(fit["t_hi"]),
                "n_grid": int(fit["n_grid"]),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "before": _eval(p, te),
                "after": _eval(q_scaled, te),
            }
        )

    def _mean_key(which: str, key: str) -> float:
        return float(np.nanmean([float(f[which][key]) for f in folds]))

    return {
        "schema": "jevbench.secondary.s3_temperature.v1",
        "prereg_clause": "Amendment 10.3 S3 Temperature scaling",
        "fit_objective": "cross_entropy_to_human_vote_distribution",
        "fit_grid": {
            "t_lo": 0.05,
            "t_hi": 100.0,
            "n_grid": 200,
            "refine": "golden_section",
            "note": (
                "Post-data fix: prior grid capped at 5.0 and Choice landed on "
                "the cap; widened to geomspace(0.05, 100, 200) + golden-section."
            ),
        },
        "folds": folds,
        "mean_temperature": float(np.mean([f["temperature"] for f in folds])),
        "any_fold_at_bound": any(bool(f["within_1pct_of_bound"]) for f in folds),
        "before_mean": {
            "soft_ece_hard": _mean_key("before", "soft_ece_hard"),
            "soft_ece_easy": _mean_key("before", "soft_ece_easy"),
            "delta_ece_raw": _mean_key("before", "delta_ece_raw"),
            "jsd_hard": _mean_key("before", "jsd_hard"),
            "jsd_easy": _mean_key("before", "jsd_easy"),
        },
        "after_mean": {
            "soft_ece_hard": _mean_key("after", "soft_ece_hard"),
            "soft_ece_easy": _mean_key("after", "soft_ece_easy"),
            "delta_ece_raw": _mean_key("after", "delta_ece_raw"),
            "jsd_hard": _mean_key("after", "jsd_hard"),
            "jsd_easy": _mean_key("after", "jsd_easy"),
        },
        "seed": seed,
    }


def ambiguity_detection_table(
    *,
    entropy: ArrayLike,
    hard_indicator: ArrayLike,
    top_unc: ArrayLike,
    cross_jsd: ArrayLike,
    flip_rate: ArrayLike,
    mean_tvd: ArrayLike,
    n_boot: int = S1_N_BOOT,
    seed: int = S1_SEED,
) -> dict[str, Any]:
    """S1 summary: AUROC (hard-vs-easy) + Spearman vs entropy; bootstrap CIs."""
    signals = {
        "one_minus_top": np.asarray(top_unc, dtype=float),
        "cross_primitive_jsd": np.asarray(cross_jsd, dtype=float),
        "flip_rate": np.asarray(flip_rate, dtype=float),
        "mean_pairwise_tvd": np.asarray(mean_tvd, dtype=float),
    }
    e = np.asarray(entropy, dtype=float)
    hard = np.asarray(hard_indicator, dtype=int)
    rows: dict[str, Any] = {}
    for name, sig in signals.items():
        rows[name] = {
            "auroc_hard_vs_easy": auroc_binary(sig, hard),
            "auroc_hard_vs_easy_bootstrap": bootstrap_metric_ci(
                sig, hard, metric="auroc_binary", n_boot=n_boot, seed=seed
            ),
            "spearman_vs_entropy": spearman_vs_entropy(sig, e),
            "spearman_vs_entropy_bootstrap": bootstrap_metric_ci(
                sig, e, metric="spearman", n_boot=n_boot, seed=seed + 1
            ),
        }
    comparisons: dict[str, Any] = {}
    names = list(signals)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            comparisons[f"{a}_minus_{b}"] = {
                "auroc_hard_vs_easy": paired_bootstrap_compare(
                    signals[a],
                    signals[b],
                    hard,
                    metric="auroc_binary",
                    n_boot=n_boot,
                    seed=seed + 2,
                ),
                "spearman_vs_entropy": paired_bootstrap_compare(
                    signals[a],
                    signals[b],
                    e,
                    metric="spearman",
                    n_boot=n_boot,
                    seed=seed + 3,
                ),
            }
    return {
        "schema": "jevbench.secondary.s1_ambiguity.v1",
        "prereg_clause": "Amendment 10.3 S1 Ambiguity detection",
        "signals": rows,
        "paired_deltas": comparisons,
        "n_boot": n_boot,
        "seed": seed,
        "exploratory": True,
        "primary_unchanged": True,
    }


def stability_reconciliation_table(
    *,
    flip_rate: ArrayLike,
    entropy: ArrayLike,
    margin: ArrayLike,
    n_boot: int = S1_N_BOOT,
    seed: int = S1_SEED,
) -> dict[str, Any]:
    """S2: Spearman(flip, entropy), Spearman(flip, margin), partial | margin."""
    flip = np.asarray(flip_rate, dtype=float)
    ent = np.asarray(entropy, dtype=float)
    mar = np.asarray(margin, dtype=float)
    return {
        "schema": "jevbench.secondary.s2_stability.v1",
        "prereg_clause": "Amendment 10.3 S2 Stability reconciliation",
        "spearman_flip_entropy": spearman_vs_entropy(flip, ent),
        "spearman_flip_entropy_bootstrap": bootstrap_metric_ci(
            flip, ent, metric="spearman", n_boot=n_boot, seed=seed
        ),
        "spearman_flip_margin": spearman_vs_entropy(flip, mar),
        "spearman_flip_margin_bootstrap": bootstrap_metric_ci(
            flip, mar, metric="spearman", n_boot=n_boot, seed=seed + 1
        ),
        "partial_spearman_flip_entropy_given_margin": partial_spearman(
            flip, ent, mar
        ),
        "n_boot": n_boot,
        "seed": seed,
        "exploratory": True,
        "primary_unchanged": True,
    }
