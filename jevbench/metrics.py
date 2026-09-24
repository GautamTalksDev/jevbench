"""Calibration, discrimination, selective prediction, and inference utilities.

Pure functions over arrays. No I/O.

=============================================================================
CONFIDENCE VS PROBABILITY — READ THIS
=============================================================================
Jev returns BOTH ``probabilities`` (a distribution over options / a noul in
[0, 1]) and ``confidence`` (a single number derived from the *shape* of that
distribution — flatter ⇒ lower confidence).

``confidence`` is a SHARPNESS statistic, not a correctness estimate. A model
can be extremely sharp and extremely wrong.

Calibration metrics in this module therefore run on ``probabilities`` /
``noul`` ONLY. Passing a confidence array into a calibration function raises
``TypeError``. Evaluate confidence separately as a routing / selective signal
(see ``selective_from_confidence``).
=============================================================================

=============================================================================
TOP-LABEL ECE (pinned) + ΔECE BIAS TRAP
=============================================================================
Primary ECE is TOP-LABEL: conf = max predicted-class probability; correct =
1[prediction == label]. Binary Noul uses conf = max(p, 1-p), not p itself.
Bin confidence is the MEAN of conf_i in the bin — never the bin midpoint.
Empty bins contribute zero.

ECE is a biased estimator; bias grows as n shrinks. Comparing strata with
unequal n invents positive ΔECE from sample size alone. ``delta_ece``
requires equal n (or explicit ``allow_unequal_n=True``). Use
``subsample_to_equal_n`` and report ``null_delta_ece_band``.

The four-tier ECE-vs-accuracy slope is DESCRIPTIVE only (2 residual df).
The primary EXP-1 endpoint is ``delta_ece`` (stratified two-sample bootstrap
on disjoint hard vs easy strata — never paired).
=============================================================================

Calibration mathematics: thin wrappers around netcal (when installed) plus
occupancy reporting. Discrimination / selective extras use scikit-learn /
scipy. We do not hand-roll ECE for the paper's primary scalar — we validate
wrappers against netcal in ``tests/test_agreement.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn import metrics as skm
from sklearn.metrics import brier_score_loss

BinStrategy = Literal["uniform", "quantile"]

_CONFIDENCE_REJECT_MSG = (
    "Calibration metrics must run on probabilities / noul, not on Jev "
    "`confidence`. `confidence` is a SHARPNESS statistic derived from the "
    "shape of the probability distribution (flat ⇒ low confidence) — it is "
    "NOT a correctness estimate. A model can be sharp and wrong. Pass the "
    "probability / noul array instead, and evaluate confidence separately "
    "as a routing signal (see selective_from_confidence)."
)


# ---------------------------------------------------------------------------
# Confidence guard
# ---------------------------------------------------------------------------


class ConfidenceArray(np.ndarray):
    """ndarray subclass marking sharpness / confidence scores.

    Calibration entry points reject this type. Construct via
    ``as_confidence(...)``.
    """


def as_confidence(x: ArrayLike) -> ConfidenceArray:
    """Mark an array as Jev-style confidence (sharpness), not probability."""
    arr = np.asarray(x, dtype=float)
    out = arr.view(ConfidenceArray)
    return out


def _reject_confidence(probs: Any, *, arg_name: str = "probs") -> np.ndarray:
    if isinstance(probs, ConfidenceArray):
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    # Reject sneaky kwargs pattern handled at call sites; also reject object
    # arrays tagged with the attribute.
    if getattr(probs, "_jevbench_is_confidence", False):
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    return np.asarray(probs, dtype=float)


def _as_labels(labels: ArrayLike) -> np.ndarray:
    return np.asarray(labels)


def _for_ece(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-label confidence and correctness for ECE / reliability.

    PINNED DEFINITION (do not substitute a variant):

    - Multiclass Choice (2-D probs):
        conf_i = max_k p_ik
        correct_i = 1[argmax_k p_ik == y_i]

    - Binary Noul (1-D p_i = P(positive)):
        pred_i = 1[p_i >= 0.5]
        conf_i = max(p_i, 1 - p_i)   # confidence OF THE PREDICTED CLASS
        correct_i = 1[pred_i == y_i]

    This is NOT classic binary reliability of P(y=1) against the label.
    Using p_i itself as the confidence score is a different, wrong estimator
    for top-label ECE. See ``paper/appendix/validation.md``.
    """
    labels = np.asarray(labels)
    if probs.ndim == 1:
        y = labels.astype(int)
        if not set(np.unique(y.astype(float))).issubset({0.0, 1.0}):
            raise ValueError(
                "1-D probs require binary labels in {0,1} (or bool). "
                "For multiclass, pass a 2-D probability matrix."
            )
        pred = (probs >= 0.5).astype(int)
        conf = np.maximum(probs, 1.0 - probs)
        return conf, (pred == y).astype(float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 1-D or 2-D, got shape {probs.shape}")
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    if labels.ndim == 2:
        y_true = labels.argmax(axis=1)
    else:
        y_true = labels
        if y_true.dtype.kind in "UO":
            raise ValueError(
                "multiclass labels must be integer class indices matching "
                "probability column order"
            )
        y_true = y_true.astype(int)
    return conf, (pred == y_true).astype(float)


def soft_correctness(probs: ArrayLike, label_dist: ArrayLike) -> np.ndarray:
    """Primary correctness: share of annotators who chose the model's argmax.

    ``correct_i = label_dist[i, argmax_k probs[i]]``, with ``label_dist``
    rows summing to 1 in the same column order as ``probs``. ECE bin
    accuracy is the mean of these values in the bin (pass them as
    ``correctness`` to :func:`expected_calibration_error`), not a 0/1 hit rate.
    """
    p = np.asarray(probs, dtype=float)
    dist = np.asarray(label_dist, dtype=float)
    if p.ndim != 2:
        raise ValueError(f"soft correctness needs a 2-D probability matrix, got {p.shape}")
    if dist.shape != p.shape:
        raise ValueError(
            f"label_dist shape {dist.shape} must match probs shape {p.shape}"
        )
    pred = p.argmax(axis=1)
    return dist[np.arange(len(pred)), pred].astype(float)


def jensen_shannon_divergence(p: ArrayLike, q: ArrayLike, *, axis: int = -1) -> np.ndarray:
    """Jensen–Shannon divergence (base-2 bits) between discrete distributions.

    Secondary EXP-1 metric (Amendment 9): divergence from Jev's distribution
    to the human annotator distribution. Uses the average of KL(p‖m) and
    KL(q‖m) with m = (p+q)/2.
    """
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    q_arr = np.clip(np.asarray(q, dtype=float), 1e-12, 1.0)
    p_arr = p_arr / p_arr.sum(axis=axis, keepdims=True)
    q_arr = q_arr / q_arr.sum(axis=axis, keepdims=True)
    m = 0.5 * (p_arr + q_arr)
    kl_pm = np.sum(p_arr * (np.log2(p_arr) - np.log2(m)), axis=axis)
    kl_qm = np.sum(q_arr * (np.log2(q_arr) - np.log2(m)), axis=axis)
    return (0.5 * (kl_pm + kl_qm)).astype(float)


def total_variation_distance(p: ArrayLike, q: ArrayLike, *, axis: int = -1) -> np.ndarray:
    """Total variation distance ½‖p−q‖₁ between discrete distributions."""
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    p_arr = p_arr / np.clip(p_arr.sum(axis=axis, keepdims=True), 1e-12, None)
    q_arr = q_arr / np.clip(q_arr.sum(axis=axis, keepdims=True), 1e-12, None)
    return (0.5 * np.sum(np.abs(p_arr - q_arr), axis=axis)).astype(float)


def uniform_baseline_divergence(
    label_dist: ArrayLike, *, n_classes: int | None = None
) -> dict[str, float]:
    """JSD/TVD of a uniform guess against the human distribution (baseline)."""
    dist = np.asarray(label_dist, dtype=float)
    if dist.ndim == 1:
        dist = dist[None, :]
    k = int(n_classes or dist.shape[-1])
    uni = np.full_like(dist, 1.0 / k)
    return {
        "jsd_mean": float(jensen_shannon_divergence(uni, dist).mean()),
        "tvd_mean": float(total_variation_distance(uni, dist).mean()),
        "n": int(dist.shape[0]),
        "n_classes": k,
    }


def hard_correctness(probs: ArrayLike, majority: ArrayLike) -> np.ndarray:
    """Hard correctness: ``1[argmax == majority label]``.

    Hard scoring puts label noise in the hard stratum only, which inflates ΔECE in the direction of the hypothesis.

    The hard stratum is the high-entropy tail of the 100-annotator
    distribution, where the majority is a thin mode over a dispersed vote.
    Scoring a miss against that mode charges every minority vote to the
    model, and those items sit only in the hard stratum. The easy stratum
    is nearly unanimous, so the same rule barely moves it. Accuracy falls
    and top-label ECE rises on the hard side alone, so
    ΔECE = ECE(hard) − ECE(easy) is pushed upward — the sign H1 reads as
    "hard items are less calibrated." Soft scoring
    (:func:`soft_correctness`) is the primary endpoint because it credits
    the share of annotators who chose the argmax.
    """
    p = np.asarray(probs, dtype=float)
    maj = np.asarray(majority)
    if p.ndim != 2:
        raise ValueError(f"hard correctness needs a 2-D probability matrix, got {p.shape}")
    if len(maj) != len(p):
        raise ValueError("majority length must match probs")
    if maj.dtype.kind in "UO":
        raise ValueError("majority must be integer class indices, not strings")
    return (p.argmax(axis=1) == maj.astype(int)).astype(float)


def _for_decisions(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain-score and correctness for selective / automation curves.

    Score is the probability assigned to the *predicted* class (sharpness of
    the decision). Correctness is 1{prediction == label}.
    """
    labels = np.asarray(labels)
    if probs.ndim == 1:
        y = labels.astype(int)
        pred = (probs >= 0.5).astype(int)
        score = np.where(pred == 1, probs, 1.0 - probs)
        return score, (pred == y).astype(float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 1-D or 2-D, got shape {probs.shape}")
    pred = probs.argmax(axis=1)
    score = probs.max(axis=1)
    if labels.ndim == 2:
        y_true = labels.argmax(axis=1)
    else:
        y_true = labels.astype(int)
    return score, (pred == y_true).astype(float)


# Back-compat name used in earlier drafts
def _binary_scores_and_labels(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return _for_ece(probs, labels)


# ---------------------------------------------------------------------------
# Result types — ECE never returns a bare float
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinBreakdown:
    """One calibration bin with occupancy (required, not optional)."""

    index: int
    lo: float
    hi: float
    count: int
    fraction: float
    mean_confidence: float
    accuracy: float
    abs_gap: float


@dataclass(frozen=True)
class ECEResult:
    """Expected calibration error WITH per-bin occupancy.

    Reporting ECE without occupancy is the methodological failure this
    project criticises — a single crowded bin can determine the whole figure.

    Definition: top-label ECE
        ECE = sum_b (n_b / N) * |acc(b) - conf_bar(b)|
    where conf_bar(b) is the MEAN of conf_i in the bin (not the bin midpoint).
    Empty bins contribute zero (excluded from the sum, not |0 - midpoint|).
    """

    ece: float
    mce: float
    n_bins: int
    strategy: BinStrategy
    n: int
    bins: tuple[BinBreakdown, ...]
    # Parallel arrays for charting
    bin_edges: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_mean_confidence: tuple[float, ...]
    bin_accuracy: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def max_bin_fraction(self) -> float:
        if not self.bin_counts or self.n == 0:
            return 0.0
        return max(self.bin_counts) / self.n


@dataclass(frozen=True)
class BootstrapIntervalPair:
    """Percentile and BCa intervals from the same bootstrap replicates."""

    percentile_low: float
    percentile_high: float
    bca_low: float
    bca_high: float
    z0: float
    acceleration: float
    primary: Literal["percentile", "bca"] = "percentile"

    @property
    def primary_low(self) -> float:
        return self.bca_low if self.primary == "bca" else self.percentile_low

    @property
    def primary_high(self) -> float:
        return self.bca_high if self.primary == "bca" else self.percentile_high

    def to_dict(self) -> dict[str, Any]:
        return {
            "percentile": [self.percentile_low, self.percentile_high],
            "bca": [self.bca_low, self.bca_high],
            "z0": self.z0,
            "acceleration": self.acceleration,
            "primary": self.primary,
        }


@dataclass(frozen=True)
class DeltaECEResult:
    """Primary EXP-1 endpoint: ΔECE = ECE(hard) − ECE(easy).

    Interval is a STRATIFIED TWO-SAMPLE bootstrap (disjoint strata), never
    paired. ``equal_n`` records whether the bias-control guard was satisfied.
    Both percentile and BCa intervals are always computed; ``ci_low`` /
    ``ci_high`` mirror the primary method (default percentile until coverage
    evidence prefers BCa).
    """

    point: float
    ci_low: float
    ci_high: float
    n_hard: int
    n_easy: int
    ece_hard: ECEResult
    ece_easy: ECEResult
    n_bins: int
    strategy: BinStrategy
    seed: int
    n_boot: int
    equal_n: bool
    alpha: float = 0.05
    ci_method: Literal["percentile", "bca"] = "percentile"
    ci_percentile_low: float = float("nan")
    ci_percentile_high: float = float("nan")
    ci_bca_low: float = float("nan")
    ci_bca_high: float = float("nan")
    bca_z0: float = float("nan")
    bca_acceleration: float = float("nan")

    def crosses_zero(self) -> bool:
        return self.ci_low < 0.0 < self.ci_high

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "ci_method": self.ci_method,
            "ci_percentile": [self.ci_percentile_low, self.ci_percentile_high],
            "ci_bca": [self.ci_bca_low, self.ci_bca_high],
            "bca_z0": self.bca_z0,
            "bca_acceleration": self.bca_acceleration,
            "n_hard": self.n_hard,
            "n_easy": self.n_easy,
            "ece_hard": self.ece_hard.to_dict(),
            "ece_easy": self.ece_easy.to_dict(),
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "seed": self.seed,
            "n_boot": self.n_boot,
            "equal_n": self.equal_n,
            "alpha": self.alpha,
            "crosses_zero": self.crosses_zero(),
            "interpretation": (
                "inconclusive (interval crosses zero)"
                if self.crosses_zero()
                else "interval excludes zero"
            ),
        }


@dataclass(frozen=True)
class EqualNSubsample:
    """Balanced strata after discarding surplus items from the larger set."""

    probs_a: np.ndarray
    labels_a: np.ndarray
    item_ids_a: tuple[Any, ...]
    probs_b: np.ndarray
    labels_b: np.ndarray
    item_ids_b: tuple[Any, ...]
    discarded_item_ids: tuple[Any, ...]
    target_n: int
    seed: int


@dataclass(frozen=True)
class NullDeltaECEBand:
    """What ΔECE a perfectly calibrated model produces at fixed n and bins.

    Any observed ΔECE must clear this floor before it means anything.
    """

    n_per_stratum: int
    n_bins: int
    strategy: BinStrategy
    n_sims: int
    seed: int
    mean: float
    std: float
    p025: float
    p975: float
    samples: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["samples"] = list(self.samples)
        return d


@dataclass(frozen=True)
class ReliabilityCurve:
    bin_mean_confidence: tuple[float, ...]
    bin_accuracy: tuple[float, ...]
    bin_counts: tuple[int, ...]
    bin_edges: tuple[float, ...]
    strategy: BinStrategy
    n_bins: int


@dataclass(frozen=True)
class TierCalibration:
    tier: str
    n: int
    accuracy: float
    ece_uniform: ECEResult
    ece_quantile: ECEResult
    brier: float


@dataclass(frozen=True)
class CalibrationByTier:
    """Flagship output: ECE-vs-accuracy shape across difficulty tiers."""

    tiers: tuple[TierCalibration, ...]
    # Ordered for plotting
    accuracies: tuple[float, ...]
    ece_uniform: tuple[float, ...]
    ece_quantile: tuple[float, ...]
    tier_names: tuple[str, ...]
    slope_uniform: float  # ECE ~ accuracy ordinary least squares slope
    slope_quantile: float
    note: str = (
        "If ECE rises as accuracy falls, calibration tracks accuracy. "
        "If the curve is flat (slope near 0), calibration is independent. "
        "Always inspect per-tier occupancy before interpreting ECE."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    low: float
    high: float
    n_resamples: int
    alpha: float = 0.05

    def crosses_zero(self) -> bool:
        return self.low < 0.0 < self.high

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["crosses_zero"] = self.crosses_zero()
        d["interpretation"] = (
            "inconclusive (interval crosses zero)"
            if self.crosses_zero()
            else "interval excludes zero"
        )
        return d


@dataclass(frozen=True)
class McNemarResult:
    statistic: float
    pvalue: float
    n_a_only: int  # A correct, B wrong
    n_b_only: int  # B correct, A wrong
    # Continuity-corrected chi-square; interval via bootstrap of difference
    accuracy_delta: BootstrapCI

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistic": self.statistic,
            "pvalue": self.pvalue,
            "n_a_only": self.n_a_only,
            "n_b_only": self.n_b_only,
            "accuracy_delta": self.accuracy_delta.to_dict(),
        }


@dataclass(frozen=True)
class AutomationResult:
    target_accuracy: float
    threshold: float
    coverage: float  # fraction automated
    accuracy_on_automated: float
    n_automated: int
    n_total: int
    feasible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskCoverageCurve:
    coverage: tuple[float, ...]
    risk: tuple[float, ...]  # 1 - accuracy on retained set
    thresholds: tuple[float, ...]
    aurc: float


@dataclass(frozen=True)
class NegationSumResult:
    """Distribution of p(x) + p(¬x); ideal is exactly 1.0."""

    mean: float
    std: float
    median: float
    mean_abs_deviation: float
    frac_outside_eps: float
    eps: float
    values: tuple[float, ...]  # full sample for plotting
    example_note: str = (
        "TypeSafe's jaggedness docs publish an example summing to 1.19. "
        "Do not assume the invariant holds — measure violation."
    )


# ---------------------------------------------------------------------------
# Binning + ECE (occupancy always returned)
# ---------------------------------------------------------------------------


def _bin_edges(scores: np.ndarray, n_bins: int, strategy: BinStrategy) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(scores, quantiles)
        # Ensure strictly increasing edges for digitize
        edges = np.maximum.accumulate(edges)
        # Unique edges can collapse bins when scores are saturated — keep
        # declared n_bins slots by nudging duplicates slightly.
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = min(1.0, edges[i - 1] + 1e-12)
        edges[0] = 0.0
        edges[-1] = 1.0
        return edges
    raise ValueError(f"unknown strategy {strategy!r}; use 'uniform' or 'quantile'")


def _bin_index(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # rightmost edge inclusive
    idx = np.digitize(scores, edges[1:-1], right=False)
    return np.clip(idx, 0, len(edges) - 2)


def _ece_from_bins(
    scores: np.ndarray,
    correct: np.ndarray,
    n_bins: int,
    strategy: BinStrategy,
) -> ECEResult:
    n = len(scores)
    if n == 0:
        empty = ECEResult(
            ece=float("nan"),
            mce=float("nan"),
            n_bins=n_bins,
            strategy=strategy,
            n=0,
            bins=(),
            bin_edges=tuple(float(x) for x in np.linspace(0, 1, n_bins + 1)),
            bin_counts=tuple(0 for _ in range(n_bins)),
            bin_mean_confidence=tuple(float("nan") for _ in range(n_bins)),
            bin_accuracy=tuple(float("nan") for _ in range(n_bins)),
        )
        return empty

    edges = _bin_edges(scores, n_bins, strategy)
    idx = _bin_index(scores, edges)
    bins: list[BinBreakdown] = []
    counts: list[int] = []
    means: list[float] = []
    accs: list[float] = []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        counts.append(count)
        lo, hi = float(edges[b]), float(edges[b + 1])
        if count == 0:
            means.append(float("nan"))
            accs.append(float("nan"))
            bins.append(
                BinBreakdown(
                    index=b,
                    lo=lo,
                    hi=hi,
                    count=0,
                    fraction=0.0,
                    mean_confidence=float("nan"),
                    accuracy=float("nan"),
                    abs_gap=float("nan"),
                )
            )
            continue
        mean_c = float(scores[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(mean_c - acc)
        means.append(mean_c)
        accs.append(acc)
        frac = count / n
        ece += frac * gap
        mce = max(mce, gap)
        bins.append(
            BinBreakdown(
                index=b,
                lo=lo,
                hi=hi,
                count=count,
                fraction=frac,
                mean_confidence=mean_c,
                accuracy=acc,
                abs_gap=gap,
            )
        )
    return ECEResult(
        ece=float(ece),
        mce=float(mce),
        n_bins=n_bins,
        strategy=strategy,
        n=n,
        bins=tuple(bins),
        bin_edges=tuple(float(x) for x in edges),
        bin_counts=tuple(counts),
        bin_mean_confidence=tuple(means),
        bin_accuracy=tuple(accs),
    )


def expected_calibration_error(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    *,
    confidence: Any = None,
    correctness: ArrayLike | None = None,
) -> ECEResult:
    """Top-label ECE with per-bin occupancy. Never returns a bare float.

    ECE = sum_b (n_b / N) * |acc(b) - mean(conf in b)|

    Bin confidence is the MEAN of conf_i in the bin, not the bin midpoint.
    Empty bins contribute zero. Bin accuracy is the mean of per-item
    correctness in the bin. By default that correctness is the hard
    indicator from :func:`_for_ece`. Pass ``correctness`` in ``[0, 1]``
    (see :func:`soft_correctness`) to make bin accuracy the mean soft
    correctness instead.

    Report BOTH ``uniform`` and ``quantile`` strategies in the paper
    (see :func:`ece_both_strategies`) so binning is not cherry-picked.

    Parameters
    ----------
    probs :
        Probabilities / noul — NOT Jev ``confidence``.
    labels :
        Binary {0,1} for 1-D probs, or class indices for 2-D probs.
        Still required when ``correctness`` is passed (confidence is
        derived from ``probs``; ``labels`` keep the call aligned).
    n_bins, strategy :
        ``uniform`` equal-width bins on [0, 1]; ``quantile`` equal-mass bins.
        Default M=10.
    confidence :
        If provided (even as kwarg), raises ``TypeError``.
    correctness :
        Optional per-item correctness in ``[0, 1]``. When set, bin accuracy
        is its mean. Used for ChaosNLI soft scoring.
    """
    if confidence is not None:
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    p = _reject_confidence(probs)
    y = _as_labels(labels)
    scores, correct = _for_ece(p, y)
    if correctness is not None:
        correct = np.asarray(correctness, dtype=float)
        if correct.shape != scores.shape:
            raise ValueError(
                f"correctness shape {correct.shape} != confidence shape {scores.shape}"
            )
        if np.any(correct < -1e-8) or np.any(correct > 1.0 + 1e-8):
            raise ValueError("correctness values must lie in [0, 1]")
        correct = np.clip(correct, 0.0, 1.0)
    return _ece_from_bins(scores, correct, n_bins=n_bins, strategy=strategy)


# Alias required by the refined build spec
def ece_with_occupancy(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    *,
    confidence: Any = None,
) -> ECEResult:
    """Alias for :func:`expected_calibration_error` — occupancy is mandatory."""
    return expected_calibration_error(
        probs, labels, n_bins=n_bins, strategy=strategy, confidence=confidence
    )


def ece_both_strategies(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
    *,
    confidence: Any = None,
) -> tuple[ECEResult, ECEResult]:
    """Return (uniform ECE, quantile ECE) on identical inputs.

    A paper must show the number was not cherry-picked by binning strategy.
    """
    return (
        expected_calibration_error(
            probs, labels, n_bins=n_bins, strategy="uniform", confidence=confidence
        ),
        expected_calibration_error(
            probs, labels, n_bins=n_bins, strategy="quantile", confidence=confidence
        ),
    )


_UNEQUAL_N_MSG = (
    "ΔECE across strata with unequal n is biased: ECE is a biased estimator "
    "whose bias grows as sample size shrinks. With a fixed bin count, finite "
    "samples produce nonzero ECE even under perfect calibration (absolute "
    "scatter cannot cancel). A smaller hard stratum therefore yields positive "
    "ΔECE from sample size alone — exactly the signal this study seeks, for "
    "exactly the wrong reason. Subsample the larger stratum to match the "
    "smaller via subsample_to_equal_n(...), record discarded item_ids, then "
    "call delta_ece. Override only with allow_unequal_n=True (sets "
    "equal_n=False on the result so the report cannot hide it)."
)


def _bca_endpoints(
    point: float,
    samples: np.ndarray,
    jackknife: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """BCa endpoints from bootstrap replicates + jackknife leave-ones.

    Returns (low, high, z0, acceleration). Uses the standard Efron formula
    with Φ^{-1} / Φ from scipy. Degenerate jackknife (zero variance) falls
    back to acceleration=0 (bias-corrected but not accelerated).
    """
    from scipy.stats import norm

    samples = np.asarray(samples, dtype=float)
    jackknife = np.asarray(jackknife, dtype=float)
    n_boot = len(samples)
    if n_boot < 1:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Bias-correction: fraction of bootstrap replicates below the point
    prop = float(np.mean(samples < point))
    # Clamp away from {0,1} so Φ^{-1} is finite
    prop = min(max(prop, 1.0 / (n_boot + 1)), n_boot / (n_boot + 1.0))
    z0 = float(norm.ppf(prop))

    # Acceleration from jackknife
    theta_dot = float(jackknife.mean()) if len(jackknife) else point
    diffs = theta_dot - jackknife
    sum_d2 = float(np.sum(diffs**2))
    sum_d3 = float(np.sum(diffs**3))
    if sum_d2 <= 0.0:
        a = 0.0
    else:
        a = sum_d3 / (6.0 * (sum_d2**1.5))

    def _adj_alpha(raw_alpha: float) -> float:
        z_a = float(norm.ppf(raw_alpha))
        num = z0 + z_a
        den = 1.0 - a * num
        if abs(den) < 1e-12:
            return raw_alpha
        return float(norm.cdf(z0 + num / den))

    a1 = _adj_alpha(alpha / 2.0)
    a2 = _adj_alpha(1.0 - alpha / 2.0)
    a1 = min(max(a1, 0.0), 1.0)
    a2 = min(max(a2, 0.0), 1.0)
    low = float(np.quantile(samples, a1))
    high = float(np.quantile(samples, a2))
    return low, high, z0, float(a)


def _delta_ece_scalar(
    probs_h: np.ndarray,
    labels_h: np.ndarray,
    probs_e: np.ndarray,
    labels_e: np.ndarray,
    n_bins: int,
    strategy: BinStrategy,
    correct_h: np.ndarray | None = None,
    correct_e: np.ndarray | None = None,
) -> float:
    eh = expected_calibration_error(
        probs_h, labels_h, n_bins=n_bins, strategy=strategy, correctness=correct_h
    )
    ee = expected_calibration_error(
        probs_e, labels_e, n_bins=n_bins, strategy=strategy, correctness=correct_e
    )
    return float(eh.ece - ee.ece)


def _jackknife_delta_ece(
    probs_h: np.ndarray,
    labels_h: np.ndarray,
    probs_e: np.ndarray,
    labels_e: np.ndarray,
    n_bins: int,
    strategy: BinStrategy,
    correct_h: np.ndarray | None = None,
    correct_e: np.ndarray | None = None,
) -> np.ndarray:
    """Leave-one-out ΔECE over the pooled hard∪easy item set."""
    n_h = len(labels_h)
    n_e = len(labels_e)
    out = np.empty(n_h + n_e, dtype=float)
    for i in range(n_h):
        mask = np.ones(n_h, dtype=bool)
        mask[i] = False
        out[i] = _delta_ece_scalar(
            probs_h[mask],
            labels_h[mask],
            probs_e,
            labels_e,
            n_bins,
            strategy,
            None if correct_h is None else correct_h[mask],
            correct_e,
        )
    for j in range(n_e):
        mask = np.ones(n_e, dtype=bool)
        mask[j] = False
        out[n_h + j] = _delta_ece_scalar(
            probs_h,
            labels_h,
            probs_e[mask],
            labels_e[mask],
            n_bins,
            strategy,
            correct_h,
            None if correct_e is None else correct_e[mask],
        )
    return out


def subsample_to_equal_n(
    probs_a: ArrayLike,
    labels_a: ArrayLike,
    item_ids_a: Sequence[Any],
    probs_b: ArrayLike,
    labels_b: ArrayLike,
    item_ids_b: Sequence[Any],
    *,
    seed: int,
    target_n: int | None = None,
) -> EqualNSubsample:
    """Subsample down to equal n (deterministic).

    Default target is ``min(n_a, n_b)``: only the larger stratum loses items.
    ``target_n`` may be smaller than both, in which case both strata are
    subsampled. Stratum A is drawn first, then B, from one Generator seeded
    by ``seed``. Never invents items — only drops surplus.
    """
    pa = np.asarray(probs_a)
    ya = np.asarray(labels_a)
    pb = np.asarray(probs_b)
    yb = np.asarray(labels_b)
    ids_a = list(item_ids_a)
    ids_b = list(item_ids_b)
    if len(ids_a) != len(ya) or len(ids_b) != len(yb):
        raise ValueError("item_ids length must match labels length in each stratum")
    if len(ya) != len(pa) or len(yb) != len(pb):
        raise ValueError("probs/labels length mismatch within a stratum")

    n_a, n_b = len(ya), len(yb)
    if min(n_a, n_b) == 0:
        raise ValueError("both strata must be non-empty")

    natural = min(n_a, n_b)
    if target_n is None:
        target = natural
    else:
        if int(target_n) < 1 or int(target_n) > natural:
            raise ValueError(
                f"target_n={target_n} must be in [1, min(n_a, n_b)={natural}]"
            )
        target = int(target_n)

    rng = np.random.default_rng(seed)
    discarded: list[Any] = []

    def _keep(probs: np.ndarray, labels: np.ndarray, ids: list[Any], n: int) -> tuple:
        if n == target:
            return probs, labels, tuple(ids)
        idx = rng.choice(n, size=target, replace=False)
        idx_sorted = np.sort(idx)
        kept_ids = tuple(ids[i] for i in idx_sorted)
        drop = [ids[i] for i in range(n) if i not in set(idx.tolist())]
        discarded.extend(drop)
        return probs[idx_sorted], labels[idx_sorted], kept_ids

    pa_o, ya_o, ids_a_o = _keep(pa, ya, ids_a, n_a)
    pb_o, yb_o, ids_b_o = _keep(pb, yb, ids_b, n_b)
    return EqualNSubsample(
        probs_a=pa_o,
        labels_a=ya_o,
        item_ids_a=ids_a_o,
        probs_b=pb_o,
        labels_b=yb_o,
        item_ids_b=ids_b_o,
        discarded_item_ids=tuple(discarded),
        target_n=target,
        seed=seed,
    )


def delta_ece(
    probs_hard: ArrayLike,
    labels_hard: ArrayLike,
    probs_easy: ArrayLike,
    labels_easy: ArrayLike,
    *,
    seed: int,
    n_boot: int = 10_000,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    allow_unequal_n: bool = False,
    confidence: Any = None,
    ci_method: Literal["percentile", "bca"] = "percentile",
    alpha: float = 0.05,
    correctness_hard: ArrayLike | None = None,
    correctness_easy: ArrayLike | None = None,
) -> DeltaECEResult:
    """Primary EXP-1 endpoint: ΔECE = ECE(hard) − ECE(easy).

    Point estimate uses top-label ECE on each stratum. The interval is a
    STRATIFIED TWO-SAMPLE bootstrap — the strata are DISJOINT item sets and
    must not be paired. Each resample recomputes ECE from scratch (ECE is a
    nonlinear functional of the whole sample, not a mean of per-item values).

    Both percentile and BCa (jackknife acceleration) intervals are always
    computed. ``ci_method`` selects which pair fills ``ci_low`` / ``ci_high``
    (headline primary). Report both in the paper.

    For comparing two MODELS on the SAME items (Jev vs baseline), use
    :func:`paired_bootstrap` instead.

    Hard guard: ``n_hard`` must equal ``n_easy`` unless
    ``allow_unequal_n=True`` (see Part 0 bias trap / :data:`_UNEQUAL_N_MSG`).
    """
    if confidence is not None:
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if ci_method not in ("percentile", "bca"):
        raise ValueError("ci_method must be 'percentile' or 'bca'")

    ph = _reject_confidence(probs_hard)
    yh = _as_labels(labels_hard)
    pe = _reject_confidence(probs_easy)
    ye = _as_labels(labels_easy)
    if len(ph) != len(yh) or len(pe) != len(ye):
        raise ValueError("probs/labels length mismatch within a stratum")

    n_hard, n_easy = len(yh), len(ye)
    equal = n_hard == n_easy
    if not equal and not allow_unequal_n:
        raise ValueError(_UNEQUAL_N_MSG)
    if (correctness_hard is None) ^ (correctness_easy is None):
        raise ValueError("pass correctness for both strata, or for neither")
    ch = None if correctness_hard is None else np.asarray(correctness_hard, dtype=float)
    ce = None if correctness_easy is None else np.asarray(correctness_easy, dtype=float)
    if ch is not None and (len(ch) != n_hard or len(ce) != n_easy):
        raise ValueError("correctness arrays must match stratum lengths")

    ece_hard = expected_calibration_error(
        ph, yh, n_bins=n_bins, strategy=strategy, correctness=ch
    )
    ece_easy = expected_calibration_error(
        pe, ye, n_bins=n_bins, strategy=strategy, correctness=ce
    )
    point = float(ece_hard.ece - ece_easy.ece)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx_h = rng.integers(0, n_hard, size=n_hard)
        idx_e = rng.integers(0, n_easy, size=n_easy)
        eh = expected_calibration_error(
            ph[idx_h],
            yh[idx_h],
            n_bins=n_bins,
            strategy=strategy,
            correctness=None if ch is None else ch[idx_h],
        )
        ee = expected_calibration_error(
            pe[idx_e],
            ye[idx_e],
            n_bins=n_bins,
            strategy=strategy,
            correctness=None if ce is None else ce[idx_e],
        )
        samples[b] = float(eh.ece - ee.ece)

    p_low = float(np.quantile(samples, alpha / 2.0))
    p_high = float(np.quantile(samples, 1.0 - alpha / 2.0))

    jack = _jackknife_delta_ece(ph, yh, pe, ye, n_bins, strategy, ch, ce)
    b_low, b_high, z0, accel = _bca_endpoints(point, samples, jack, alpha=alpha)

    if ci_method == "bca":
        ci_low, ci_high = b_low, b_high
    else:
        ci_low, ci_high = p_low, p_high

    return DeltaECEResult(
        point=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_hard=n_hard,
        n_easy=n_easy,
        ece_hard=ece_hard,
        ece_easy=ece_easy,
        n_bins=n_bins,
        strategy=strategy,
        seed=seed,
        n_boot=n_boot,
        equal_n=equal,
        alpha=alpha,
        ci_method=ci_method,
        ci_percentile_low=p_low,
        ci_percentile_high=p_high,
        ci_bca_low=b_low,
        ci_bca_high=b_high,
        bca_z0=z0,
        bca_acceleration=accel,
    )


def null_delta_ece_band(
    n_per_stratum: int,
    *,
    seed: int,
    n_sims: int = 1_000,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    n_boot: int = 0,
) -> NullDeltaECEBand:
    """Monte Carlo null: perfectly calibrated strata at equal n.

    Draws Bernoulli labels from known probs independently in each stratum,
    computes ΔECE (point only; set n_boot>0 to also bootstrap each sim — slow).
    The [p025, p975] band is the floor an observed effect must clear.
    """
    if n_per_stratum < 1:
        raise ValueError("n_per_stratum must be >= 1")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for i in range(n_sims):
        # Independent perfectly calibrated draws per stratum
        ph = rng.uniform(0.05, 0.95, n_per_stratum)
        yh = (rng.uniform(0, 1, n_per_stratum) < ph).astype(int)
        pe = rng.uniform(0.05, 0.95, n_per_stratum)
        ye = (rng.uniform(0, 1, n_per_stratum) < pe).astype(int)
        if n_boot > 0:
            res = delta_ece(
                ph,
                yh,
                pe,
                ye,
                seed=int(rng.integers(0, 2**31 - 1)),
                n_boot=n_boot,
                n_bins=n_bins,
                strategy=strategy,
            )
            samples.append(res.point)
        else:
            eh = expected_calibration_error(ph, yh, n_bins=n_bins, strategy=strategy)
            ee = expected_calibration_error(pe, ye, n_bins=n_bins, strategy=strategy)
            samples.append(float(eh.ece - ee.ece))
    arr = np.asarray(samples, dtype=float)
    return NullDeltaECEBand(
        n_per_stratum=n_per_stratum,
        n_bins=n_bins,
        strategy=strategy,
        n_sims=n_sims,
        seed=seed,
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        p025=float(np.quantile(arr, 0.025)),
        p975=float(np.quantile(arr, 0.975)),
        samples=tuple(float(x) for x in arr),
    )


def maximum_calibration_error(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    *,
    confidence: Any = None,
) -> ECEResult:
    """MCE is the ``mce`` field on the same occupancy-bearing result as ECE."""
    return expected_calibration_error(
        probs, labels, n_bins=n_bins, strategy=strategy, confidence=confidence
    )


def brier_score(
    probs: ArrayLike,
    labels: ArrayLike,
    *,
    confidence: Any = None,
) -> float:
    """Brier score (sklearn). Binary 1-D or multiclass 2-D probabilities."""
    if confidence is not None:
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    p = _reject_confidence(probs)
    y = _as_labels(labels)
    if p.ndim == 1:
        y_bin = y.astype(int)
        return float(brier_score_loss(y_bin, p))
    # multiclass: mean squared error over one-hot
    if y.ndim == 1:
        n_classes = p.shape[1]
        y_oh = np.eye(n_classes)[y.astype(int)]
    else:
        y_oh = y.astype(float)
    return float(np.mean(np.sum((p - y_oh) ** 2, axis=1)))


def reliability_curve(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
    strategy: BinStrategy = "uniform",
    *,
    confidence: Any = None,
) -> ReliabilityCurve:
    """Reliability diagram data (same bins / occupancy as ECE)."""
    ece = expected_calibration_error(
        probs, labels, n_bins=n_bins, strategy=strategy, confidence=confidence
    )
    return ReliabilityCurve(
        bin_mean_confidence=ece.bin_mean_confidence,
        bin_accuracy=ece.bin_accuracy,
        bin_counts=ece.bin_counts,
        bin_edges=ece.bin_edges,
        strategy=ece.strategy,
        n_bins=ece.n_bins,
    )


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return float("nan")
    x = x.astype(float)
    y = y.astype(float)
    x_mean = x.mean()
    y_mean = y.mean()
    den = np.sum((x - x_mean) ** 2)
    if den == 0:
        return float("nan")
    return float(np.sum((x - x_mean) * (y - y_mean)) / den)


def calibration_by_tier(
    probs_by_tier: Mapping[str, ArrayLike],
    labels_by_tier: Mapping[str, ArrayLike],
    n_bins: int = 10,
    tier_order: Sequence[str] | None = None,
    *,
    confidence: Any = None,
) -> CalibrationByTier:
    """Flagship: ECE-vs-accuracy shape across difficulty tiers.

    Report both uniform and quantile ECE, with occupancy on every tier.
    """
    if confidence is not None:
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    if set(probs_by_tier) != set(labels_by_tier):
        raise ValueError("probs_by_tier and labels_by_tier must share the same keys")
    order = list(tier_order) if tier_order is not None else list(probs_by_tier.keys())
    for t in order:
        if t not in probs_by_tier:
            raise ValueError(f"tier {t!r} missing from probs_by_tier")

    tiers: list[TierCalibration] = []
    accs: list[float] = []
    ece_u: list[float] = []
    ece_q: list[float] = []
    names: list[str] = []

    for tier in order:
        p = _reject_confidence(probs_by_tier[tier])
        y = _as_labels(labels_by_tier[tier])
        scores, outcomes = _for_ece(p, y)
        # Tier accuracy = decision accuracy (not mean label)
        _, correct = _for_decisions(p, y)
        acc = float(correct.mean()) if len(correct) else float("nan")
        eu = _ece_from_bins(scores, outcomes, n_bins=n_bins, strategy="uniform")
        eq = _ece_from_bins(scores, outcomes, n_bins=n_bins, strategy="quantile")
        br = brier_score(p, y)
        tiers.append(
            TierCalibration(
                tier=tier,
                n=len(correct),
                accuracy=acc,
                ece_uniform=eu,
                ece_quantile=eq,
                brier=br,
            )
        )
        names.append(tier)
        accs.append(acc)
        ece_u.append(eu.ece)
        ece_q.append(eq.ece)

    return CalibrationByTier(
        tiers=tuple(tiers),
        accuracies=tuple(accs),
        ece_uniform=tuple(ece_u),
        ece_quantile=tuple(ece_q),
        tier_names=tuple(names),
        slope_uniform=_ols_slope(np.asarray(accs), np.asarray(ece_u)),
        slope_quantile=_ols_slope(np.asarray(accs), np.asarray(ece_q)),
    )


# ---------------------------------------------------------------------------
# Discrimination
# ---------------------------------------------------------------------------


def accuracy(preds: ArrayLike, labels: ArrayLike) -> float:
    p = np.asarray(preds)
    y = np.asarray(labels)
    return float(np.mean(p == y))


def macro_f1(preds: ArrayLike, labels: ArrayLike) -> float:
    return float(
        skm.f1_score(np.asarray(labels), np.asarray(preds), average="macro", zero_division=0)
    )


def confusion_matrix(
    preds: ArrayLike, labels: ArrayLike, labels_order: Sequence[Any] | None = None
) -> np.ndarray:
    return skm.confusion_matrix(
        np.asarray(labels), np.asarray(preds), labels=labels_order
    )


def auroc(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    confidence: Any = None,
) -> float:
    """Area under the ROC curve for binary scores.

    IMPORTANT — error-RANKING (AUROC) is NOT calibration
    ----------------------------------------------------
    AUROC asks whether higher scores tend to rank positives above negatives.
    It does **not** ask whether a score of 0.9 means \"90% correct\".

    An existing public Jev benchmark reported error-ranking AUROC and had to
    clarify that this is not calibration in the \"0.9 means 90%\" sense. Do not
    blur the two in this codebase or in the paper.

    For calibration (\"does 0.9 mean ~90% accurate?\"), use
    :func:`expected_calibration_error` on probabilities / noul.
    """
    if confidence is not None:
        # AUROC is discrimination, not calibration — but if someone passes the
        # keyword `confidence=` thinking AUROC *is* calibration, still warn?
        # Spec only requires calibration functions to reject confidence.
        # Allow scores to be confidence when used as a ranking signal.
        pass
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    if isinstance(s, ConfidenceArray) or getattr(scores, "_jevbench_is_confidence", False):
        # Ranking on confidence is allowed — that is a routing experiment.
        s = np.asarray(s, dtype=float)
    return float(skm.roc_auc_score(y, s))


def auprc(scores: ArrayLike, labels: ArrayLike) -> float:
    """Area under the precision-recall curve (binary)."""
    y = np.asarray(labels).astype(int)
    s = np.asarray(scores, dtype=float)
    return float(skm.average_precision_score(y, s))


# ---------------------------------------------------------------------------
# Selective prediction
# ---------------------------------------------------------------------------


def risk_coverage_curve(
    probs: ArrayLike,
    labels: ArrayLike,
    n_thresholds: int = 100,
    *,
    confidence: Any = None,
) -> RiskCoverageCurve:
    """Risk-coverage curve using probability / max-prob as the retain score.

    Risk = 1 - accuracy on the retained (high-score) subset.
    Coverage = fraction retained.
    """
    if confidence is not None:
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    p = _reject_confidence(probs)
    y = _as_labels(labels)
    scores, correct = _for_decisions(p, y)
    # Thresholds from high to low confidence
    thresholds = np.unique(scores)
    if len(thresholds) > n_thresholds:
        thresholds = np.quantile(scores, np.linspace(0, 1, n_thresholds))
    thresholds = np.sort(thresholds)

    covs: list[float] = []
    risks: list[float] = []
    ths: list[float] = []
    n = len(scores)
    for t in thresholds:
        mask = scores >= t
        k = int(mask.sum())
        if k == 0:
            continue
        cov = k / n
        risk = 1.0 - float(correct[mask].mean())
        covs.append(cov)
        risks.append(risk)
        ths.append(float(t))
    # Integrate risk over coverage (trapezoid); also include (0, ?) endpoint
    if not covs:
        return RiskCoverageCurve((), (), (), aurc=float("nan"))
    # Sort by coverage ascending for integration
    order = np.argsort(covs)
    c = np.asarray(covs)[order]
    r = np.asarray(risks)[order]
    # Prepend coverage 0
    c_int = np.concatenate([[0.0], c])
    r_int = np.concatenate([[r[0]], r])
    aurc = float(np.trapezoid(r_int, c_int))
    return RiskCoverageCurve(
        coverage=tuple(float(x) for x in c),
        risk=tuple(float(x) for x in r),
        thresholds=tuple(float(thresholds[i]) for i in np.argsort(covs)),
        aurc=aurc,
    )


def aurc(
    probs: ArrayLike,
    labels: ArrayLike,
    *,
    confidence: Any = None,
) -> float:
    """Area under the risk-coverage curve."""
    return risk_coverage_curve(probs, labels, confidence=confidence).aurc


def automation_at_accuracy(
    probs: ArrayLike,
    labels: ArrayLike,
    target_accuracy: float = 0.90,
    *,
    confidence: Any = None,
) -> AutomationResult:
    """Largest fraction of decisions automatable at ``target_accuracy``.

    Answers: \"what percentage of decisions can you automate at 90% accuracy?\"
    Sweep thresholds on the probability / max-prob score; pick the lowest
    threshold whose retained accuracy is still >= target (maximising coverage).
    """
    if confidence is not None:
        raise TypeError(_CONFIDENCE_REJECT_MSG)
    if not 0.0 < target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in (0, 1]")
    p = _reject_confidence(probs)
    y = _as_labels(labels)
    scores, correct = _for_decisions(p, y)
    n = len(scores)
    if n == 0:
        return AutomationResult(
            target_accuracy=target_accuracy,
            threshold=float("nan"),
            coverage=0.0,
            accuracy_on_automated=float("nan"),
            n_automated=0,
            n_total=0,
            feasible=False,
        )

    best: AutomationResult | None = None
    for t in sorted(np.unique(scores)):
        mask = scores >= float(t)
        k = int(mask.sum())
        if k == 0:
            continue
        acc = float(correct[mask].mean())
        if acc + 1e-12 >= target_accuracy:
            cand = AutomationResult(
                target_accuracy=target_accuracy,
                threshold=float(t),
                coverage=k / n,
                accuracy_on_automated=acc,
                n_automated=k,
                n_total=n,
                feasible=True,
            )
            if best is None or cand.coverage > best.coverage:
                best = cand
    if best is None:
        return AutomationResult(
            target_accuracy=target_accuracy,
            threshold=float("nan"),
            coverage=0.0,
            accuracy_on_automated=float("nan"),
            n_automated=0,
            n_total=n,
            feasible=False,
        )
    return best


def selective_from_confidence(
    confidence: ArrayLike,
    correct: ArrayLike,
    target_accuracy: float = 0.90,
) -> AutomationResult:
    """Selective automation using Jev ``confidence`` as a routing signal.

    This is the legitimate use of confidence — NOT calibration. ``correct``
    must be a boolean / {0,1} array of whether the decision was right.
    """
    conf = np.asarray(confidence, dtype=float)
    corr = np.asarray(correct).astype(int)
    if len(conf) != len(corr):
        raise ValueError("confidence and correct must have the same length")
    # Reuse automation sweep by treating confidence as a 1-D \"prob\" and
    # correct as labels — but automation_at_accuracy would reject if marked.
    # Inline the sweep to avoid the calibration API.
    n = len(conf)
    best: AutomationResult | None = None
    for t in sorted(np.unique(conf)):
        mask = conf >= float(t)
        k = int(mask.sum())
        if k == 0:
            continue
        acc = float(corr[mask].mean())
        if acc + 1e-12 >= target_accuracy:
            cand = AutomationResult(
                target_accuracy=target_accuracy,
                threshold=float(t),
                coverage=k / n,
                accuracy_on_automated=acc,
                n_automated=k,
                n_total=n,
                feasible=True,
            )
            if best is None or cand.coverage > best.coverage:
                best = cand
    if best is None:
        return AutomationResult(
            target_accuracy=target_accuracy,
            threshold=float("nan"),
            coverage=0.0,
            accuracy_on_automated=float("nan"),
            n_automated=0,
            n_total=n,
            feasible=False,
        )
    return best


# ---------------------------------------------------------------------------
# Invariants (EXP-4) — measure violation; do not assume they hold
# ---------------------------------------------------------------------------


def primitive_disagreement_rate(
    noul_probs: ArrayLike,
    choice_probs: ArrayLike,
    *,
    noul_threshold: float = 0.5,
) -> float:
    """Fraction of items where Noul threshold and Choice argmax disagree.

    ``choice_probs`` may be 1-D P(positive option) or 2-D with column 0 =
    negative / column 1 = positive for binary choice. For 2-D with named
    options already reduced, pass 1-D P(option corresponding to noul=true).
    """
    noul = np.asarray(noul_probs, dtype=float)
    choice = np.asarray(choice_probs, dtype=float)
    if choice.ndim == 2:
        if choice.shape[1] < 2:
            raise ValueError("choice_probs 2-D must have >= 2 columns")
        # Assume last column is the "true"/positive option aligned with noul
        choice_pos = choice[:, -1]
        choice_hat = choice_pos >= 0.5
    else:
        choice_hat = choice >= 0.5
    noul_hat = noul >= noul_threshold
    if len(noul_hat) != len(choice_hat):
        raise ValueError("noul_probs and choice_probs length mismatch")
    return float(np.mean(noul_hat != choice_hat))


def negation_sum_distribution(
    p_x: ArrayLike,
    p_not_x: ArrayLike,
    *,
    eps: float = 0.05,
) -> NegationSumResult:
    """Distribution of ``p(x) + p(¬x)``; ideal is 1.0 — measure drift.

    Do NOT assume the invariant holds. TypeSafe's own docs publish an example
    summing to 1.19.
    """
    a = np.asarray(p_x, dtype=float)
    b = np.asarray(p_not_x, dtype=float)
    if a.shape != b.shape:
        raise ValueError("p_x and p_not_x must have the same shape")
    s = a + b
    return NegationSumResult(
        mean=float(s.mean()) if len(s) else float("nan"),
        std=float(s.std()) if len(s) else float("nan"),
        median=float(np.median(s)) if len(s) else float("nan"),
        mean_abs_deviation=float(np.mean(np.abs(s - 1.0))) if len(s) else float("nan"),
        frac_outside_eps=float(np.mean(np.abs(s - 1.0) > eps)) if len(s) else float("nan"),
        eps=eps,
        values=tuple(float(x) for x in s),
    )


# ---------------------------------------------------------------------------
# Uncertainty — every comparative claim ships an interval
# ---------------------------------------------------------------------------


def paired_bootstrap(
    a: ArrayLike,
    b: ArrayLike,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """PAIRED bootstrap CI for ``statistic(a, b)`` on the SAME items.

    Correct when comparing two MODELS on identical items (Jev vs baseline):
    resample item indices once and apply to both arrays together.

    Incorrect for ΔECE across difficulty strata — those are DISJOINT item
    sets. Use :func:`delta_ece` (stratified two-sample bootstrap) instead.

    An interval crossing zero is **inconclusive**, never \"equivalent\".
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if len(a) != len(b):
        raise ValueError("a and b must have the same length")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    m = len(a)
    point = float(statistic(a, b))
    if m == 0:
        return BootstrapCI(point=point, low=float("nan"), high=float("nan"), n_resamples=n, alpha=alpha)
    idx = rng.integers(0, m, size=(n, m))
    samples = np.empty(n, dtype=float)
    for i in range(n):
        samples[i] = float(statistic(a[idx[i]], b[idx[i]]))
    low = float(np.quantile(samples, alpha / 2))
    high = float(np.quantile(samples, 1 - alpha / 2))
    return BootstrapCI(point=point, low=low, high=high, n_resamples=n, alpha=alpha)


def accuracy_delta(a_correct: ArrayLike, b_correct: ArrayLike) -> float:
    """mean(a) - mean(b) on paired boolean correctness."""
    a = np.asarray(a_correct).astype(float)
    b = np.asarray(b_correct).astype(float)
    return float(a.mean() - b.mean())


def mcnemar(
    a_correct: ArrayLike,
    b_correct: ArrayLike,
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> McNemarResult:
    """McNemar's test on paired correctness, plus bootstrap CI on Δ accuracy.

    Always returns an interval via ``accuracy_delta`` — no bare point estimate
    for the comparative claim.
    """
    a = np.asarray(a_correct).astype(bool)
    b = np.asarray(b_correct).astype(bool)
    if len(a) != len(b):
        raise ValueError("a_correct and b_correct must have the same length")
    n01 = int(np.sum(~a & b))  # A wrong, B correct
    n10 = int(np.sum(a & ~b))  # A correct, B wrong
    # Continuity-corrected McNemar chi-square
    if n01 + n10 == 0:
        stat = 0.0
        pvalue = 1.0
    else:
        stat = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
        # chi-square df=1 survival function
        from scipy.stats import chi2

        pvalue = float(chi2.sf(stat, df=1))
    delta = paired_bootstrap(
        a.astype(float),
        b.astype(float),
        accuracy_delta,
        n=n_bootstrap,
        seed=seed,
    )
    return McNemarResult(
        statistic=float(stat),
        pvalue=pvalue,
        n_a_only=n10,
        n_b_only=n01,
        accuracy_delta=delta,
    )


# ---------------------------------------------------------------------------
# netcal agreement helpers (used by tests / appendix)
# ---------------------------------------------------------------------------


def netcal_ece(
    probs: ArrayLike,
    labels: ArrayLike,
    n_bins: int = 10,
) -> float:
    """Scalar ECE from netcal on our top-label (conf, correct) inputs.

    We feed netcal the same ``conf_i`` / ``correct_i`` arrays our ECE uses, so
    agreement tests the binning arithmetic — not whether netcal's default
    binary API matches top-label ECE. Passing raw ``p_i`` and ``y_i`` into
    ``netcal.metrics.ECE.measure`` is the classic P(y=1) reliability variant;
    that is a different estimator (documented in the agreement tests).
    """
    p = _reject_confidence(probs)
    y = _as_labels(labels)
    scores, correct = _for_ece(p, y)
    try:
        from netcal.metrics import ECE as NetcalECE
    except ImportError as exc:
        raise ImportError(
            "netcal is required for netcal_ece(); install netcal or use "
            "expected_calibration_error() which reports occupancy."
        ) from exc
    return float(NetcalECE(n_bins).measure(scores, correct))
