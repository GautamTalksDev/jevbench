"""EXP-6 — label vs confidence vs distribution stability.

WHY: two published sources disagree. A Zenodo toolkit paper (daf-jev /
Jev in Practice, Zenodo 22816188) reports Jev's confidence is self-consistent
across repeated evaluations. Public benchmark repos report Jev's *answers*
move between passes. This module separates the measurements so both can be
true.

Three quantities — NEVER collapsed into one "stability" number:

  1. LABEL stability — does the decision flip?
  2. CONFIDENCE stability — does the sharpness scalar drift?
  3. DISTRIBUTION stability — do the probability vectors move? (TV + JS)

Central analysis: flip_rate vs boundary_distance
  (|mean confidence − decision threshold|). Hypothesis: confidence is stable
  AND labels flip, concentrated near the boundary. If that holds, both
  published results are correct and were measuring different quantities —
  a reconciliation, not a refutation of either group.

Pure functions over repeat panels. No I/O. No network.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from jevbench.metrics import BootstrapCI, paired_bootstrap

DecisionKind = Literal["choice", "noul"]

# Jensen–Shannon with natural log is bounded by ln(2).
JS_MAX = float(np.log(2.0))

_RECONCILIATION = (
    "If confidence is stable while labels flip near the decision boundary, "
    "both published results are correct and were measuring different "
    "quantities: Zenodo/daf-jev measured confidence self-consistency; the "
    "public benches measured answer/label movement. That is a reconciliation, "
    "not a refutation of either group."
)


# ---------------------------------------------------------------------------
# Divergences
# ---------------------------------------------------------------------------


def total_variation(p: ArrayLike, q: ArrayLike) -> float:
    """TV = 0.5 * sum_k |p_k - q_k|. Both must be non-negative and same length."""
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    if np.any(p < -1e-12) or np.any(q < -1e-12):
        raise ValueError("probabilities must be non-negative")
    return float(0.5 * np.sum(np.abs(p - q)))


def _kl_term(p: np.ndarray, m: np.ndarray) -> float:
    """KL(p || m) with 0 log 0 = 0; finite when p has exact zeros."""
    out = 0.0
    for pi, mi in zip(p, m, strict=True):
        if pi <= 0.0:
            continue
        if mi <= 0.0:
            # Should not happen if m = 0.5(p+q) and pi > 0, but guard.
            return float("inf")
        out += float(pi * np.log(pi / mi))
    return out


def jensen_shannon(p: ArrayLike, q: ArrayLike) -> float:
    """Symmetric JS divergence (nats). Bounded in [0, ln 2]; 0 iff p == q.

    Prefer JS over KL: KL is unbounded and undefined on zero support, and Jev
    returns exact 0.0 probabilities.
    """
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    # Renormalise lightly in case of float drift; keep zeros exact.
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("probability vectors must have positive mass")
    p = p / p_sum
    q = q / q_sum
    m = 0.5 * (p + q)
    js = 0.5 * _kl_term(p, m) + 0.5 * _kl_term(q, m)
    # Numerical ceiling
    return float(min(max(js, 0.0), JS_MAX + 1e-12))


# ---------------------------------------------------------------------------
# Per-item / panel types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepeatObservation:
    """One evaluation of one item."""

    item_id: str
    repeat: int
    decision: str | int | bool  # argmax label or noul threshold bit
    probabilities: tuple[float, ...]  # aligned to a fixed label order
    confidence: float  # Jev sharpness, or max-prob if absent
    tier: str = "unknown"
    serving_path: str = "native"
    resolved_model: str = "unknown"
    wall_ms: float | None = None
    client: str = "jev"


@dataclass(frozen=True)
class ItemLabelStability:
    item_id: str
    flips: bool  # True if ≥2 distinct decisions across repeats
    n_distinct_labels: int
    modal_label: str | int | bool
    modal_count: int
    decisions: tuple[Any, ...]
    tier: str
    serving_path: str
    resolved_models: tuple[str, ...]


@dataclass(frozen=True)
class ItemConfidenceStability:
    item_id: str
    n_repeats: int
    mean: float
    sd: float
    range: float  # max - min
    max_abs_dev_from_mean: float
    values: tuple[float, ...]
    tier: str
    serving_path: str
    resolved_models: tuple[str, ...]


@dataclass(frozen=True)
class ItemDistributionStability:
    item_id: str
    n_pairs: int
    mean_tv: float
    max_tv: float
    mean_js: float
    max_js: float
    tier: str
    serving_path: str
    resolved_models: tuple[str, ...]


@dataclass(frozen=True)
class ModelVersionReport:
    """Surface mid-run alias moves loudly — do not average over them."""

    unique_models: tuple[str, ...]
    n_changes: int  # count of adjacent-repeat model string changes (global timeline)
    change_events: tuple[dict[str, Any], ...]
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryBin:
    decile: int
    lo: float
    hi: float
    n_items: int
    flip_rate: float
    mean_boundary_distance: float
    mean_confidence_sd: float


@dataclass(frozen=True)
class BoundaryAnalysis:
    """Central EXP-6 analysis: flip_rate vs |mean conf − threshold|."""

    threshold: float
    n_items: int
    flip_rate_overall: float
    mean_confidence_sd: float
    # OLS: flip (0/1) ~ boundary_distance
    slope: float
    intercept: float
    # Spearman-ish: correlation between distance and flip
    corr_distance_flip: float
    bins: tuple[BoundaryBin, ...]
    supports_reconciliation: bool
    statement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StabilityReport:
    """Full EXP-6 panel — three metrics kept separate."""

    n_items: int
    n_repeats: int
    client: str
    label: dict[str, Any]
    confidence: dict[str, Any]
    distribution: dict[str, Any]
    boundary: BoundaryAnalysis
    model_versions: ModelVersionReport
    by_tier: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_serving_path: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_resolved_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_boundary_decile: dict[str, dict[str, Any]] = field(default_factory=dict)
    reconciliation_note: str = _RECONCILIATION

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["boundary"] = self.boundary.to_dict()
        d["model_versions"] = self.model_versions.to_dict()
        return d


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def decision_from_probs(
    probs: ArrayLike,
    *,
    kind: DecisionKind = "choice",
    label_order: Sequence[str] | None = None,
    noul_threshold: float = 0.5,
) -> str | int | bool:
    """Argmax (Choice) or threshold decision (Noul)."""
    p = np.asarray(probs, dtype=float).ravel()
    if kind == "noul":
        if p.size != 1:
            # Allow 2-class [p_pos, p_neg] — use first as P(positive)
            p_pos = float(p[0]) if p.size >= 1 else float("nan")
        else:
            p_pos = float(p[0])
        return bool(p_pos >= noul_threshold)
    idx = int(np.argmax(p))
    if label_order is not None:
        return label_order[idx]
    return idx


def confidence_from_observation(
    probs: ArrayLike,
    confidence: float | None = None,
    *,
    kind: DecisionKind = "choice",
) -> float:
    """Prefer Jev sharpness; else top-label confidence of the predicted class."""
    if confidence is not None and np.isfinite(confidence):
        return float(confidence)
    p = np.asarray(probs, dtype=float).ravel()
    if kind == "noul":
        p_pos = float(p[0])
        return float(max(p_pos, 1.0 - p_pos))
    return float(np.max(p))


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def _group_by_item(
    observations: Sequence[RepeatObservation],
) -> dict[str, list[RepeatObservation]]:
    by: dict[str, list[RepeatObservation]] = defaultdict(list)
    for obs in observations:
        by[obs.item_id].append(obs)
    for item_id in by:
        by[item_id] = sorted(by[item_id], key=lambda o: o.repeat)
    return dict(by)


def label_stability_per_item(
    observations: Sequence[RepeatObservation],
) -> list[ItemLabelStability]:
    out: list[ItemLabelStability] = []
    for item_id, reps in _group_by_item(observations).items():
        decisions = tuple(r.decision for r in reps)
        counts = Counter(decisions)
        modal, modal_count = counts.most_common(1)[0]
        distinct = len(counts)
        out.append(
            ItemLabelStability(
                item_id=item_id,
                flips=distinct > 1,
                n_distinct_labels=distinct,
                modal_label=modal,
                modal_count=modal_count,
                decisions=decisions,
                tier=reps[0].tier,
                serving_path=reps[0].serving_path,
                resolved_models=tuple(r.resolved_model for r in reps),
            )
        )
    return out


def confidence_stability_per_item(
    observations: Sequence[RepeatObservation],
) -> list[ItemConfidenceStability]:
    out: list[ItemConfidenceStability] = []
    for item_id, reps in _group_by_item(observations).items():
        vals = np.asarray([r.confidence for r in reps], dtype=float)
        mean = float(vals.mean())
        out.append(
            ItemConfidenceStability(
                item_id=item_id,
                n_repeats=len(vals),
                mean=mean,
                sd=float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                range=float(vals.max() - vals.min()) if len(vals) else 0.0,
                max_abs_dev_from_mean=float(np.max(np.abs(vals - mean))) if len(vals) else 0.0,
                values=tuple(float(v) for v in vals),
                tier=reps[0].tier,
                serving_path=reps[0].serving_path,
                resolved_models=tuple(r.resolved_model for r in reps),
            )
        )
    return out


def distribution_stability_per_item(
    observations: Sequence[RepeatObservation],
) -> list[ItemDistributionStability]:
    out: list[ItemDistributionStability] = []
    for item_id, reps in _group_by_item(observations).items():
        tvs: list[float] = []
        jss: list[float] = []
        for i in range(len(reps)):
            for j in range(i + 1, len(reps)):
                pi = np.asarray(reps[i].probabilities, dtype=float)
                pj = np.asarray(reps[j].probabilities, dtype=float)
                tvs.append(total_variation(pi, pj))
                jss.append(jensen_shannon(pi, pj))
        out.append(
            ItemDistributionStability(
                item_id=item_id,
                n_pairs=len(tvs),
                mean_tv=float(np.mean(tvs)) if tvs else 0.0,
                max_tv=float(np.max(tvs)) if tvs else 0.0,
                mean_js=float(np.mean(jss)) if jss else 0.0,
                max_js=float(np.max(jss)) if jss else 0.0,
                tier=reps[0].tier,
                serving_path=reps[0].serving_path,
                resolved_models=tuple(r.resolved_model for r in reps),
            )
        )
    return out


def flip_rate(
    label_items: Sequence[ItemLabelStability],
    *,
    seed: int = 0,
    n_boot: int = 10_000,
) -> BootstrapCI:
    """Fraction of items that flip, with paired-style item bootstrap CI."""
    flags = np.asarray([1.0 if it.flips else 0.0 for it in label_items], dtype=float)
    if len(flags) == 0:
        return BootstrapCI(
            point=float("nan"), low=float("nan"), high=float("nan"), n_resamples=n_boot
        )
    # Bootstrap over items (not paired across models — single rate)
    rng = np.random.default_rng(seed)
    n = len(flags)
    point = float(flags.mean())
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        samples[b] = float(flags[rng.integers(0, n, size=n)].mean())
    return BootstrapCI(
        point=point,
        low=float(np.quantile(samples, 0.025)),
        high=float(np.quantile(samples, 0.975)),
        n_resamples=n_boot,
    )


# ---------------------------------------------------------------------------
# Model version timeline
# ---------------------------------------------------------------------------


def detect_model_version_changes(
    observations: Sequence[RepeatObservation],
) -> ModelVersionReport:
    """Log resolved ``model`` on every call; surface mid-run alias moves."""
    # Global timeline: sort by (repeat, item_id) so interleaved schedule is visible
    ordered = sorted(observations, key=lambda o: (o.repeat, o.item_id, o.wall_ms or 0.0))
    unique = tuple(sorted({o.resolved_model for o in ordered}))
    events: list[dict[str, Any]] = []
    n_changes = 0
    prev: str | None = None
    prev_obs: RepeatObservation | None = None
    for obs in ordered:
        if prev is not None and obs.resolved_model != prev:
            n_changes += 1
            events.append(
                {
                    "from": prev,
                    "to": obs.resolved_model,
                    "at_repeat": obs.repeat,
                    "at_item_id": obs.item_id,
                    "prev_item_id": prev_obs.item_id if prev_obs else None,
                    "wall_ms": obs.wall_ms,
                }
            )
        prev = obs.resolved_model
        prev_obs = obs
    warning = None
    if n_changes > 0:
        warning = (
            f"PUBLISHABLE: resolved model string changed {n_changes} time(s) mid-run "
            f"({unique}). Do not average across alias moves — report strata separately."
        )
    return ModelVersionReport(
        unique_models=unique,
        n_changes=n_changes,
        change_events=tuple(events),
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Boundary analysis
# ---------------------------------------------------------------------------


def boundary_distance(mean_confidence: float, threshold: float = 0.5) -> float:
    return abs(float(mean_confidence) - float(threshold))


def boundary_analysis(
    label_items: Sequence[ItemLabelStability],
    confidence_items: Sequence[ItemConfidenceStability],
    *,
    threshold: float = 0.5,
    n_deciles: int = 10,
) -> BoundaryAnalysis:
    """Regress / bin flip_rate against |mean confidence − threshold|.

    Supports reconciliation when flips concentrate at low boundary distance
    while mean confidence SD stays small.
    """
    conf_by_id = {c.item_id: c for c in confidence_items}
    distances: list[float] = []
    flips: list[float] = []
    sds: list[float] = []
    for lab in label_items:
        conf = conf_by_id.get(lab.item_id)
        if conf is None:
            continue
        d = boundary_distance(conf.mean, threshold)
        distances.append(d)
        flips.append(1.0 if lab.flips else 0.0)
        sds.append(conf.sd)

    dist = np.asarray(distances, dtype=float)
    flip = np.asarray(flips, dtype=float)
    sd_arr = np.asarray(sds, dtype=float)
    n = len(dist)
    if n < 2:
        return BoundaryAnalysis(
            threshold=threshold,
            n_items=n,
            flip_rate_overall=float(flip.mean()) if n else float("nan"),
            mean_confidence_sd=float(sd_arr.mean()) if n else float("nan"),
            slope=float("nan"),
            intercept=float("nan"),
            corr_distance_flip=float("nan"),
            bins=(),
            supports_reconciliation=False,
            statement="Insufficient items for boundary analysis.",
        )

    # OLS flip ~ distance
    x_mean, y_mean = dist.mean(), flip.mean()
    den = np.sum((dist - x_mean) ** 2)
    slope = float(np.sum((dist - x_mean) * (flip - y_mean)) / den) if den > 0 else float("nan")
    intercept = float(y_mean - slope * x_mean)
    if dist.std() > 0 and flip.std() > 0:
        corr = float(np.corrcoef(dist, flip)[0, 1])
    else:
        corr = float("nan")

    # Decile bins
    qs = np.quantile(dist, np.linspace(0, 1, n_deciles + 1))
    # Nudge duplicate edges
    for i in range(1, len(qs)):
        if qs[i] <= qs[i - 1]:
            qs[i] = qs[i - 1] + 1e-12
    bins: list[BoundaryBin] = []
    for d in range(n_deciles):
        lo, hi = float(qs[d]), float(qs[d + 1])
        if d < n_deciles - 1:
            mask = (dist >= lo) & (dist < hi)
        else:
            mask = (dist >= lo) & (dist <= hi)
        if not np.any(mask):
            bins.append(
                BoundaryBin(
                    decile=d,
                    lo=lo,
                    hi=hi,
                    n_items=0,
                    flip_rate=float("nan"),
                    mean_boundary_distance=float("nan"),
                    mean_confidence_sd=float("nan"),
                )
            )
            continue
        bins.append(
            BoundaryBin(
                decile=d,
                lo=lo,
                hi=hi,
                n_items=int(mask.sum()),
                flip_rate=float(flip[mask].mean()),
                mean_boundary_distance=float(dist[mask].mean()),
                mean_confidence_sd=float(sd_arr[mask].mean()),
            )
        )

    # Reconciliation: near-boundary bins flip more; overall confidence SD low.
    # Use lowest/highest *occupied* bins (deciles may be empty when distances
    # are bimodal — two point masses).
    occupied = [b for b in bins if b.n_items > 0 and np.isfinite(b.flip_rate)]
    if len(occupied) >= 2:
        near_flip = float(occupied[0].flip_rate)
        far_flip = float(occupied[-1].flip_rate)
        # If multiple bins share the near/far mass, average extremes
        n_edge = max(1, len(occupied) // 3)
        near_flip = float(np.mean([b.flip_rate for b in occupied[:n_edge]]))
        far_flip = float(np.mean([b.flip_rate for b in occupied[-n_edge:]]))
    else:
        near_flip = float("nan")
        far_flip = float("nan")
    mean_sd = float(sd_arr.mean())
    flip_overall = float(flip.mean())
    supports = (
        np.isfinite(near_flip)
        and np.isfinite(far_flip)
        and near_flip > far_flip
        and mean_sd < 0.05  # confidence relatively stable
        and flip_overall > 0.0
    )
    if supports:
        statement = (
            f"Data support the reconciliation: flip_rate is higher near the "
            f"decision boundary (near-decile flip={near_flip:.3f} vs far={far_flip:.3f}) "
            f"while mean confidence SD={mean_sd:.4f} stays small. "
            + _RECONCILIATION
        )
    elif flip_overall == 0.0 and mean_sd < 0.05:
        statement = (
            "Labels and confidence are both stable in this panel — supports the "
            "Zenodo/daf-jev confidence-consistency claim under these conditions; "
            "does not exercise the public-bench flip reports."
        )
        supports = False
    else:
        statement = (
            f"Boundary pattern inconclusive or contrary "
            f"(near_flip={near_flip!r}, far_flip={far_flip!r}, "
            f"mean_conf_sd={mean_sd:.4f}, flip_rate={flip_overall:.3f}). "
            + _RECONCILIATION
        )

    return BoundaryAnalysis(
        threshold=threshold,
        n_items=n,
        flip_rate_overall=flip_overall,
        mean_confidence_sd=mean_sd,
        slope=slope,
        intercept=intercept,
        corr_distance_flip=corr,
        bins=tuple(bins),
        supports_reconciliation=supports,
        statement=statement,
    )


# ---------------------------------------------------------------------------
# Stratification helpers
# ---------------------------------------------------------------------------


def _subset_summary(
    label_items: Sequence[ItemLabelStability],
    confidence_items: Sequence[ItemConfidenceStability],
    distribution_items: Sequence[ItemDistributionStability],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    fr = flip_rate(label_items, seed=seed, n_boot=2_000)
    return {
        "n_items": len(label_items),
        "flip_rate": fr.to_dict(),
        "mean_confidence_sd": float(np.mean([c.sd for c in confidence_items]))
        if confidence_items
        else float("nan"),
        "mean_tv": float(np.mean([d.mean_tv for d in distribution_items]))
        if distribution_items
        else float("nan"),
        "mean_js": float(np.mean([d.mean_js for d in distribution_items]))
        if distribution_items
        else float("nan"),
        "max_js": float(np.max([d.max_js for d in distribution_items]))
        if distribution_items
        else float("nan"),
    }


def _stratify(
    label_items: Sequence[ItemLabelStability],
    confidence_items: Sequence[ItemConfidenceStability],
    distribution_items: Sequence[ItemDistributionStability],
    key_fn,
) -> dict[str, dict[str, Any]]:
    lab_g: dict[str, list[ItemLabelStability]] = defaultdict(list)
    conf_g: dict[str, list[ItemConfidenceStability]] = defaultdict(list)
    dist_g: dict[str, list[ItemDistributionStability]] = defaultdict(list)
    for it in label_items:
        lab_g[str(key_fn(it))].append(it)
    for it in confidence_items:
        conf_g[str(key_fn(it))].append(it)
    for it in distribution_items:
        dist_g[str(key_fn(it))].append(it)
    keys = sorted(set(lab_g) | set(conf_g) | set(dist_g))
    return {
        k: _subset_summary(lab_g.get(k, []), conf_g.get(k, []), dist_g.get(k, []))
        for k in keys
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------


def analyze_stability(
    observations: Sequence[RepeatObservation],
    *,
    decision_threshold: float = 0.5,
    seed: int = 0,
    client: str | None = None,
) -> StabilityReport:
    """Compute the three stability metrics + boundary analysis + strata."""
    if not observations:
        raise ValueError("observations must be non-empty")
    if client is not None:
        observations = [o for o in observations if o.client == client]
        if not observations:
            raise ValueError(f"no observations for client={client!r}")
    resolved_client = client or observations[0].client

    labels = label_stability_per_item(observations)
    confs = confidence_stability_per_item(observations)
    dists = distribution_stability_per_item(observations)
    fr = flip_rate(labels, seed=seed)
    boundary = boundary_analysis(labels, confs, threshold=decision_threshold)
    models = detect_model_version_changes(observations)

    n_repeats = max(len(v) for v in _group_by_item(observations).values())

    # Boundary decile stratification
    conf_by = {c.item_id: c for c in confs}
    dist_vals = np.asarray(
        [boundary_distance(conf_by[l.item_id].mean, decision_threshold) for l in labels if l.item_id in conf_by]
    )
    id_order = [l.item_id for l in labels if l.item_id in conf_by]
    by_decile: dict[str, dict[str, Any]] = {}
    if len(dist_vals) >= 2:
        qs = np.quantile(dist_vals, np.linspace(0, 1, 11))
        for i in range(1, len(qs)):
            if qs[i] <= qs[i - 1]:
                qs[i] = qs[i - 1] + 1e-12
        for d in range(10):
            lo, hi = qs[d], qs[d + 1]
            if d < 9:
                mask = (dist_vals >= lo) & (dist_vals < hi)
            else:
                mask = (dist_vals >= lo) & (dist_vals <= hi)
            ids = {id_order[j] for j, m in enumerate(mask) if m}
            by_decile[f"decile_{d}"] = _subset_summary(
                [x for x in labels if x.item_id in ids],
                [x for x in confs if x.item_id in ids],
                [x for x in dists if x.item_id in ids],
                seed=seed + d,
            )

    return StabilityReport(
        n_items=len(labels),
        n_repeats=n_repeats,
        client=resolved_client,
        label={
            "flip_rate": fr.to_dict(),
            "n_items_flipped": int(sum(1 for x in labels if x.flips)),
            "per_item": [asdict(x) for x in labels],
        },
        confidence={
            "mean_sd": float(np.mean([c.sd for c in confs])),
            "mean_range": float(np.mean([c.range for c in confs])),
            "mean_max_abs_dev": float(np.mean([c.max_abs_dev_from_mean for c in confs])),
            "per_item": [asdict(x) for x in confs],
        },
        distribution={
            "mean_of_mean_tv": float(np.mean([d.mean_tv for d in dists])),
            "mean_of_max_tv": float(np.mean([d.max_tv for d in dists])),
            "mean_of_mean_js": float(np.mean([d.mean_js for d in dists])),
            "mean_of_max_js": float(np.mean([d.max_js for d in dists])),
            "js_bound_nats": JS_MAX,
            "per_item": [asdict(x) for x in dists],
        },
        boundary=boundary,
        model_versions=models,
        by_tier=_stratify(labels, confs, dists, lambda it: it.tier),
        by_serving_path=_stratify(labels, confs, dists, lambda it: it.serving_path),
        by_resolved_model=_stratify(
            labels,
            confs,
            dists,
            lambda it: ",".join(sorted(set(it.resolved_models))),
        ),
        by_boundary_decile=by_decile,
    )


def compare_clients_flip_rate(
    jev_labels: Sequence[ItemLabelStability],
    baseline_labels: Sequence[ItemLabelStability],
    *,
    seed: int = 0,
    n_boot: int = 10_000,
) -> BootstrapCI:
    """Paired Δ flip-rate (Jev − baseline) on the same item ids."""
    j = {x.item_id: (1.0 if x.flips else 0.0) for x in jev_labels}
    b = {x.item_id: (1.0 if x.flips else 0.0) for x in baseline_labels}
    common = sorted(set(j) & set(b))
    if not common:
        return BootstrapCI(
            point=float("nan"), low=float("nan"), high=float("nan"), n_resamples=n_boot
        )
    ja = np.asarray([j[i] for i in common])
    ba = np.asarray([b[i] for i in common])
    return paired_bootstrap(
        ja,
        ba,
        statistic=lambda a, b_: float(a.mean() - b_.mean()),
        n=n_boot,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Fixture builders (tests / offline demos)
# ---------------------------------------------------------------------------


def build_identical_repeats(
    *,
    n_items: int = 20,
    n_repeats: int = 10,
    n_classes: int = 3,
    confidence: float = 0.9,
    model: str = "jev-1.13.0",
) -> list[RepeatObservation]:
    """Byte-stable panel: zero flips, zero divergences."""
    obs: list[RepeatObservation] = []
    probs = (confidence,) + tuple((1.0 - confidence) / (n_classes - 1) for _ in range(n_classes - 1))
    for i in range(n_items):
        for r in range(n_repeats):
            obs.append(
                RepeatObservation(
                    item_id=f"item-{i:03d}",
                    repeat=r,
                    decision=0,
                    probabilities=probs,
                    confidence=confidence,
                    tier=["trivial", "easy", "hard", "ambiguous"][i % 4],
                    serving_path="native",
                    resolved_model=model,
                    wall_ms=float(10 + r),
                    client="jev",
                )
            )
    return obs


def build_flipped_fixture(
    *,
    n_items: int = 20,
    n_repeats: int = 10,
    flip_fraction: float = 0.5,
) -> list[RepeatObservation]:
    """Exactly ``flip_fraction`` of items alternate decisions across repeats."""
    n_flip = int(round(n_items * flip_fraction))
    obs: list[RepeatObservation] = []
    for i in range(n_items):
        flips = i < n_flip
        for r in range(n_repeats):
            if flips:
                decision = r % 2  # alternates
                probs = (0.55, 0.45) if decision == 0 else (0.45, 0.55)
                conf = 0.55  # near boundary
            else:
                decision = 0
                probs = (0.95, 0.05)
                conf = 0.95
            obs.append(
                RepeatObservation(
                    item_id=f"item-{i:03d}",
                    repeat=r,
                    decision=decision,
                    probabilities=probs,
                    confidence=conf,
                    tier="hard" if flips else "trivial",
                    serving_path="native",
                    resolved_model="jev-1.13.0",
                    client="jev",
                )
            )
    return obs


def build_planted_boundary_fixture(
    *,
    n_near: int = 40,
    n_far: int = 40,
    n_repeats: int = 10,
) -> list[RepeatObservation]:
    """Near-boundary items flip; far items do not; confidence values stable."""
    obs: list[RepeatObservation] = []
    # Near: conf≈0.52, flips
    for i in range(n_near):
        for r in range(n_repeats):
            decision = r % 2
            probs = (0.52, 0.48) if decision == 0 else (0.48, 0.52)
            obs.append(
                RepeatObservation(
                    item_id=f"near-{i:03d}",
                    repeat=r,
                    decision=decision,
                    probabilities=probs,
                    confidence=0.52,  # stable sharpness
                    tier="ambiguous",
                    serving_path="native",
                    resolved_model="jev-1.13.0",
                    client="jev",
                )
            )
    # Far: conf≈0.95, no flips
    for i in range(n_far):
        for r in range(n_repeats):
            obs.append(
                RepeatObservation(
                    item_id=f"far-{i:03d}",
                    repeat=r,
                    decision=0,
                    probabilities=(0.95, 0.05),
                    confidence=0.95,
                    tier="trivial",
                    serving_path="native",
                    resolved_model="jev-1.13.0",
                    client="jev",
                )
            )
    return obs
