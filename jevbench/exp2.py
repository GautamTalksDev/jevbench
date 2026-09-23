"""EXP-2 decomposition analysis.

=============================================================================
HONEST CLAIM — read before changing framing in results / paper / video
=============================================================================
\"Jev plus N labelled examples outperforms Jev alone AND outperforms a
zero-shot frontier LLM, at a fraction of the cost.\"

Arm A (Jev direct) is ZERO-SHOT. Arm B (decomposed → logistic) is SUPERVISED.
They are not comparable systems. Presenting \"B beats A\" alone as \"Jev is
better than it looks\" is a misrepresentation.

Comparable pairs:
  A vs C  — zero-shot (Jev direct vs LLM direct)
  B vs D  — supervised (Jev+logistic vs LLM+logistic, same N labels)

Both pairs go in the paper. Reporting only B vs C is the rigged version.
=============================================================================

Leakage controls: nested CV (outer eval / inner C), ECE on outer held-out
folds ONLY, stratify by tier AND label, fixed seed, fold-level variance.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from jevbench.cost import resolve_price
from jevbench.metrics import (
    expected_calibration_error,
    paired_bootstrap,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

LABEL_ORDER = ("billing", "technical", "other")
SIGNAL_KEYS = (
    "signal_billing",
    "signal_technical",
    "signal_other",
    "signal_multi_issue",
    "signal_urgency_framing",
)
N_SWEEP = (25, 50, 100, 200)
C_GRID = (0.01, 0.1, 1.0, 10.0)
PRICE_SNAPSHOT = "2026-09-19"
SEED = 20260922


# ---------------------------------------------------------------------------
# Framing helpers (appear in every results payload)
# ---------------------------------------------------------------------------

FAIRNESS_NOTE = (
    "Arm A is zero-shot; Arm B is supervised (uses labels). Comparable pairs: "
    "A vs C (zero-shot) and B vs D (supervised, same N). Never headline B vs C alone."
)

HONEST_CLAIM = (
    "Jev plus N labelled examples outperforms Jev alone AND outperforms a "
    "zero-shot frontier LLM, at a fraction of the cost."
)


@dataclass(frozen=True)
class ItemRow:
    item_id: str
    tier: str
    label: str
    # client → question_key → probability / value
    features: dict[str, dict[str, float]]
    # client → verdict choice + full prob dict
    verdict_choice: dict[str, str]
    verdict_probs: dict[str, dict[str, float]]
    latency_ms: dict[str, float]
    input_tokens: dict[str, int]
    output_tokens: dict[str, int]
    cost_usd: dict[str, float]


def _as_prob_dict(probs: Any) -> dict[str, float]:
    if not isinstance(probs, dict):
        return {}
    return {str(k): float(v) for k, v in probs.items()}


def _noul_value(d: dict[str, Any]) -> float:
    """Extract a [0,1] signal from a decision row."""
    if d.get("probabilities") and "true" in d["probabilities"]:
        return float(d["probabilities"]["true"])
    v = d.get("value")
    if isinstance(v, (int, float)):
        return float(v)
    # choice-as-signal fallback
    probs = _as_prob_dict(d.get("probabilities"))
    if probs:
        return float(max(probs.values()))
    return 0.0


def load_exp2_rows(
    raw_path: Path,
    *,
    labels_by_id: dict[str, str] | None = None,
    tiers_by_id: dict[str, str] | None = None,
) -> list[ItemRow]:
    """Parse raw.jsonl (live or flattened) into per-item Arm feature rows."""
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Group live runner rows: (item_id, client, pass) → decisions
    # Or flattened fixture: one row per (item, client) with single probs
    by_item: dict[str, dict[str, Any]] = {}

    for rec in records:
        iid = str(rec.get("item_id") or rec.get("id"))
        client = str(rec.get("client"))
        if not iid or not client:
            continue
        slot = by_item.setdefault(
            iid,
            {
                "tier": rec.get("tier"),
                "label": rec.get("label"),
                "clients": {},
            },
        )
        if rec.get("tier"):
            slot["tier"] = rec["tier"]
        if rec.get("label"):
            slot["label"] = rec["label"]

        cslot = slot["clients"].setdefault(
            client,
            {
                "signals": {},
                "verdict_choice": None,
                "verdict_probs": {},
                "latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
            },
        )

        if "decisions" in rec:
            # Prefer lowest pass
            pass_idx = int(rec.get("pass", 0))
            if "_pass" in cslot and pass_idx > cslot["_pass"]:
                continue
            cslot["_pass"] = pass_idx
            cslot["latency_ms"] = 0.0
            cslot["signals"] = {}
            for d in rec.get("decisions") or []:
                qk = str(d.get("question_key") or d.get("question_id") or "")
                cslot["latency_ms"] = max(
                    cslot["latency_ms"], float(d.get("latency_ms") or 0.0)
                )
                if qk == "verdict" or qk == "department":
                    probs = _as_prob_dict(d.get("probabilities"))
                    cslot["verdict_probs"] = probs
                    choice = d.get("value") or d.get("choice") or d.get("answer")
                    if choice is not None:
                        cslot["verdict_choice"] = str(choice)
                elif qk.startswith("signal_"):
                    cslot["signals"][qk] = _noul_value(d)
            usage = rec.get("usage") or {}
            cslot["input_tokens"] = int(usage.get("input_tokens") or 0)
            cslot["output_tokens"] = int(usage.get("output_tokens") or 0)
            cslot["cost_usd"] = float(rec.get("est_cost_usd") or 0.0)
        else:
            # Flattened fixture: treat probabilities as verdict
            probs = _as_prob_dict(rec.get("probabilities"))
            if probs:
                cslot["verdict_probs"] = probs
                cslot["verdict_choice"] = str(
                    rec.get("choice") or max(probs, key=probs.get)
                )
            cslot["latency_ms"] = float(rec.get("latency_ms") or 0.0)
            cslot["input_tokens"] = int(rec.get("input_tokens") or 0)
            cslot["output_tokens"] = int(rec.get("output_tokens") or 0)
            # Synthesize signals from verdict probs when decomposed absent
            # (offline fixture only — live runs must have real signals)
            if probs and not cslot["signals"]:
                cslot["signals"] = {
                    "signal_billing": float(probs.get("billing", 0.0)),
                    "signal_technical": float(probs.get("technical", 0.0)),
                    "signal_other": float(probs.get("other", 0.0)),
                    "signal_multi_issue": 0.5,
                    "signal_urgency_framing": 0.5,
                }
                cslot["_synthetic_signals"] = True

    rows: list[ItemRow] = []
    for iid, slot in sorted(by_item.items()):
        label = slot.get("label") or (labels_by_id or {}).get(iid)
        tier = slot.get("tier") or (tiers_by_id or {}).get(iid)
        if label is None or tier is None:
            continue
        features: dict[str, dict[str, float]] = {}
        verdict_choice: dict[str, str] = {}
        verdict_probs: dict[str, dict[str, float]] = {}
        latency: dict[str, float] = {}
        tin: dict[str, int] = {}
        tout: dict[str, int] = {}
        cost: dict[str, float] = {}
        for client, cslot in slot["clients"].items():
            features[client] = dict(cslot.get("signals") or {})
            if cslot.get("verdict_choice"):
                verdict_choice[client] = str(cslot["verdict_choice"])
            verdict_probs[client] = dict(cslot.get("verdict_probs") or {})
            latency[client] = float(cslot.get("latency_ms") or 0.0)
            tin[client] = int(cslot.get("input_tokens") or 0)
            tout[client] = int(cslot.get("output_tokens") or 0)
            cost[client] = float(cslot.get("cost_usd") or 0.0)
        rows.append(
            ItemRow(
                item_id=iid,
                tier=str(tier),
                label=str(label),
                features=features,
                verdict_choice=verdict_choice,
                verdict_probs=verdict_probs,
                latency_ms=latency,
                input_tokens=tin,
                output_tokens=tout,
                cost_usd=cost,
            )
        )
    return rows


def _feature_matrix(
    rows: list[ItemRow],
    client: str,
    signal_keys: tuple[str, ...] = SIGNAL_KEYS,
) -> np.ndarray:
    X = np.zeros((len(rows), len(signal_keys)), dtype=float)
    for i, row in enumerate(rows):
        feats = row.features.get(client) or {}
        for j, key in enumerate(signal_keys):
            X[i, j] = float(feats.get(key, 0.0))
    return X


def _stratification_keys(rows: list[ItemRow]) -> np.ndarray:
    """tier|label keys for stratified folds (no degenerate folds)."""
    return np.asarray([f"{r.tier}|{r.label}" for r in rows])


def _latency_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _multiclass_auroc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """OvR macro AUROC; NaN if a class is missing in y_true."""
    try:
        return float(
            roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")


def _ece_from_proba(
    proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10
) -> dict[str, Any]:
    """Top-label ECE on probability matrix vs integer labels."""
    ece = expected_calibration_error(proba, y_true, n_bins=n_bins, strategy="uniform")
    return {
        "ece": ece.ece,
        "n": ece.n,
        "n_bins": ece.n_bins,
        "bin_counts": list(ece.bin_counts),
        "bin_mean_confidence": list(ece.bin_mean_confidence),
        "bin_accuracy": list(ece.bin_accuracy),
        "max_bin_fraction": ece.max_bin_fraction,
    }


def _zero_shot_preds(
    rows: list[ItemRow], client: str, label_encoder: LabelEncoder
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_hat indices, proba matrix) from verdict head."""
    y_hat = []
    probas = []
    classes = list(label_encoder.classes_)
    for row in rows:
        choice = row.verdict_choice.get(client)
        probs = row.verdict_probs.get(client) or {}
        if choice is None and probs:
            choice = max(probs, key=probs.get)
        if choice is None:
            choice = classes[0]
        if choice not in classes:
            # unseen choice → map to first class
            choice = classes[0]
        y_hat.append(label_encoder.transform([choice])[0])
        vec = np.array([float(probs.get(c, 0.0)) for c in classes], dtype=float)
        s = vec.sum()
        if s <= 0:
            vec = np.ones(len(classes)) / len(classes)
        else:
            vec = vec / s
        probas.append(vec)
    return np.asarray(y_hat, dtype=int), np.asarray(probas, dtype=float)


def _select_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strat_train: np.ndarray,
    *,
    inner_folds: int,
    seed: int,
) -> float:
    """Inner CV over C_GRID — never peek at outer test."""
    # Need enough strata for inner folds
    n_splits = min(inner_folds, _max_folds(strat_train))
    if n_splits < 2:
        return 1.0
    best_c, best_score = 1.0, -1.0
    for C in C_GRID:
        scores: list[float] = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        try:
            splits = list(skf.split(X_train, strat_train))
        except ValueError:
            return 1.0
        for tr, va in splits:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            if len(np.unique(y_train[tr])) < 2:
                continue
            clf.fit(X_train[tr], y_train[tr])
            scores.append(float(accuracy_score(y_train[va], clf.predict(X_train[va]))))
        if scores and float(np.mean(scores)) > best_score:
            best_score = float(np.mean(scores))
            best_c = float(C)
    return best_c


def _max_folds(strat: np.ndarray) -> int:
    _, counts = np.unique(strat, return_counts=True)
    return int(counts.min()) if len(counts) else 0


def nested_cv_supervised(
    rows: list[ItemRow],
    client: str,
    *,
    n_train_cap: int | None = None,
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = SEED,
    signal_keys: tuple[str, ...] = SIGNAL_KEYS,
) -> dict[str, Any]:
    """Fit logistic on decomposed signals with nested CV.

    Predictions / ECE use OUTER held-out folds only. In-sample ECE is never
    computed.
    """
    le = LabelEncoder()
    y = le.fit_transform([r.label for r in rows])
    X = _feature_matrix(rows, client, signal_keys)
    strat = _stratification_keys(rows)
    ids = [r.item_id for r in rows]

    n_splits = min(outer_folds, _max_folds(strat))
    if n_splits < 2:
        return {
            "error": "insufficient per-(tier|label) cells for stratified outer CV",
            "n": len(rows),
            "min_cell": int(_max_folds(strat)),
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.full(len(rows), -1, dtype=int)
    oof_proba = np.zeros((len(rows), len(le.classes_)), dtype=float)
    fold_acc: list[float] = []
    fold_details: list[dict[str, Any]] = []
    chosen_Cs: list[float] = []

    rng = np.random.default_rng(seed)

    for fold_i, (tr, te) in enumerate(skf.split(X, strat)):
        tr = np.asarray(tr)
        te = np.asarray(te)
        if n_train_cap is not None and len(tr) > n_train_cap:
            # Subsample train to N, preserving stratification as far as possible
            tr = _stratified_subsample(tr, strat[tr], n_train_cap, rng)

        C = _select_C(
            X[tr], y[tr], strat[tr], inner_folds=inner_folds, seed=seed + fold_i
        )
        chosen_Cs.append(C)
        if len(np.unique(y[tr])) < 2:
            # Degenerate train — fall back to majority
            maj = int(np.bincount(y[tr]).argmax())
            oof_pred[te] = maj
            oof_proba[te, maj] = 1.0
            acc = float(accuracy_score(y[te], oof_pred[te]))
        else:
            clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
            clf.fit(X[tr], y[tr])
            oof_pred[te] = clf.predict(X[te])
            proba = clf.predict_proba(X[te])
            # Align columns to le.classes_
            full = np.zeros((len(te), len(le.classes_)))
            for j, cls in enumerate(clf.classes_):
                full[:, int(cls)] = proba[:, j]
            oof_proba[te] = full
            acc = float(accuracy_score(y[te], oof_pred[te]))
        fold_acc.append(acc)
        fold_details.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "C": C,
                "accuracy": acc,
                "test_ids": [ids[i] for i in te.tolist()],
            }
        )

    mask = oof_pred >= 0
    y_m, pred_m, proba_m = y[mask], oof_pred[mask], oof_proba[mask]
    metrics = _classification_bundle(y_m, pred_m, proba_m, le)
    # ECE on outer held-out only
    metrics["ece_outer_held_out"] = _ece_from_proba(proba_m, y_m)
    metrics["ece_note"] = (
        "ECE measured on concatenated outer held-out predictions ONLY. "
        "In-sample ECE for a fitted combiner is meaningless and is not reported."
    )
    return {
        "client": client,
        "mode": "supervised_nested_cv",
        "n": len(rows),
        "n_train_cap": n_train_cap,
        "n_outer_folds": n_splits,
        "fold_accuracy": fold_acc,
        "fold_accuracy_mean": float(np.mean(fold_acc)),
        "fold_accuracy_std": float(np.std(fold_acc, ddof=1)) if len(fold_acc) > 1 else 0.0,
        "chosen_C_per_fold": chosen_Cs,
        "folds": fold_details,
        "metrics": metrics,
        "classes": list(le.classes_),
        "oof_pred_labels": le.inverse_transform(pred_m).tolist(),
        "oof_true_labels": le.inverse_transform(y_m).tolist(),
        "oof_item_ids": [ids[i] for i, m in enumerate(mask) if m],
        "oof_correct": (pred_m == y_m).astype(float).tolist(),
    }


def _stratified_subsample(
    indices: np.ndarray,
    strat: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(indices) <= n:
        return indices
    # Proportional allocation per stratum, at least 1 when possible
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, key in zip(indices.tolist(), strat.tolist()):
        groups[str(key)].append(idx)
    keys = sorted(groups)
    alloc = {k: 0 for k in keys}
    remaining = n
    # Round-robin until filled
    while remaining > 0:
        progressed = False
        for k in keys:
            if remaining == 0:
                break
            if alloc[k] < len(groups[k]):
                alloc[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    chosen: list[int] = []
    for k in keys:
        pool = list(groups[k])
        rng.shuffle(pool)
        chosen.extend(pool[: alloc[k]])
    return np.asarray(chosen, dtype=int)


def _classification_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    le: LabelEncoder,
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "auroc_ovr_macro": _multiclass_auroc(y_true, proba),
        "n": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
    }


def evaluate_zero_shot_arm(
    rows: list[ItemRow],
    client: str,
    *,
    seed: int = SEED,
) -> dict[str, Any]:
    le = LabelEncoder()
    y = le.fit_transform([r.label for r in rows])
    pred, proba = _zero_shot_preds(rows, client, le)
    metrics = _classification_bundle(y, pred, proba, le)
    metrics["ece"] = _ece_from_proba(proba, y)
    lat = [r.latency_ms.get(client, float("nan")) for r in rows]
    cost = sum(r.cost_usd.get(client, 0.0) for r in rows)
    n_correct = metrics["n_correct"]
    return {
        "client": client,
        "mode": "zero_shot",
        "metrics": metrics,
        "latency": _latency_percentiles([x for x in lat if x == x]),
        "total_cost_usd": cost,
        "cost_per_1000_correct": (
            (cost / n_correct) * 1000.0 if n_correct > 0 else float("nan")
        ),
        "classes": list(le.classes_),
        "correct": (pred == y).astype(float).tolist(),
        "item_ids": [r.item_id for r in rows],
        "seed": seed,
    }


def paired_vs_arm_a(
    correct_a: list[float],
    correct_other: list[float],
    *,
    seed: int,
    n_boot: int = 10_000,
) -> dict[str, Any]:
    """Paired bootstrap on accuracy difference (same items)."""
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_other, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired arms must share item alignment")
    ci = paired_bootstrap(
        a,
        b,
        statistic=lambda x, y: float(np.mean(x) - np.mean(y)),
        n=n_boot,
        seed=seed,
    )
    return {
        **ci.to_dict(),
        "interpretation": (
            "Δacc = arm − Arm A; CI excludes 0 ⇒ distinguishable; "
            "crossing 0 ⇒ inconclusive (not equivalence)."
        ),
    }


def run_label_sweep(
    rows: list[ItemRow],
    *,
    client: str = "jev",
    Ns: tuple[int, ...] = N_SWEEP,
    seed: int = SEED,
) -> dict[str, Any]:
    """How many labels does Arm B need? Accuracy vs N on outer folds."""
    points: list[dict[str, Any]] = []
    for N in Ns:
        if N > len(rows):
            points.append(
                {
                    "N": N,
                    "skipped": True,
                    "reason": f"N={N} > n_items={len(rows)}",
                }
            )
            continue
        res = nested_cv_supervised(
            rows, client, n_train_cap=N, seed=seed + N
        )
        if "error" in res:
            points.append({"N": N, "skipped": True, "reason": res["error"]})
            continue
        points.append(
            {
                "N": N,
                "skipped": False,
                "accuracy_mean": res["fold_accuracy_mean"],
                "accuracy_std": res["fold_accuracy_std"],
                "fold_accuracy": res["fold_accuracy"],
                "ece_outer": res["metrics"]["ece_outer_held_out"]["ece"],
            }
        )
    return {
        "client": client,
        "arm": "B",
        "Ns": list(Ns),
        "points": points,
        "note": (
            "Held-out outer-fold accuracy vs label count. "
            "'It takes N labels to beat a frontier model' is the practitioner number."
        ),
    }


def estimate_arm_cost(
    rows: list[ItemRow],
    client: str,
    *,
    client_type: str,
    model: str | None,
) -> dict[str, float]:
    """Sum measured costs, or re-price from tokens if est_cost missing."""
    total = sum(r.cost_usd.get(client, 0.0) for r in rows)
    tin = sum(r.input_tokens.get(client, 0) for r in rows)
    tout = sum(r.output_tokens.get(client, 0) for r in rows)
    if total <= 0 and (tin or tout):
        try:
            price = resolve_price(
                PRICE_SNAPSHOT,
                client_type,
                model if client_type == "adapter" else None,
            )
            total = price.cost_usd(
                tin, 0.0 if client_type == "jev" else float(tout)
            )
        except Exception:
            pass
    return {
        "total_usd": float(total),
        "input_tokens": float(tin),
        "output_tokens": float(tout),
    }


def render_label_sweep_chart(
    sweep: dict[str, Any],
    *,
    arm_c_accuracy: float | None,
    out_dir: Path,
) -> dict[str, Path]:
    from jevbench.charts import make_context

    ctx = make_context(
        run_id="exp2-label-sweep",
        resolved_model="jev-1.13.0+logistic",
        out_dir=out_dir,
        extra_footer="EXP-2 · Arm B nested-CV accuracy vs N labels",
    )
    fig = ctx.new_figure()
    ax = fig.add_subplot(111)
    pts = [p for p in sweep["points"] if not p.get("skipped")]
    if not pts:
        ax.text(0.5, 0.5, "no sweep points", ha="center", transform=ax.transAxes)
    else:
        ns = [p["N"] for p in pts]
        means = [p["accuracy_mean"] for p in pts]
        stds = [p["accuracy_std"] for p in pts]
        ax.errorbar(
            ns,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.5,
            markersize=10,
            color=ctx.style.series("jev")["color"],
            label="Arm B (Jev signals → logistic)",
        )
        if arm_c_accuracy is not None and arm_c_accuracy == arm_c_accuracy:
            ax.axhline(
                arm_c_accuracy,
                color=ctx.style.series("adapter")["color"],
                linestyle="--",
                linewidth=2.0,
                label=f"Arm C zero-shot LLM ({arm_c_accuracy:.2f})",
            )
    ax.set_xlabel("N labelled training examples (per outer fold cap)")
    ax.set_ylabel("Held-out outer-fold accuracy")
    ax.set_title("How many labels does Arm B need?")
    ax.set_ylim(0.0, 1.02)
    ax.legend(loc="best", frameon=True)
    fig.subplots_adjust(bottom=0.14, top=0.9, left=0.1, right=0.96)
    return ctx.save(fig, "exp2_label_sweep")


def run_exp2_analysis(
    *,
    raw_path: Path,
    out_path: Path | None = None,
    labels_by_id: dict[str, str] | None = None,
    tiers_by_id: dict[str, str] | None = None,
    seed: int = SEED,
    n_boot: int = 10_000,
    write_chart: bool = True,
    jev_client: str = "jev",
    llm_client: str = "adapter_baseline",
    trivial_client: str = "trivial",
) -> dict[str, Any]:
    """Full EXP-2 analysis → results/exp2.json."""
    rows = load_exp2_rows(
        raw_path, labels_by_id=labels_by_id, tiers_by_id=tiers_by_id
    )
    # Fixture naming
    clients_present = set()
    for r in rows:
        clients_present |= set(r.verdict_choice) | set(r.features)
    if llm_client not in clients_present and "adapter" in clients_present:
        llm_client = "adapter"

    if len(rows) < 10:
        raise RuntimeError(f"EXP-2 needs more items; got {len(rows)}")

    # Align client aliases for trivial
    if trivial_client not in clients_present:
        trivial_client = "trivial" if "trivial" in clients_present else trivial_client

    arm_a = evaluate_zero_shot_arm(rows, jev_client, seed=seed)
    arm_c = evaluate_zero_shot_arm(rows, llm_client, seed=seed + 1)
    arm_e = (
        evaluate_zero_shot_arm(rows, trivial_client, seed=seed + 2)
        if trivial_client in clients_present
        else {"error": f"client {trivial_client!r} absent", "correct": []}
    )

    arm_b = nested_cv_supervised(rows, jev_client, seed=seed)
    arm_d = nested_cv_supervised(rows, llm_client, seed=seed + 10)

    # Costs for zero-shot arms (supervised combiner adds no API cost)
    for arm, client, ctype, model in (
        (arm_a, jev_client, "jev", None),
        (arm_c, llm_client, "adapter", "openai/gpt-4o-mini"),
        (arm_e if "metrics" in arm_e else None, trivial_client, "trivial", None),
    ):
        if arm is None or "metrics" not in arm:
            continue
        priced = estimate_arm_cost(rows, client, client_type=ctype, model=model)
        arm["total_cost_usd"] = priced["total_usd"]
        n_correct = arm["metrics"]["n_correct"]
        arm["cost_per_1000_correct"] = (
            (priced["total_usd"] / n_correct) * 1000.0 if n_correct else float("nan")
        )

    # Supervised arms inherit API cost of their feature-source client
    if "metrics" in arm_b:
        priced = estimate_arm_cost(rows, jev_client, client_type="jev", model=None)
        arm_b["total_cost_usd"] = priced["total_usd"]
        n_correct = arm_b["metrics"]["n_correct"]
        arm_b["cost_per_1000_correct"] = (
            (priced["total_usd"] / n_correct) * 1000.0 if n_correct else float("nan")
        )
        arm_b["latency"] = _latency_percentiles(
            [r.latency_ms.get(jev_client, float("nan")) for r in rows]
        )
    if "metrics" in arm_d:
        priced = estimate_arm_cost(
            rows, llm_client, client_type="adapter", model="openai/gpt-4o-mini"
        )
        arm_d["total_cost_usd"] = priced["total_usd"]
        n_correct = arm_d["metrics"]["n_correct"]
        arm_d["cost_per_1000_correct"] = (
            (priced["total_usd"] / n_correct) * 1000.0 if n_correct else float("nan")
        )
        arm_d["latency"] = _latency_percentiles(
            [r.latency_ms.get(llm_client, float("nan")) for r in rows]
        )

    # Paired CIs vs Arm A (align on item order of arm_a)
    comparisons: dict[str, Any] = {}
    correct_a = arm_a["correct"]
    for name, arm in (("C", arm_c), ("E", arm_e)):
        if not arm.get("correct"):
            continue
        comparisons[f"{name}_minus_A"] = paired_vs_arm_a(
            correct_a, arm["correct"], seed=seed + hash(name) % 1000, n_boot=n_boot
        )
    # B and D oof correct — align to arm_a item ids
    id_to_idx = {iid: i for i, iid in enumerate(arm_a["item_ids"])}
    for name, arm in (("B", arm_b), ("D", arm_d)):
        if "oof_item_ids" not in arm:
            continue
        correct_other = [float("nan")] * len(correct_a)
        for iid, c in zip(arm["oof_item_ids"], arm["oof_correct"]):
            if iid in id_to_idx:
                correct_other[id_to_idx[iid]] = float(c)
        # Drop nan pairs
        paired_a, paired_o = [], []
        for ca, co in zip(correct_a, correct_other):
            if co == co:
                paired_a.append(ca)
                paired_o.append(co)
        comparisons[f"{name}_minus_A"] = paired_vs_arm_a(
            paired_a, paired_o, seed=seed + hash(name) % 1000, n_boot=n_boot
        )

    # B vs D at full nested CV (same N = all train per fold)
    if "oof_correct" in arm_b and "oof_correct" in arm_d:
        # Align on intersection of oof ids
        b_map = dict(zip(arm_b["oof_item_ids"], arm_b["oof_correct"]))
        d_map = dict(zip(arm_d["oof_item_ids"], arm_d["oof_correct"]))
        common = sorted(set(b_map) & set(d_map))
        comparisons["B_minus_D"] = paired_vs_arm_a(
            [b_map[i] for i in common],
            [d_map[i] for i in common],
            seed=seed + 99,
            n_boot=n_boot,
        )

    sweep = run_label_sweep(rows, client=jev_client, seed=seed)
    # Also sweep D for fairness
    sweep_d = run_label_sweep(rows, client=llm_client, seed=seed + 7)
    sweep_d["arm"] = "D"

    chart_paths: dict[str, str] = {}
    if write_chart:
        paths = render_label_sweep_chart(
            sweep,
            arm_c_accuracy=(arm_c.get("metrics") or {}).get("accuracy"),
            out_dir=RESULTS,
        )
        for k, pth in paths.items():
            try:
                chart_paths[k] = str(pth.relative_to(REPO))
            except ValueError:
                chart_paths[k] = str(pth)
        stable = RESULTS / "exp2_label_sweep.png"
        if "1080p" in paths:
            stable.write_bytes(paths["1080p"].read_bytes())
            chart_paths["png"] = str(stable.relative_to(REPO))

    # Practitioner number: smallest N where B mean acc > C acc
    c_acc = (arm_c.get("metrics") or {}).get("accuracy")
    beat_n = None
    if c_acc is not None:
        for p in sweep["points"]:
            if not p.get("skipped") and p["accuracy_mean"] > c_acc:
                beat_n = p["N"]
                break

    verdict = _build_verdict(arm_a, arm_b, arm_c, arm_d, arm_e, comparisons, beat_n)

    payload: dict[str, Any] = {
        "schema": "jevbench.exp2.v1",
        "honest_claim": HONEST_CLAIM,
        "fairness_note": FAIRNESS_NOTE,
        "comparable_pairs": {
            "zero_shot": "A vs C",
            "supervised": "B vs D",
            "forbidden_headline": "B vs C alone",
        },
        "source": {"raw": str(raw_path), "n_items": len(rows), "seed": seed},
        "arms": {
            "A": {**arm_a, "name": "jev_direct", "shot": "zero_shot"},
            "B": {**arm_b, "name": "jev_decomposed_logistic", "shot": "supervised"},
            "C": {**arm_c, "name": "llm_direct", "shot": "zero_shot"},
            "D": {**arm_d, "name": "llm_logistic", "shot": "supervised"},
            "E": {**arm_e, "name": "non_ai_floor", "shot": "zero_shot"},
        },
        "comparisons_paired_vs_A": comparisons,
        "label_sweep_B": sweep,
        "label_sweep_D": sweep_d,
        "labels_to_beat_zero_shot_llm": beat_n,
        "charts": chart_paths,
        "verdict": verdict,
        "leakage_controls": {
            "nested_cv": True,
            "ece_on": "outer_held_out_only",
            "stratify_by": ["tier", "label"],
            "seed": seed,
            "question_set_frozen": True,
        },
    }

    out_path = out_path or (RESULTS / "exp2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky oof arrays from disk? Keep them — auditability.
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _build_verdict(
    arm_a: dict[str, Any],
    arm_b: dict[str, Any],
    arm_c: dict[str, Any],
    arm_d: dict[str, Any],
    arm_e: dict[str, Any],
    comparisons: dict[str, Any],
    beat_n: int | None,
) -> str:
    def acc(arm: dict[str, Any]) -> str:
        m = arm.get("metrics") or {}
        if "accuracy" in m:
            return f"{m['accuracy']:.3f}"
        if "fold_accuracy_mean" in arm:
            return f"{arm['fold_accuracy_mean']:.3f}"
        return "n/a"

    b_mean = arm_b.get("fold_accuracy_mean", (arm_b.get("metrics") or {}).get("accuracy"))
    parts = [
        f"Honest claim under test: {HONEST_CLAIM}",
        f"Zero-shot (A vs C): Arm A accuracy={acc(arm_a)}, Arm C={acc(arm_c)}.",
        f"Supervised (B vs D): Arm B outer-fold mean={b_mean}, Arm D="
        f"{arm_d.get('fold_accuracy_mean', 'n/a')}.",
        f"Non-AI floor E={acc(arm_e)}.",
    ]
    if beat_n is not None:
        parts.append(
            f"Label efficiency: Arm B exceeds zero-shot LLM (Arm C) by N={beat_n} "
            f"labels on the sweep."
        )
    else:
        parts.append(
            "Label efficiency: no N in {25,50,100,200} beat Arm C on this corpus "
            "(or corpus too small) — report the curve anyway."
        )
    parts.append(
        "ECE for B/D is outer-held-out only; in-sample combiner ECE is not reported. "
        + FAIRNESS_NOTE
    )
    return " ".join(parts)
