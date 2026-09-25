"""EXP-1 analysis: ΔECE primary + descriptive secondaries + M-sweep.

Reads a run ``raw.jsonl`` (live) or the offline fixture flattened format.
Writes ``results/exp1.json`` and a one-paragraph plain-language verdict.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jevbench.chaosnli import LABEL_ORDER as NLI_LABEL_ORDER
from jevbench.chaosnli import contamination_table
from jevbench.metrics import (
    calibration_by_tier,
    delta_ece,
    expected_calibration_error,
    hard_correctness,
    jensen_shannon_divergence,
    paired_bootstrap,
    soft_correctness,
    subsample_to_equal_n,
    total_variation_distance,
    uniform_baseline_divergence,
)
from jevbench.power import SOFT_NULL_OPERATING_KAPPA, soft_delta_ece_corrected
from jevbench.verdict import decide_verdict, p_holds_one_sided, simulate_corrected_delta_ece

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

HARD_TIERS = ("hard", "ambiguous")
EASY_TIERS = ("trivial", "easy")
TIER_ORDER = ("trivial", "easy", "hard", "ambiguous")
LABEL_ORDER = ("billing", "technical", "other")
N_BOOT = 10_000
M_SWEEP = (5, 10, 15, 20)
STRATEGIES = ("uniform", "quantile")


def _prob_matrix(
    prob_dicts: list[dict[str, float]], label_order: tuple[str, ...] = LABEL_ORDER
) -> np.ndarray:
    rows = []
    for d in prob_dicts:
        rows.append([float(d.get(lab, 0.0)) for lab in label_order])
    return np.asarray(rows, dtype=float)


def _label_index(label: str, label_order: tuple[str, ...] = LABEL_ORDER) -> int:
    if label not in label_order:
        raise ValueError(f"unknown label {label!r}; expected one of {label_order}")
    return label_order.index(label)


def _label_order_for(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    labels = {str(r.get("label")) for r in rows if r.get("label") is not None}
    if labels and labels <= set(NLI_LABEL_ORDER):
        return NLI_LABEL_ORDER
    return LABEL_ORDER


def load_raw_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


CHAOSNLI_LABEL_ORDER = ("entailment", "neutral", "contradiction")
NOUL_QUESTION_KEYS = (
    "noul_entailment",
    "noul_neutral",
    "noul_contradiction",
)
# Near-zero sum rule (PROMPT R): if Σ noul < floor, use uniform and flag.
# Defined before any scored call; pilot counts how often it triggers.
NOUL_SUM_FLOOR = 1e-6


def _noul_probs_from_decisions(
    decisions: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Normalize three Noul yes-probabilities into a 3-way distribution.

    Thin wrapper around :func:`normalize_noul_triplet` that returns only the
    probability map (or None). Use ``normalize_noul_triplet`` when the
    near-zero-sum flag is needed.
    """
    result = normalize_noul_triplet(decisions)
    if result is None:
        return None
    return result["probabilities"]


def normalize_noul_triplet(
    decisions: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return ``{probabilities, near_zero_sum, raw_sum, raw}`` or None if incomplete."""
    raw: dict[str, float] = {}
    for d in decisions:
        if d.get("error"):
            continue
        key = str(d.get("question_id") or d.get("question_key") or d.get("name") or "")
        if key not in NOUL_QUESTION_KEYS:
            continue
        # Noul value is P(statement true); parse stores it as ``value``.
        # Never invent a confidence field — Noul has none.
        val = d.get("value")
        if val is None:
            val = (d.get("raw") or {}).get("noul")
        if val is None:
            return None
        label = key.replace("noul_", "", 1)
        raw[label] = float(val)
    if len(raw) != 3:
        return None
    total = sum(raw.values())
    near_zero = total < NOUL_SUM_FLOOR
    if near_zero:
        probs = {lab: 1.0 / 3.0 for lab in CHAOSNLI_LABEL_ORDER}
    else:
        probs = {lab: raw[lab] / total for lab in CHAOSNLI_LABEL_ORDER}
    return {
        "probabilities": probs,
        "near_zero_sum": near_zero,
        "raw_sum": total,
        "raw": {lab: raw[lab] for lab in CHAOSNLI_LABEL_ORDER},
        "floor": NOUL_SUM_FLOOR,
    }


def _flatten_live_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a runner raw.jsonl row into fixture-style flat record.

    Amendment 9: when three Nouls are present in the same call, also attach
    ``probabilities_noul`` (normalized). Choice remains ``probabilities``.
    """
    decisions = rec.get("decisions") or []
    if not decisions:
        return None
    # Prefer the Choice decision for the primary probabilities field.
    d = decisions[0]
    for cand in decisions:
        q = str(cand.get("question_id") or cand.get("name") or "")
        if q == "relation" or (
            cand.get("kind") == "choice" and "noul" not in q
        ):
            d = cand
            break
    if d.get("error"):
        return None
    probs = d.get("probabilities")
    if not isinstance(probs, dict):
        return None
    choice = d.get("choice") or d.get("answer") or d.get("value")
    noul_pack = normalize_noul_triplet(decisions)
    out = {
        "item_id": rec["item_id"],
        "tier": rec["tier"],
        "client": rec["client"],
        "pass": rec.get("pass", 0),
        "probabilities": probs,
        "choice": choice,
        "model": d.get("resolved_model") or rec.get("model_requested"),
        "serving_path": rec.get("serving_path"),
        "latency_ms": d.get("latency_ms"),
        "input_tokens": (rec.get("usage") or {}).get("input_tokens"),
        "output_tokens": (rec.get("usage") or {}).get("output_tokens"),
        "role": rec.get("role", "primary"),
        "source_item_id": rec.get("source_item_id"),
        "primitive_arm": "choice",
    }
    if noul_pack is not None:
        out["probabilities_noul"] = noul_pack["probabilities"]
        out["choice_noul"] = max(
            noul_pack["probabilities"], key=noul_pack["probabilities"].get
        )
        out["noul_near_zero_sum"] = bool(noul_pack["near_zero_sum"])
        out["noul_raw_sum"] = float(noul_pack["raw_sum"])
    return out


def normalize_records(
    raw: list[dict[str, Any]],
    *,
    labels_by_id: dict[str, str] | None = None,
    label_dist_by_id: dict[str, list[float]] | None = None,
) -> list[dict[str, Any]]:
    """Return fixture-style rows with label / probabilities / tier / client."""
    out: list[dict[str, Any]] = []
    for rec in raw:
        if "probabilities" in rec and "label" in rec:
            row = dict(rec)
            out.append(row)
            continue
        flat = _flatten_live_record(rec)
        if flat is None:
            continue
        iid = str(flat["item_id"])
        if labels_by_id is not None:
            if iid not in labels_by_id:
                continue
            flat["label"] = labels_by_id[iid]
        elif "label" not in flat:
            continue
        if label_dist_by_id is not None and iid in label_dist_by_id:
            flat["label_dist"] = list(label_dist_by_id[iid])
        out.append(flat)
    return out


def _client_pass0(
    records: list[dict[str, Any]], client: str
) -> list[dict[str, Any]]:
    """One row per item: prefer pass==0, else lowest pass, else first."""
    by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if r.get("client") != client:
            continue
        by_item[str(r["item_id"])].append(r)
    chosen: list[dict[str, Any]] = []
    for iid, rows in by_item.items():
        rows_sorted = sorted(rows, key=lambda x: int(x.get("pass", 0)))
        # Prefer pass 0 / 1
        pref = [x for x in rows_sorted if int(x.get("pass", 0)) in (0, 1)]
        chosen.append(pref[0] if pref else rows_sorted[0])
    chosen.sort(key=lambda r: r["item_id"])
    return chosen


def _stratum_arrays(
    rows: list[dict[str, Any]],
    tiers: tuple[str, ...],
    label_order: tuple[str, ...],
    *,
    probs_key: str = "probabilities",
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    sel = [
        r
        for r in rows
        if r.get("tier") in tiers
        and r.get("role", "primary") != "paraphrase"
        and isinstance(r.get(probs_key), dict)
    ]
    if not sel:
        return np.zeros((0, len(label_order))), np.zeros(0, dtype=int), [], None
    probs = _prob_matrix([r[probs_key] for r in sel], label_order)
    labels = np.asarray(
        [_label_index(str(r["label"]), label_order) for r in sel], dtype=int
    )
    ids = [str(r["item_id"]) for r in sel]
    if all(isinstance(r.get("label_dist"), (list, tuple)) for r in sel):
        dists = np.asarray([r["label_dist"] for r in sel], dtype=float)
    else:
        dists = None
    return probs, labels, ids, dists


def _align_kept(
    ids_all: list[str], ids_kept: list[str], *arrays: np.ndarray
) -> list[np.ndarray]:
    pos = {i: k for k, i in enumerate(ids_all)}
    idx = np.asarray([pos[i] for i in ids_kept], dtype=int)
    return [arr[idx] for arr in arrays]


def _holds_effect_reference(*, seed: int = 20260924) -> np.ndarray:
    """Corrected ΔECE draws under true ΔECE=0.09 for the holds one-sided test."""
    cache = RESULTS / "holds_effect_null.json"
    if cache.is_file():
        doc = json.loads(cache.read_text(encoding="utf-8"))
        arr = np.asarray(doc.get("corrected") or [], dtype=float)
        if arr.size >= 50:
            return arr
    sim = simulate_corrected_delta_ece(
        setting="alternative",
        n_trials=400,
        n_boot=400,
        n_e0=400,
        n_null_pval=400,
        seed=seed,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {
                "schema": "jevbench.holds_effect_null.v1",
                "n_trials": int(sim["corrected"].size),
                "mean_corrected": sim["mean_corrected"],
                "corrected": sim["corrected"].tolist(),
                "jev_data_observed": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return sim["corrected"]


def _descriptive_direction_cell(
    probs: np.ndarray,
    correct: np.ndarray,
    labels: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Per-stratum direction summary on the exact rows used for ECE."""
    conf = np.asarray(probs, dtype=float).max(axis=1)
    correct_arr = np.asarray(correct, dtype=float)
    mean_conf = float(conf.mean()) if len(conf) else float("nan")
    mean_correct = float(correct_arr.mean()) if len(correct_arr) else float("nan")
    ece = expected_calibration_error(
        probs,
        labels,
        n_bins=n_bins,
        strategy="uniform",
        correctness=correct_arr,
    )
    bins: list[dict[str, Any]] = []
    for b in range(n_bins):
        count = int(ece.bin_counts[b])
        bins.append(
            {
                "count": count,
                "mean_conf": (
                    float("nan") if count == 0 else float(ece.bin_mean_confidence[b])
                ),
                "mean_correct": (
                    float("nan") if count == 0 else float(ece.bin_accuracy[b])
                ),
            }
        )
    return {
        "n": int(len(conf)),
        "mean_top_label_confidence": mean_conf,
        "mean_correctness": mean_correct,
        "overconfidence": float(mean_conf - mean_correct),
        "bins": bins,
    }


def _descriptive_direction_block(
    ph: np.ndarray,
    yh: np.ndarray,
    pe: np.ndarray,
    ye: np.ndarray,
    *,
    correct_h: np.ndarray | None,
    correct_e: np.ndarray | None,
) -> dict[str, Any]:
    """Arm × stratum × scoring direction tables (additive; not used for verdict)."""
    hard_h = hard_correctness(ph, yh)
    hard_e = hard_correctness(pe, ye)
    out: dict[str, Any] = {
        "note": (
            "Descriptive only — same rows/pass/items as the ECE endpoint. "
            "overconfidence = mean_top_label_confidence − mean_correctness. "
            "Bins are the 10 uniform confidence bins used for ECE."
        ),
        "hard": {
            "hard": _descriptive_direction_cell(ph, hard_h, yh),
            "soft": None,
        },
        "easy": {
            "hard": _descriptive_direction_cell(pe, hard_e, ye),
            "soft": None,
        },
    }
    if correct_h is not None and correct_e is not None:
        out["hard"]["soft"] = _descriptive_direction_cell(ph, correct_h, yh)
        out["easy"]["soft"] = _descriptive_direction_cell(pe, correct_e, ye)
    return out


def _p_value_display(p_value: float, *, n_null_pval: int, n_null_extreme: int) -> str:
    """Report Monte Carlo p; use p < 1/(N+1) when no null draw matches the extreme."""
    if int(n_null_extreme) == 0:
        return f"p < 1/{int(n_null_pval) + 1}"
    return f"p = {float(p_value):.6g}"


def _corrected_soft_primary(
    ph: np.ndarray,
    correct_h: np.ndarray,
    pe: np.ndarray,
    correct_e: np.ndarray,
    *,
    seed: int,
    kappa: float = SOFT_NULL_OPERATING_KAPPA,
    n_boot: int = 2000,
    n_e0: int = 2000,
    n_null_pval: int = 2000,
) -> dict[str, Any]:
    """Raw + corrected soft ΔECE and parametric p-values (Amendment 8–9)."""
    rng = np.random.default_rng(seed)
    top_h = ph.max(axis=1)
    top_e = pe.max(axis=1)
    iv = soft_delta_ece_corrected(
        top_h,
        correct_h,
        top_e,
        correct_e,
        kappa=kappa,
        n_boot=n_boot,
        n_e0=n_e0,
        n_null_pval=n_null_pval,
        rng=rng,
    )
    effect_ref = _holds_effect_reference(seed=seed + 5)
    p_hold = p_holds_one_sided(float(iv["corrected_delta_ece"]), effect_ref)
    n_null = int(iv.get("n_null_pval", n_null_pval))
    n_extreme = int(iv.get("n_null_extreme", -1))
    return {
        "delta_raw": float(iv["raw_delta_ece"]),
        "delta_corrected": float(iv["corrected_delta_ece"]),
        "p_value": float(iv["p_value"]),
        "p_holds_reject_ge_effect": float(p_hold),
        "e0_hard": float(iv["e0_hard"]),
        "e0_easy": float(iv["e0_easy"]),
        "bias0_delta_ece": float(iv["bias0_delta_ece"]),
        "ci_low_corrected": float(iv["ci_low"]),
        "ci_high_corrected": float(iv["ci_high"]),
        "kappa": float(kappa),
        # Additive — existing numeric fields above unchanged.
        "n_null_pval": n_null,
        "n_null_extreme": n_extreme,
        "p_value_display": _p_value_display(
            float(iv["p_value"]), n_null_pval=n_null, n_null_extreme=n_extreme
        ),
    }


def _divergence_secondary(
    ph: np.ndarray,
    dh: np.ndarray,
    pe: np.ndarray,
    de: np.ndarray,
) -> dict[str, Any]:
    """JSD/TVD to humans per stratum, vs uniform-guess baseline (Amendment 9)."""
    jsd_h = jensen_shannon_divergence(ph, dh)
    jsd_e = jensen_shannon_divergence(pe, de)
    tvd_h = total_variation_distance(ph, dh)
    tvd_e = total_variation_distance(pe, de)
    uni_h = uniform_baseline_divergence(dh)
    uni_e = uniform_baseline_divergence(de)
    return {
        "hard": {
            "jsd_mean": float(jsd_h.mean()),
            "tvd_mean": float(tvd_h.mean()),
            "n": int(len(jsd_h)),
        },
        "easy": {
            "jsd_mean": float(jsd_e.mean()),
            "tvd_mean": float(tvd_e.mean()),
            "n": int(len(jsd_e)),
        },
        "uniform_baseline": {"hard": uni_h, "easy": uni_e},
        "note": (
            "Secondary only. Compare Jev→human JSD/TVD to the uniform-guess "
            "baseline. Motivated by Baan et al. (EMNLP 2022 / arXiv 2210.16133)."
        ),
    }


def _delta_block(
    ph: np.ndarray,
    yh: np.ndarray,
    pe: np.ndarray,
    ye: np.ndarray,
    *,
    seed: int,
    n_boot: int,
    correct_h: np.ndarray | None = None,
    correct_e: np.ndarray | None = None,
) -> dict[str, Any]:
    primary = delta_ece(
        ph, yh, pe, ye,
        seed=seed, n_boot=n_boot, n_bins=10, strategy="uniform", ci_method="bca",
        correctness_hard=correct_h, correctness_easy=correct_e,
    )
    primary_pct = delta_ece(
        ph, yh, pe, ye,
        seed=seed, n_boot=n_boot, n_bins=10, strategy="uniform", ci_method="percentile",
        correctness_hard=correct_h, correctness_easy=correct_e,
    )
    return {
        "point": primary.point,
        "ci_method_primary": "bca",
        "ci_bca": [primary.ci_bca_low, primary.ci_bca_high],
        "ci_percentile": [primary_pct.ci_percentile_low, primary_pct.ci_percentile_high],
        "bca_z0": primary.bca_z0,
        "bca_acceleration": primary.bca_acceleration,
        "crosses_zero_bca": primary.ci_bca_low < 0 < primary.ci_bca_high,
        "crosses_zero_percentile": (
            primary_pct.ci_percentile_low < 0 < primary_pct.ci_percentile_high
        ),
        "n_boot": n_boot,
        "n_bins": 10,
        "strategy": "uniform",
        "equal_n": primary.equal_n,
        "ece_hard": primary.ece_hard.ece,
        "ece_easy": primary.ece_easy.ece,
    }


def analyze_client(
    rows: list[dict[str, Any]],
    *,
    client: str,
    seed: int = 20260922,
    n_boot: int = N_BOOT,
    primitive_arm: str = "choice",
) -> dict[str, Any]:
    """Primary ΔECE + descriptive calibration + M-sweep for one client.

    When every item carries a ``label_dist``, soft scoring is primary and
    hard scoring is reported beside it. Hard scoring puts label noise in
    the hard stratum only, which inflates ΔECE in the direction of the
    hypothesis — see :func:`jevbench.metrics.hard_correctness`.

    ``primitive_arm`` is ``choice`` (default) or ``noul`` (normalized three
    Nouls from the same call — Amendment 9). Neither arm is "the" result.
    """
    probs_key = "probabilities_noul" if primitive_arm == "noul" else "probabilities"
    items = [
        r for r in _client_pass0(rows, client) if r.get("role", "primary") != "paraphrase"
    ]
    if primitive_arm == "noul":
        items = [r for r in items if isinstance(r.get("probabilities_noul"), dict)]
    label_order = _label_order_for(items)
    ph, yh, ids_h, dh = _stratum_arrays(
        items, HARD_TIERS, label_order, probs_key=probs_key
    )
    pe, ye, ids_e, de = _stratum_arrays(
        items, EASY_TIERS, label_order, probs_key=probs_key
    )

    equal_n_info: dict[str, Any]
    if len(yh) == 0 or len(ye) == 0:
        return {
            "client": client,
            "error": "empty stratum",
            "n_hard_raw": int(len(yh)),
            "n_easy_raw": int(len(ye)),
        }

    if len(yh) != len(ye):
        sub = subsample_to_equal_n(
            ph, yh, ids_h, pe, ye, ids_e, seed=seed
        )
        ids_h_kept, ids_e_kept = list(sub.item_ids_a), list(sub.item_ids_b)
        ph, yh = _align_kept(ids_h, ids_h_kept, ph, yh)
        pe, ye = _align_kept(ids_e, ids_e_kept, pe, ye)
        if dh is not None:
            dh = _align_kept(ids_h, ids_h_kept, dh)[0]
        if de is not None:
            de = _align_kept(ids_e, ids_e_kept, de)[0]
        ids_h, ids_e = ids_h_kept, ids_e_kept
        equal_n_info = {
            "applied": True,
            "target_n": sub.target_n,
            "discarded_item_ids": list(sub.discarded_item_ids),
        }
    else:
        equal_n_info = {"applied": False, "target_n": int(len(yh)), "discarded_item_ids": []}

    hard_block = _delta_block(ph, yh, pe, ye, seed=seed, n_boot=n_boot)
    soft_block = None
    correct_h = correct_e = None
    if dh is not None and de is not None:
        correct_h = soft_correctness(ph, dh)
        correct_e = soft_correctness(pe, de)
        soft_block = _delta_block(
            ph, yh, pe, ye,
            seed=seed, n_boot=n_boot, correct_h=correct_h, correct_e=correct_e,
        )
    scoring_primary = "soft" if soft_block is not None else "hard"
    primary_block = soft_block if soft_block is not None else hard_block
    direction = _descriptive_direction_block(
        ph, yh, pe, ye, correct_h=correct_h, correct_e=correct_e
    )

    # Amendment 9 — parametric corrected primary + divergence secondaries.
    corrected_block: dict[str, Any] | None = None
    verdict_block: dict[str, Any] | None = None
    divergence_block: dict[str, Any] | None = None
    if soft_block is not None and correct_h is not None and correct_e is not None:
        corrected_block = _corrected_soft_primary(
            ph, correct_h, pe, correct_e, seed=seed + 91
        )
        verdict_block = decide_verdict(
            corrected_delta_ece=float(corrected_block["delta_corrected"]),
            p_value=float(corrected_block["p_value"]),
            p_holds_reject_ge_effect=float(
                corrected_block["p_holds_reject_ge_effect"]
            ),
        )
        if dh is not None and de is not None:
            divergence_block = _divergence_secondary(ph, dh, pe, de)

    # Four-tier descriptive (support-ticket tiers). ChaosNLI has only easy/hard.
    by_tier_p: dict[str, list] = {t: [] for t in TIER_ORDER}
    by_tier_y: dict[str, list] = {t: [] for t in TIER_ORDER}
    for r in items:
        t = r["tier"]
        if t not in by_tier_p:
            continue
        by_tier_p[t].append(r["probabilities"])
        by_tier_y[t].append(_label_index(str(r["label"]), label_order))
    probs_by = {
        t: _prob_matrix(by_tier_p[t], label_order) for t in TIER_ORDER if by_tier_p[t]
    }
    labels_by = {
        t: np.asarray(by_tier_y[t], dtype=int) for t in TIER_ORDER if by_tier_y[t]
    }
    cal = None
    if set(probs_by) == set(TIER_ORDER):
        cal = calibration_by_tier(probs_by, labels_by, tier_order=TIER_ORDER)

    # M-sweep robustness
    sweep: list[dict[str, Any]] = []
    for m in M_SWEEP:
        for strat in STRATEGIES:
            res = delta_ece(
                ph,
                yh,
                pe,
                ye,
                seed=seed + m,
                n_boot=min(n_boot, 2_000) if n_boot > 2000 else n_boot,
                n_bins=m,
                strategy=strat,  # type: ignore[arg-type]
                ci_method="percentile",
                correctness_hard=correct_h,
                correctness_easy=correct_e,
            )
            sweep.append(
                {
                    "n_bins": m,
                    "strategy": strat,
                    "point": res.point,
                    "ci_low": res.ci_low,
                    "ci_high": res.ci_high,
                    "excludes_zero": not res.crosses_zero(),
                    "sign": (
                        "positive"
                        if res.ci_low > 0
                        else "negative"
                        if res.ci_high < 0
                        else "inconclusive"
                    ),
                }
            )

    signs = {s["sign"] for s in sweep}
    conclusion_flips = len(signs - {"inconclusive"}) > 1 or (
        "inconclusive" in signs and len(signs) > 1 and any(
            s in signs for s in ("positive", "negative")
        )
    )
    # Stricter: flip if some exclude positive and some exclude negative
    has_pos = any(s["sign"] == "positive" for s in sweep)
    has_neg = any(s["sign"] == "negative" for s in sweep)
    conclusion_flips = has_pos and has_neg

    per_tier = {}
    if cal is not None:
        for tc in cal.tiers:
            per_tier[tc.tier] = {
                "n": tc.n,
                "accuracy": tc.accuracy,
                "brier": tc.brier,
                "ece_uniform": tc.ece_uniform.to_dict(),
                "ece_quantile": tc.ece_quantile.to_dict(),
            }

    return {
        "client": client,
        "primitive_arm": primitive_arm,
        "n_items": len(items),
        "n_hard": int(len(yh)),
        "n_easy": int(len(ye)),
        "equal_n": equal_n_info,
        "scoring_primary": scoring_primary,
        "primary_delta_ece": primary_block,
        "delta_raw": (
            None if corrected_block is None else corrected_block["delta_raw"]
        ),
        "delta_corrected": (
            None if corrected_block is None else corrected_block["delta_corrected"]
        ),
        "p_value": None if corrected_block is None else corrected_block["p_value"],
        "n_null_pval": (
            None if corrected_block is None else corrected_block.get("n_null_pval")
        ),
        "p_value_display": (
            None if corrected_block is None else corrected_block.get("p_value_display")
        ),
        "p_holds_reject_ge_effect": (
            None
            if corrected_block is None
            else corrected_block["p_holds_reject_ge_effect"]
        ),
        "verdict": None if verdict_block is None else verdict_block["verdict"],
        "verdict_detail": verdict_block,
        "corrected_soft": corrected_block,
        "divergence_to_humans": divergence_block,
        "scoring": {
            "primary": scoring_primary,
            "soft": soft_block,
            "hard": hard_block,
            "hard_scoring_note": (
                "Hard scoring puts label noise in the hard stratum only, which "
                "inflates ΔECE in the direction of the hypothesis."
            ),
        },
        "descriptive_direction": direction,
        "descriptive_calibration_by_tier": (
            {
                "tier_names": list(cal.tier_names),
                "accuracies": list(cal.accuracies),
                "ece_uniform": list(cal.ece_uniform),
                "ece_quantile": list(cal.ece_quantile),
                "slope_uniform": cal.slope_uniform,
                "slope_quantile": cal.slope_quantile,
                "note": "DESCRIPTIVE ONLY — two residual df; not the powered endpoint.",
                "by_tier": per_tier,
            }
            if cal is not None
            else None
        ),
        "robustness_m_sweep": {
            "cells": sweep,
            "conclusion_flips_with_M": conclusion_flips,
            "note": (
                "If conclusion_flips_with_M is true, the conclusion IS the binning."
                if conclusion_flips
                else "Primary sign stable across M∈{5,10,15,20} and both strategies."
            ),
        },
    }


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return float("nan")
    correct = [
        1.0 if str(r.get("choice")) == str(r.get("label")) else 0.0 for r in rows
    ]
    return float(np.mean(correct))


def no_verdict_reason(payload: dict[str, Any]) -> str:
    """Why Amendment 9 cannot emit tracks/holds/inconclusive yet."""
    jev = payload.get("jev") or {}
    gate = payload.get("gates") or {}
    if gate.get("block_reason"):
        return f"gate blocked: {gate['block_reason']}"
    if jev.get("delta_corrected") is None or jev.get("p_value") is None:
        return (
            "delta_corrected or p_value is None — Amendment 9 verdict requires "
            "corrected soft ΔECE (ChaosNLI label_dist / soft scoring); "
            "refusing to emit 'inconclusive' as a stand-in"
        )
    if jev.get("verdict") is None:
        return "choice-arm verdict label is None"
    return "verdict unavailable"


def build_verdict(payload: dict[str, Any]) -> str | None:
    """Plain-language Amendment 9 verdict, or None when metrics are missing.

    Never substitutes ``\"inconclusive\"`` when ``delta_corrected`` or
    ``p_value`` is None — that label is only valid after the parametric test.
    """
    jev = payload.get("jev") or {}
    jev_noul = payload.get("jev_noul") or {}
    gate = payload.get("gates") or {}

    if gate.get("block_reason"):
        return (
            f"EXP-1 is blocked before a substantive claim: {gate['block_reason']} "
            f"No hypothesis is supported until the gate clears."
        )

    if jev.get("delta_corrected") is None or jev.get("p_value") is None:
        return None
    label = jev.get("verdict")
    if label is None:
        return None
    detail = jev.get("verdict_detail") or {}
    noul_label = jev_noul.get("verdict")
    parts = [
        f"Choice arm Amendment 9 verdict: {label}.",
        detail.get("reason") or "",
    ]
    if noul_label:
        parts.append(
            f"Noul arm (same call, normalized) verdict: {noul_label} "
            f"(corrected ΔECE={jev_noul.get('delta_corrected')}, "
            f"p={jev_noul.get('p_value')}). Neither arm is 'the' result."
        )
    flips = (jev.get("robustness_m_sweep") or {}).get("conclusion_flips_with_M")
    if flips:
        parts.append(
            "The M-binning sweep flips the raw-interval sign — treat raw "
            "binning claims as fragile; the parametric corrected test is primary."
        )
    return " ".join(p for p in parts if p)


def run_exp1_analysis(
    *,
    raw_path: Path,
    out_path: Path | None = None,
    labels_by_id: dict[str, str] | None = None,
    label_dist_by_id: dict[str, list[float]] | None = None,
    jev_client: str = "jev",
    baseline_client: str = "adapter_baseline",
    seed: int = 20260922,
    n_boot: int = N_BOOT,
    agreement: dict[str, Any] | None = None,
    power: dict[str, Any] | None = None,
    scored: bool = False,
    labelling_ceiling_applies: bool = True,
    contamination_arms: list[str] | None = None,
) -> dict[str, Any]:
    raw = load_raw_records(raw_path)
    rows = normalize_records(
        raw, labels_by_id=labels_by_id, label_dist_by_id=label_dist_by_id
    )
    # Fixture uses client name "adapter"
    clients = {r.get("client") for r in rows}
    if baseline_client not in clients and "adapter" in clients:
        baseline_client = "adapter"
    if jev_client not in clients and "trivial" in clients and "jev" not in clients:
        # offline fixture may lack jev — analyze trivial as stand-in only for harness
        pass

    gates: dict[str, Any] = {"scored": scored, "block_reason": None}
    if agreement is not None:
        gates["agreement"] = agreement
        if agreement.get("n_paired", 0) > 0 and not agreement.get("passes_floor", False):
            gates["block_reason"] = (
                f"Cohen's kappa={agreement.get('kappa')} < "
                f"{agreement.get('kappa_floor', 0.6)} on the double-labelled "
                "subset — tighten LABEL_GUIDE.md and re-label."
            )
    if power is not None:
        gates["power"] = {
            "chosen_n_per_stratum": power.get("chosen_n_per_stratum"),
            "underpowered": power.get("underpowered"),
        }
        if power.get("underpowered") and gates["block_reason"] is None and scored:
            note = str(power.get("underpowered_note") or "")
            if (not labelling_ceiling_applies) and "labelling ceiling" in note:
                gates["labelling_ceiling_applies"] = False
                gates["labelling_ceiling_note"] = (
                    "The hand-labelling ceiling does not apply: ChaosNLI already "
                    "supplies 100 annotations per item."
                )
            else:
                gates["block_reason"] = (
                    note or "Study is underpowered relative to F1 pessimistic corner."
                )

    jev = analyze_client(
        rows, client=jev_client, seed=seed, n_boot=n_boot, primitive_arm="choice"
    )
    jev_noul = analyze_client(
        rows, client=jev_client, seed=seed + 3, n_boot=n_boot, primitive_arm="noul"
    )
    baseline = analyze_client(
        rows,
        client=baseline_client,
        seed=seed + 1,
        n_boot=n_boot,
        primitive_arm="choice",
    )
    baseline_noul = analyze_client(
        rows,
        client=baseline_client,
        seed=seed + 4,
        n_boot=n_boot,
        primitive_arm="noul",
    )

    # Equal-accuracy comparison note
    jev_rows = _client_pass0(rows, jev_client)
    base_rows = _client_pass0(rows, baseline_client)
    common = sorted({r["item_id"] for r in jev_rows} & {r["item_id"] for r in base_rows})
    j_by = {r["item_id"]: r for r in jev_rows}
    b_by = {r["item_id"]: r for r in base_rows}
    if common:
        j_acc = np.asarray(
            [1.0 if j_by[i].get("choice") == j_by[i].get("label") else 0.0 for i in common]
        )
        b_acc = np.asarray(
            [1.0 if b_by[i].get("choice") == b_by[i].get("label") else 0.0 for i in common]
        )
        acc_delta = paired_bootstrap(
            j_acc,
            b_acc,
            statistic=lambda a, b: float(np.mean(a) - np.mean(b)),
            n=min(n_boot, 10_000),
            seed=seed + 2,
        )
        equal_acc_note = {
            "n_common": len(common),
            "jev_accuracy": float(j_acc.mean()),
            "baseline_accuracy": float(b_acc.mean()),
            "accuracy_delta_jev_minus_baseline": acc_delta.to_dict(),
            "question": (
                "Is Jev better calibrated than a temp-0 LLM baseline on the "
                "same items? Compare primary_delta_ece across clients; "
                "accuracy delta is reported so the reader can see whether "
                "the comparison is at roughly equal accuracy."
            ),
        }
    else:
        equal_acc_note = {"n_common": 0}

    # Provenance
    models = sorted(
        {str(r.get("model")) for r in rows if r.get("model")}
    )
    paths = sorted(
        {str(r.get("serving_path")) for r in rows if r.get("serving_path")}
    )

    payload: dict[str, Any] = {
        "schema": "jevbench.exp1.v1",
        "source": {
            "raw": str(raw_path),
            "scored": scored,
            "resolved_models": models,
            "serving_paths": paths,
            "n_raw_records": len(raw),
            "n_normalized": len(rows),
        },
        "gates": gates,
        "jev": jev,
        "jev_noul": jev_noul,
        "baseline": baseline,
        "baseline_noul": baseline_noul,
        "primitive_arms": {
            "note": (
                "Amendment 9: Choice and normalized three-Noul arms from the "
                "same call. Report both; neither is 'the' result."
            ),
            "choice": "jev",
            "noul": "jev_noul",
        },
        "jev_vs_baseline": equal_acc_note,
        "strata": {"hard": list(HARD_TIERS), "easy": list(EASY_TIERS)},
        "contamination": contamination_table(
            rows,
            arms=contamination_arms
            or sorted({str(r.get("client")) for r in rows if r.get("client")}),
        ),
    }
    payload["verdict"] = build_verdict(payload)
    if payload["verdict"] is None:
        payload["no_verdict_reason"] = no_verdict_reason(payload)
    # Certificate / paper primary stamp is Choice arm unless gate blocks.
    # Never promote a missing corrected test into an Amendment 9 label.
    result_verdict = jev.get("verdict")
    if jev.get("delta_corrected") is None or jev.get("p_value") is None:
        result_verdict = None
    payload["result"] = {
        "delta_raw": jev.get("delta_raw"),
        "delta_corrected": jev.get("delta_corrected"),
        "p_value": jev.get("p_value"),
        "n_null_pval": jev.get("n_null_pval"),
        "p_value_display": jev.get("p_value_display"),
        "verdict": result_verdict,
        "primitive_arm": "choice",
        "noul_verdict": jev_noul.get("verdict"),
        "noul_delta_corrected": jev_noul.get("delta_corrected"),
        "noul_p_value": jev_noul.get("p_value"),
        "noul_n_null_pval": jev_noul.get("n_null_pval"),
        "noul_p_value_display": jev_noul.get("p_value_display"),
    }

    out_path = out_path or (RESULTS / "exp1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
