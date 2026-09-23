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
    paired_bootstrap,
    soft_correctness,
    subsample_to_equal_n,
)

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


def _flatten_live_record(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a runner raw.jsonl row into fixture-style flat record."""
    decisions = rec.get("decisions") or []
    if not decisions:
        return None
    # Prefer the department / first choice decision
    d = decisions[0]
    for cand in decisions:
        q = str(cand.get("question_id") or cand.get("name") or "")
        if "department" in q or cand.get("kind") == "choice":
            d = cand
            break
    if d.get("error"):
        return None
    probs = d.get("probabilities")
    if not isinstance(probs, dict):
        return None
    choice = d.get("choice") or d.get("answer")
    # Ground-truth label is not on live records — caller joins dataset
    return {
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
    }


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
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    sel = [r for r in rows if r.get("tier") in tiers and r.get("role", "primary") != "paraphrase"]
    if not sel:
        return np.zeros((0, len(label_order))), np.zeros(0, dtype=int), [], None
    probs = _prob_matrix([r["probabilities"] for r in sel], label_order)
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
) -> dict[str, Any]:
    """Primary ΔECE + descriptive calibration + M-sweep for one client.

    When every item carries a ``label_dist``, soft scoring is primary and
    hard scoring is reported beside it. Hard scoring puts label noise in
    the hard stratum only, which inflates ΔECE in the direction of the
    hypothesis — see :func:`jevbench.metrics.hard_correctness`.
    """
    items = [
        r for r in _client_pass0(rows, client) if r.get("role", "primary") != "paraphrase"
    ]
    label_order = _label_order_for(items)
    ph, yh, ids_h, dh = _stratum_arrays(items, HARD_TIERS, label_order)
    pe, ye, ids_e, de = _stratum_arrays(items, EASY_TIERS, label_order)

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
        "n_items": len(items),
        "n_hard": int(len(yh)),
        "n_easy": int(len(ye)),
        "equal_n": equal_n_info,
        "scoring_primary": scoring_primary,
        "primary_delta_ece": primary_block,
        "scoring": {
            "primary": scoring_primary,
            "soft": soft_block,
            "hard": hard_block,
            "hard_scoring_note": (
                "Hard scoring puts label noise in the hard stratum only, which "
                "inflates ΔECE in the direction of the hypothesis."
            ),
        },
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


def build_verdict(payload: dict[str, Any]) -> str:
    """One-paragraph plain-language verdict."""
    jev = payload.get("jev") or {}
    base = payload.get("baseline") or {}
    primary = (jev.get("primary_delta_ece") or {})
    gate = payload.get("gates") or {}

    if gate.get("block_reason"):
        return (
            f"EXP-1 is blocked before a substantive claim: {gate['block_reason']} "
            f"No hypothesis is supported until the gate clears."
        )

    point = primary.get("point")
    bca = primary.get("ci_bca") or [None, None]
    crosses = primary.get("crosses_zero_bca", True)
    flips = (jev.get("robustness_m_sweep") or {}).get("conclusion_flips_with_M")

    if crosses:
        hyp = (
            "The data are inconclusive for H1 (ΔECE CI crosses zero under BCa); "
            "neither 'calibration independent of difficulty' nor 'hard is less "
            "calibrated' is established at the powered endpoint"
        )
    elif bca[0] is not None and bca[0] > 0:
        hyp = (
            "The data support H1 in the literature direction: the hard stratum "
            "is less calibrated than the easy stratum (ΔECE BCa CI excludes zero from above)"
        )
    else:
        hyp = (
            "The data support a ΔECE distinguishable from zero, but not in the "
            "pre-specified hard-worse direction (CI excludes zero from below)"
        )

    base_line = ""
    bp = (base.get("primary_delta_ece") or {})
    if bp:
        base_line = (
            f" On the adapter baseline under the same items and equal-n rule, "
            f"ΔECE={bp.get('point')} (BCa {bp.get('ci_bca')})."
        )

    flip_line = (
        " The M-binning sweep flips the primary sign — treat the conclusion as "
        "binning-dependent."
        if flips
        else " The M∈{5,10,15,20} sweep does not flip the primary sign."
    )

    rule = jev.get("scoring_primary", "hard")
    return (
        f"{hyp} (primary scoring={rule}, point ΔECE={point}, BCa {bca}, percentile "
        f"{primary.get('ci_percentile')}, n_hard={jev.get('n_hard')}, "
        f"n_easy={jev.get('n_easy')}).{flip_line}{base_line} "
        f"Hard scoring is reported beside soft scoring when annotator distributions "
        f"are present. The four-tier ECE-vs-accuracy slope remains descriptive only."
    )


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

    jev = analyze_client(rows, client=jev_client, seed=seed, n_boot=n_boot)
    baseline = analyze_client(
        rows, client=baseline_client, seed=seed + 1, n_boot=n_boot
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
        "baseline": baseline,
        "jev_vs_baseline": equal_acc_note,
        "strata": {"hard": list(HARD_TIERS), "easy": list(EASY_TIERS)},
        "contamination": contamination_table(
            rows,
            arms=contamination_arms
            or sorted({str(r.get("client")) for r in rows if r.get("client")}),
        ),
    }
    payload["verdict"] = build_verdict(payload)

    out_path = out_path or (RESULTS / "exp1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload
