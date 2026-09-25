"""Post-hoc pass sensitivity for EXP-1 (NOT PREREGISTERED — robustness only).

Computes corrected soft ΔECE for each pass 0..9 and for per-item mean
probabilities across passes. Does not modify primary EXP-1 outputs.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from jevbench.exp1 import (
    EASY_TIERS,
    HARD_TIERS,
    _corrected_soft_primary,
    _label_order_for,
    _prob_matrix,
    _stratum_arrays,
    load_raw_records,
    normalize_records,
)
from jevbench.metrics import soft_correctness, subsample_to_equal_n

PASSES = tuple(range(10))
SENSITIVITY_LABEL = "NOT PREREGISTERED — robustness only"
# Match analyze_client corrected-soft seed offset.
PRIMARY_SEED = 20260922
CORRECTED_SEED = PRIMARY_SEED + 91


def _client_pass(
    rows: list[dict[str, Any]], client: str, pass_idx: int
) -> list[dict[str, Any]]:
    """One row per item at a fixed pass (primary role only)."""
    by_item: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("client") != client:
            continue
        if r.get("role", "primary") == "paraphrase":
            continue
        if int(r.get("pass", 0)) != int(pass_idx):
            continue
        iid = str(r["item_id"])
        by_item[iid] = r
    return [by_item[k] for k in sorted(by_item)]


def _mean_across_passes(
    rows: list[dict[str, Any]],
    client: str,
    *,
    probs_key: str,
    item_ids: set[str],
) -> list[dict[str, Any]]:
    """Per-item mean probability vectors across passes 0..9 (same metadata)."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("client") != client:
            continue
        if r.get("role", "primary") == "paraphrase":
            continue
        iid = str(r["item_id"])
        if iid not in item_ids:
            continue
        if not isinstance(r.get(probs_key), dict):
            continue
        buckets[iid].append(r)
    out: list[dict[str, Any]] = []
    for iid in sorted(item_ids):
        group = buckets.get(iid) or []
        if len(group) != len(PASSES):
            raise RuntimeError(
                f"item {iid!r} has {len(group)} passes for {probs_key}; need {len(PASSES)}"
            )
        group = sorted(group, key=lambda x: int(x.get("pass", 0)))
        label_order = _label_order_for(group)
        mats = _prob_matrix([g[probs_key] for g in group], label_order)
        mean_vec = mats.mean(axis=0)
        mean_probs = {lab: float(mean_vec[i]) for i, lab in enumerate(label_order)}
        base = dict(group[0])
        base[probs_key] = mean_probs
        base["pass"] = "mean"
        out.append(base)
    return out


def _locked_item_ids(
    rows: list[dict[str, Any]],
    client: str,
    *,
    probs_key: str,
    seed: int,
) -> tuple[list[str], list[str], tuple[str, ...]]:
    """Hard/easy item ids after the same equal-n lock as primary (pass 0)."""
    items = _client_pass(rows, client, 0)
    if probs_key != "probabilities":
        items = [r for r in items if isinstance(r.get(probs_key), dict)]
    label_order = _label_order_for(items)
    ph, yh, ids_h, dh = _stratum_arrays(
        items, HARD_TIERS, label_order, probs_key=probs_key
    )
    pe, ye, ids_e, de = _stratum_arrays(
        items, EASY_TIERS, label_order, probs_key=probs_key
    )
    if len(yh) == 0 or len(ye) == 0:
        raise RuntimeError("empty stratum while locking sensitivity item ids")
    if len(yh) != len(ye):
        sub = subsample_to_equal_n(ph, yh, ids_h, pe, ye, ids_e, seed=seed)
        ids_h, ids_e = list(sub.item_ids_a), list(sub.item_ids_b)
    if len(ids_h) != 750 or len(ids_e) != 750:
        raise RuntimeError(
            f"expected 750/750 locked items; got hard={len(ids_h)} easy={len(ids_e)}"
        )
    return ids_h, ids_e, label_order


def _arrays_for_items(
    items: list[dict[str, Any]],
    *,
    ids_h: list[str],
    ids_e: list[str],
    label_order: tuple[str, ...],
    probs_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build stratum arrays restricted to the locked 750/750 ids."""
    by_id = {str(r["item_id"]): r for r in items}
    missing = [i for i in ids_h + ids_e if i not in by_id]
    if missing:
        raise RuntimeError(f"missing locked items in pass slice: {missing[:5]}")
    hard_rows = [by_id[i] for i in ids_h]
    easy_rows = [by_id[i] for i in ids_e]
    ph = _prob_matrix([r[probs_key] for r in hard_rows], label_order)
    pe = _prob_matrix([r[probs_key] for r in easy_rows], label_order)
    if not all(isinstance(r.get("label_dist"), (list, tuple)) for r in hard_rows + easy_rows):
        raise RuntimeError("soft sensitivity requires label_dist on every locked item")
    dh = np.asarray([r["label_dist"] for r in hard_rows], dtype=float)
    de = np.asarray([r["label_dist"] for r in easy_rows], dtype=float)
    return ph, pe, dh, de


def _corrected_for_arrays(
    ph: np.ndarray,
    pe: np.ndarray,
    dh: np.ndarray,
    de: np.ndarray,
    *,
    seed: int,
    n_null_pval: int = 2000,
) -> dict[str, Any]:
    correct_h = soft_correctness(ph, dh)
    correct_e = soft_correctness(pe, de)
    block = _corrected_soft_primary(
        ph,
        correct_h,
        pe,
        correct_e,
        seed=seed,
        n_null_pval=n_null_pval,
    )
    return {
        "delta_raw": block["delta_raw"],
        "delta_corrected": block["delta_corrected"],
        "p_value": block["p_value"],
        "p_value_display": block.get("p_value_display"),
        "n_null_pval": block.get("n_null_pval", n_null_pval),
        "n_hard": int(len(ph)),
        "n_easy": int(len(pe)),
    }


def run_pass_sensitivity(
    *,
    rows: list[dict[str, Any]],
    client: str = "jev",
    seed: int = PRIMARY_SEED,
    n_null_pval: int = 2000,
) -> dict[str, Any]:
    """Return sensitivity payload for choice and noul arms."""
    arms_out: dict[str, Any] = {}
    for arm, probs_key in (("choice", "probabilities"), ("noul", "probabilities_noul")):
        ids_h, ids_e, label_order = _locked_item_ids(
            rows, client, probs_key=probs_key, seed=seed
        )
        by_pass: dict[str, Any] = {}
        values: list[float] = []
        for p in PASSES:
            items = _client_pass(rows, client, p)
            if probs_key != "probabilities":
                items = [r for r in items if isinstance(r.get(probs_key), dict)]
            ph, pe, dh, de = _arrays_for_items(
                items,
                ids_h=ids_h,
                ids_e=ids_e,
                label_order=label_order,
                probs_key=probs_key,
            )
            cell = _corrected_for_arrays(
                ph, pe, dh, de, seed=CORRECTED_SEED, n_null_pval=n_null_pval
            )
            by_pass[str(p)] = cell
            values.append(float(cell["delta_corrected"]))

        mean_items = _mean_across_passes(
            rows,
            client,
            probs_key=probs_key,
            item_ids=set(ids_h) | set(ids_e),
        )
        ph, pe, dh, de = _arrays_for_items(
            mean_items,
            ids_h=ids_h,
            ids_e=ids_e,
            label_order=label_order,
            probs_key=probs_key,
        )
        mean_cell = _corrected_for_arrays(
            ph, pe, dh, de, seed=CORRECTED_SEED, n_null_pval=n_null_pval
        )
        values.append(float(mean_cell["delta_corrected"]))
        arms_out[arm] = {
            "by_pass": by_pass,
            "mean_across_passes": mean_cell,
            "delta_corrected_all": values,
            "delta_corrected_min": float(min(values)),
            "delta_corrected_max": float(max(values)),
            "keys": [*(str(p) for p in PASSES), "mean_across_passes"],
            "locked_n_hard": len(ids_h),
            "locked_n_easy": len(ids_e),
        }
    return arms_out


def build_sensitivity_payload(
    *,
    raw_path: Path,
    run_id: str,
    labels_by_id: dict[str, str],
    label_dist_by_id: dict[str, list[float]],
    client: str = "jev",
    n_null_pval: int = 2000,
) -> dict[str, Any]:
    raw = load_raw_records(raw_path)
    rows = normalize_records(
        raw, labels_by_id=labels_by_id, label_dist_by_id=label_dist_by_id
    )
    arms = run_pass_sensitivity(rows=rows, client=client, n_null_pval=n_null_pval)
    return {
        "schema": "jevbench.exp1_sensitivity_passes.v1",
        "label": SENSITIVITY_LABEL,
        "note": (
            "Post-hoc robustness only. Soft corrected ΔECE on each of passes "
            "0..9 and on per-item mean probabilities across the 10 passes. "
            "Same 750/750 locked items and soft_delta_ece_corrected settings "
            f"(n_null_pval={n_null_pval}) as primary. Not a registered endpoint."
        ),
        "source": {
            "run_id": run_id,
            "raw": str(raw_path),
            "client": client,
        },
        "n_null_pval": n_null_pval,
        "arms": arms,
    }


def write_sensitivity(
    payload: dict[str, Any], out_path: Path
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    return out_path
