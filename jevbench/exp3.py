"""EXP-3 moat control analysis.

Deciding metric: paired ΔECE(Jev, prefill) — NOT accuracy, NOT latency.

If Jev's probabilities are better calibrated than raw softmaxed logprobs from
a small open model on the same items, RLCD is doing real work and the moat is
real. If not, Goedecke's reading is supported.

Latency is reported for completeness but flagged as NOT a fair comparison
(local GPU vs hosted API). Do not use EXP-3 for video latency claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from jevbench.metrics import (
    accuracy,
    expected_calibration_error,
    macro_f1,
)

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
LABEL_ORDER = ("billing", "technical", "other")
N_BOOT = 10_000
SEED = 20260922

LATENCY_NOTE = (
    "Local GPU inference and a hosted API are not comparable latency "
    "measurements. Network RTT, batching, cold start and queueing all differ. "
    "wall_clock_ms is flagged unfair; compute_only_ms is the closest honest "
    "analogue for the local prefill arm. Do NOT use EXP-3 for video latency "
    "claims — those come from EXP-1's matched serving path."
)

DECIDING_NOTE = (
    "Deciding metric is paired ΔECE(Jev, prefill) with BCa intervals. "
    "Accuracy and latency are secondary / completeness only."
)


def _prob_row(probs: dict[str, float]) -> list[float]:
    return [float(probs.get(lab, 0.0)) for lab in LABEL_ORDER]


def _label_index(label: str) -> int:
    if label not in LABEL_ORDER:
        raise ValueError(f"unknown label {label!r}")
    return LABEL_ORDER.index(label)


def load_arm_arrays(
    records: list[dict[str, Any]],
    client: str,
    *,
    pass_idx: int | None = 0,
    labels_by_id: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Return (probs [n,C], y [n], item_ids, latency_bundle) for one client."""
    by_item: dict[str, dict[str, Any]] = {}
    for rec in records:
        if rec.get("client") != client:
            continue
        iid = str(rec["item_id"])
        p = int(rec.get("pass", 0))
        if pass_idx is not None and p != pass_idx:
            continue
        prev = by_item.get(iid)
        if prev is not None and pass_idx is None and int(prev.get("pass", 0)) <= p:
            continue

        label = rec.get("label") or (labels_by_id or {}).get(iid)
        probs = None
        choice = None
        wall = float(rec.get("latency_ms") or 0.0)
        compute = None
        if "probabilities" in rec and isinstance(rec["probabilities"], dict):
            probs = rec["probabilities"]
            choice = rec.get("choice")
        else:
            for d in rec.get("decisions") or []:
                qk = str(d.get("question_key") or "")
                if (
                    qk in ("department", "verdict") or d.get("kind") == "choice"
                ) and isinstance(d.get("probabilities"), dict):
                    probs = d["probabilities"]
                    choice = d.get("value") or d.get("choice")
                    wall = float(d.get("latency_ms") or wall)
                    lat = (d.get("raw") or {}).get("latency") or {}
                    if "compute_only_ms" in lat:
                        compute = float(lat["compute_only_ms"])
                    break
        if probs is None or label is None:
            continue
        by_item[iid] = {
            "probs": probs,
            "label": label,
            "choice": choice,
            "pass": p,
            "wall_ms": wall,
            "compute_ms": compute,
        }

    if not by_item and pass_idx is not None:
        # Fallback: any pass
        return load_arm_arrays(
            records, client, pass_idx=None, labels_by_id=labels_by_id
        )

    ids = sorted(by_item)
    P = np.asarray([_prob_row(by_item[i]["probs"]) for i in ids], dtype=float)
    # Renormalize rows
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    P = P / row_sums
    y = np.asarray([_label_index(str(by_item[i]["label"])) for i in ids], dtype=int)
    preds = []
    for i in ids:
        ch = by_item[i].get("choice")
        if ch in LABEL_ORDER:
            preds.append(_label_index(str(ch)))
        else:
            preds.append(int(np.argmax(_prob_row(by_item[i]["probs"]))))
    walls = [by_item[i]["wall_ms"] for i in ids]
    computes = [
        by_item[i]["compute_ms"]
        for i in ids
        if by_item[i]["compute_ms"] is not None
    ]
    latency = {
        "wall_clock_ms": {
            "p50": float(np.percentile(walls, 50)) if walls else float("nan"),
            "p95": float(np.percentile(walls, 95)) if walls else float("nan"),
            "p99": float(np.percentile(walls, 99)) if walls else float("nan"),
            "fair_comparison": False,
            "reason": LATENCY_NOTE,
        },
        "compute_only_ms": {
            "p50": float(np.percentile(computes, 50)) if computes else float("nan"),
            "p95": float(np.percentile(computes, 95)) if computes else float("nan"),
            "p99": float(np.percentile(computes, 99)) if computes else float("nan"),
            "n_with_compute": len(computes),
            "note": "Closest honest local analogue; still not matched to hosted RTT.",
        },
    }
    meta = {
        "n": len(ids),
        "latency": latency,
        "predictions": preds,
    }
    return P, y, ids, meta


def _arm_metrics(P: np.ndarray, y: np.ndarray, preds: list[int]) -> dict[str, Any]:
    ece = expected_calibration_error(P, y, n_bins=10, strategy="uniform")
    pred_labels = [LABEL_ORDER[i] for i in preds]
    true_labels = [LABEL_ORDER[i] for i in y.tolist()]
    return {
        "n": len(y),
        "accuracy": accuracy(pred_labels, true_labels),
        "macro_f1": macro_f1(pred_labels, true_labels),
        "ece": ece.to_dict(),
        "ece_scalar": ece.ece,
    }


def permutation_stability(
    records: list[dict[str, Any]],
    *,
    client: str = "prefill",
    labels_by_id: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Accuracy / ECE across pass_idx — bias finding about the CONTROL."""
    passes = sorted(
        {
            int(r.get("pass", 0))
            for r in records
            if r.get("client") == client
        }
    )
    rows: list[dict[str, Any]] = []
    for p in passes:
        P, y, _ids, meta = load_arm_arrays(
            records, client, pass_idx=p, labels_by_id=labels_by_id
        )
        if len(y) == 0:
            continue
        m = _arm_metrics(P, y, meta["predictions"])
        rows.append({"pass_idx": p, "n": m["n"], "accuracy": m["accuracy"], "ece": m["ece_scalar"]})
    if len(rows) < 2:
        return {
            "n_passes": len(rows),
            "rows": rows,
            "accuracy_range": None,
            "finding": "insufficient passes for permutation control",
        }
    accs = [r["accuracy"] for r in rows]
    eces = [r["ece"] for r in rows]
    acc_range = float(max(accs) - min(accs))
    # Material move: >5pp accuracy or >0.02 ECE
    material = acc_range > 0.05 or (max(eces) - min(eces)) > 0.02
    return {
        "n_passes": len(rows),
        "rows": rows,
        "accuracy_range": acc_range,
        "ece_range": float(max(eces) - min(eces)),
        "material_move": material,
        "finding": (
            "Prefill accuracy/ECE moves under sentinel order permutation — "
            "positional or numeric bias in the CONTROL (not a Jev finding)."
            if material
            else "Prefill metrics stable under sentinel order permutation."
        ),
    }


def run_exp3_analysis(
    *,
    raw_path: Path,
    out_path: Path | None = None,
    labels_by_id: dict[str, str] | None = None,
    jev_client: str = "jev",
    prefill_client: str = "prefill",
    gliclass_client: str = "gliclass",
    adapter_client: str = "adapter_baseline",
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clients_present = {r.get("client") for r in records}
    if adapter_client not in clients_present and "adapter" in clients_present:
        adapter_client = "adapter"

    arms: dict[str, Any] = {}
    arrays: dict[str, tuple] = {}
    for name, client in (
        ("jev", jev_client),
        ("prefill", prefill_client),
        ("gliclass", gliclass_client),
        ("adapter", adapter_client),
    ):
        if client not in clients_present:
            arms[name] = {"error": f"client {client!r} absent from raw.jsonl"}
            continue
        P, y, ids, meta = load_arm_arrays(
            records, client, pass_idx=0, labels_by_id=labels_by_id
        )
        if len(y) == 0:
            arms[name] = {"error": "no usable rows"}
            continue
        m = _arm_metrics(P, y, meta["predictions"])
        arms[name] = {
            "client": client,
            **m,
            "latency": meta["latency"],
            "item_ids": ids,
        }
        arrays[name] = (P, y, ids)

    # Paired ΔECE: treat prefill as "hard" and jev as "easy" so point =
    # ECE(prefill) − ECE(jev). Same items — but delta_ece requires equal n
    # and does stratified two-sample bootstrap. For PAIRED same-items we need
    # a paired bootstrap of the ECE difference.
    deciding: dict[str, Any]
    if "jev" in arrays and "prefill" in arrays:
        Pj, yj, ids_j = arrays["jev"]
        Pp, _yp, ids_p = arrays["prefill"]
        common = sorted(set(ids_j) & set(ids_p))
        j_map = {i: k for k, i in enumerate(ids_j)}
        p_map = {i: k for k, i in enumerate(ids_p)}
        Pj_c = np.asarray([Pj[j_map[i]] for i in common])
        Pp_c = np.asarray([Pp[p_map[i]] for i in common])
        y_c = np.asarray([yj[j_map[i]] for i in common])

        def _delta(a_probs: np.ndarray, b_probs: np.ndarray) -> float:
            # paired bootstrap resamples rows of both together
            return float(
                expected_calibration_error(a_probs, y_c).ece
                - expected_calibration_error(b_probs, y_c).ece
            )

        # Point: ECE(prefill) − ECE(jev)
        point = float(
            expected_calibration_error(Pp_c, y_c).ece
            - expected_calibration_error(Pj_c, y_c).ece
        )
        # Paired residual bootstrap via index resampling
        rng = np.random.default_rng(seed)
        samples = np.empty(n_boot, dtype=float)
        n = len(common)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            samples[b] = float(
                expected_calibration_error(Pp_c[idx], y_c[idx]).ece
                - expected_calibration_error(Pj_c[idx], y_c[idx]).ece
            )
        # Also jackknife for BCa
        from jevbench.metrics import _bca_endpoints

        jack = np.empty(n, dtype=float)
        for i in range(n):
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            jack[i] = float(
                expected_calibration_error(Pp_c[mask], y_c[mask]).ece
                - expected_calibration_error(Pj_c[mask], y_c[mask]).ece
            )
        b_low, b_high, z0, accel = _bca_endpoints(point, samples, jack)
        p_low = float(np.quantile(samples, 0.025))
        p_high = float(np.quantile(samples, 0.975))
        deciding = {
            "metric": "paired_delta_ece",
            "definition": "ECE(prefill) − ECE(jev) on identical items",
            "n_common": n,
            "point": point,
            "ci_bca": [b_low, b_high],
            "ci_percentile": [p_low, p_high],
            "bca_z0": z0,
            "bca_acceleration": accel,
            "crosses_zero_bca": b_low < 0 < b_high,
            "crosses_zero_percentile": p_low < 0 < p_high,
            "n_boot": n_boot,
            "seed": seed,
            "interpretation": (
                "CI entirely above 0 ⇒ prefill worse calibrated (moat evidence). "
                "CI entirely below 0 ⇒ prefill better/equal (Goedecke supported). "
                "Crossing 0 ⇒ inconclusive."
            ),
        }
    else:
        deciding = {
            "error": "need both jev and prefill arms in raw.jsonl",
            "note": DECIDING_NOTE,
        }

    perm = permutation_stability(
        records, client=prefill_client, labels_by_id=labels_by_id
    )

    verdict = _verdict(deciding, arms, perm)
    payload = {
        "schema": "jevbench.exp3.v1",
        "deciding_metric_note": DECIDING_NOTE,
        "latency_note": LATENCY_NOTE,
        "source": {"raw": str(raw_path), "n_records": len(records)},
        "arms": arms,
        "paired_delta_ece_prefill_minus_jev": deciding,
        "prefill_order_permutation": perm,
        "verdict": verdict,
    }
    out_path = out_path or (RESULTS / "exp3.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _verdict(
    deciding: dict[str, Any],
    arms: dict[str, Any],
    perm: dict[str, Any],
) -> str:
    parts = [
        (
            "EXP-3 moat control. Deciding metric is paired ΔECE(prefill − Jev), "
            "not accuracy or latency."
        )
    ]
    if "point" in deciding:
        parts.append(
            f"ΔECE={deciding['point']:.4f} BCa {deciding['ci_bca']} "
            f"percentile {deciding['ci_percentile']} (n={deciding['n_common']})."
        )
        if deciding.get("crosses_zero_bca"):
            parts.append(
                "BCa CI crosses zero — inconclusive on whether RLCD improves "
                "calibration beyond constrained single-token softmax."
            )
        elif deciding["ci_bca"][0] > 0:
            parts.append(
                "BCa CI lies above zero: prefill is worse calibrated than Jev "
                "on these items — evidence the moat (RLCD) is real."
            )
        else:
            parts.append(
                "BCa CI lies below zero: prefill is better/equal calibrated — "
                "Goedecke's reading is supported on this corpus."
            )
    else:
        parts.append(str(deciding.get("error") or deciding))

    for name in ("jev", "prefill", "gliclass", "adapter"):
        arm = arms.get(name) or {}
        if "ece_scalar" in arm:
            parts.append(
                f"{name}: ECE={arm['ece_scalar']:.4f} acc={arm.get('accuracy', float('nan')):.3f}."
            )
    parts.append(perm.get("finding", ""))
    parts.append(
        "Latency figures are flagged unfair (local vs hosted); do not use "
        "EXP-3 for video latency claims."
    )
    return " ".join(parts)
