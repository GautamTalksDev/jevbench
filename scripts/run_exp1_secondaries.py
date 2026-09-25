#!/usr/bin/env python3
"""Step 15 C–E — S1/S2/S3 secondaries (Amendment 10.3). Offline only."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.chaosnli import LABEL_ORDER, shannon_entropy  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp1 import (  # noqa: E402
    HARD_TIERS,
    _client_pass0,
    load_raw_records,
    normalize_records,
)
from jevbench.prereg import datasets_root  # noqa: E402
from jevbench.secondary import (  # noqa: E402
    S1_N_BOOT,
    S1_SEED,
    ambiguity_detection_table,
    auroc_binary,
    bootstrap_metric_ci,
    choice_margin,
    cross_primitive_jsd,
    per_item_repeat_signals,
    spearman_vs_entropy,
    stability_reconciliation_table,
    temperature_scaling_cv,
    top_uncertainty,
)


def _resolve_exp1_run(repo: Path, run_arg: str):
    spec = importlib.util.spec_from_file_location(
        "run_exp1_analyze", repo / "scripts" / "run_exp1_analyze.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_exp1_run(repo, run_arg)


def _prob_row(probs: dict[str, float]) -> list[float]:
    return [float(probs.get(lab, 0.0)) for lab in LABEL_ORDER]


def _entropy_from_dist(dist: list[float]) -> float:
    counts = [max(0, int(round(x * 100))) for x in dist]
    # Prefer exact label_count when available via dist*100; shannon_entropy needs counts
    if sum(counts) <= 0:
        p = np.asarray(dist, dtype=float)
        p = p / max(p.sum(), 1e-12)
        terms = np.zeros_like(p)
        nz = p > 0
        terms[nz] = p[nz] * np.log2(p[nz])
        return float(-terms.sum())
    return float(shannon_entropy(counts))


def _aligned_primary(
    rows: list[dict[str, Any]], client: str
) -> list[dict[str, Any]]:
    items = [
        r
        for r in _client_pass0(rows, client)
        if r.get("role", "primary") != "paraphrase"
        and isinstance(r.get("probabilities"), dict)
        and isinstance(r.get("label_dist"), (list, tuple))
    ]
    items.sort(key=lambda r: str(r["item_id"]))
    return items


def _jev_repeats_by_item(
    rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("client") != "jev":
            continue
        if r.get("role", "primary") == "paraphrase":
            continue
        if not isinstance(r.get("probabilities"), dict):
            continue
        by[str(r["item_id"])].append(r)
    for iid in by:
        by[iid].sort(key=lambda x: int(x.get("pass", 0)))
    return by


def build_s1_s2_s3(
    rows: list[dict[str, Any]],
    *,
    n_boot: int = S1_N_BOOT,
    seed: int = S1_SEED,
    quick: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if quick:
        n_boot = min(n_boot, 200)

    pass0 = _aligned_primary(rows, "jev")
    ids = [str(r["item_id"]) for r in pass0]
    choice = np.asarray([_prob_row(r["probabilities"]) for r in pass0], dtype=float)
    noul_rows = [
        r for r in pass0 if isinstance(r.get("probabilities_noul"), dict)
    ]
    # Align noul to same ids (drop items missing noul)
    noul_by = {str(r["item_id"]): r for r in noul_rows}
    keep = [i for i, iid in enumerate(ids) if iid in noul_by]
    pass0 = [pass0[i] for i in keep]
    ids = [ids[i] for i in keep]
    choice = choice[keep]
    noul = np.asarray(
        [_prob_row(noul_by[iid]["probabilities_noul"]) for iid in ids], dtype=float
    )
    human = np.asarray([list(r["label_dist"]) for r in pass0], dtype=float)
    entropy = np.asarray([_entropy_from_dist(list(h)) for h in human], dtype=float)
    hard = np.asarray(
        [1 if r.get("tier") in HARD_TIERS else 0 for r in pass0], dtype=int
    )
    strata = np.asarray(
        ["hard" if h else "easy" for h in hard], dtype=object
    )
    labels = np.asarray(
        [LABEL_ORDER.index(str(r["label"])) for r in pass0], dtype=int
    )

    # Repeats for flip / TVD
    by_rep = _jev_repeats_by_item(rows)
    mats: list[np.ndarray] = []
    # Determine R from first complete item
    max_pass = max(
        (int(r.get("pass", 0)) for rows_i in by_rep.values() for r in rows_i),
        default=0,
    )
    for pidx in range(max_pass + 1):
        col = []
        ok = True
        for iid in ids:
            hits = [r for r in by_rep.get(iid, []) if int(r.get("pass", 0)) == pidx]
            if not hits:
                ok = False
                break
            col.append(_prob_row(hits[0]["probabilities"]))
        if ok:
            mats.append(np.asarray(col, dtype=float))
    repeat_sig = per_item_repeat_signals(mats)

    top_unc = top_uncertainty(choice)
    cross = cross_primitive_jsd(choice, noul)
    flip = repeat_sig["flip_rate"]
    tvd = repeat_sig["mean_tvd"]
    margin = choice_margin(choice)

    s1 = ambiguity_detection_table(
        entropy=entropy,
        hard_indicator=hard,
        top_unc=top_unc,
        cross_jsd=cross,
        flip_rate=flip,
        mean_tvd=tvd,
        n_boot=n_boot,
        seed=seed,
    )
    s1["n_items"] = len(ids)
    s1["n_repeats_used"] = len(mats)

    # Signal (1) also for each local baseline
    baseline_s1: dict[str, Any] = {}
    for client in ("prefill_qwen15", "gliclass", "bart_mnli_ref", "trivial"):
        bitems = _aligned_primary(rows, client)
        b_by = {str(r["item_id"]): r for r in bitems}
        b_choice = []
        b_hard = []
        b_ent = []
        for iid, h_flag, ent in zip(ids, hard, entropy, strict=True):
            if iid not in b_by:
                continue
            b_choice.append(_prob_row(b_by[iid]["probabilities"]))
            b_hard.append(int(h_flag))
            b_ent.append(float(ent))
        if len(b_choice) < 10:
            baseline_s1[client] = {"error": "too_few_items", "n": len(b_choice)}
            continue
        bc = np.asarray(b_choice, dtype=float)
        unc = top_uncertainty(bc)
        bh = np.asarray(b_hard, dtype=int)
        be = np.asarray(b_ent, dtype=float)
        baseline_s1[client] = {
            "n": int(len(b_choice)),
            "signal": "one_minus_top",
            "auroc_hard_vs_easy": auroc_binary(unc, bh),
            "auroc_hard_vs_easy_bootstrap": bootstrap_metric_ci(
                unc, bh, metric="auroc_binary", n_boot=n_boot, seed=seed
            ),
            "spearman_vs_entropy": spearman_vs_entropy(unc, be),
            "spearman_vs_entropy_bootstrap": bootstrap_metric_ci(
                unc, be, metric="spearman", n_boot=n_boot, seed=seed + 1
            ),
        }
    s1["local_baseline_signal_one_minus_top"] = baseline_s1

    s2 = stability_reconciliation_table(
        flip_rate=flip,
        entropy=entropy,
        margin=margin,
        n_boot=n_boot,
        seed=seed,
    )
    s2["n_items"] = len(ids)

    soft_dummy = choice.max(axis=1)  # unused by fit; API compat
    s3_choice = temperature_scaling_cv(
        choice,
        soft_dummy,
        human,
        strata,
        labels=labels,
        seed=seed,
    )
    s3_choice["arm"] = "choice"
    s3_noul = temperature_scaling_cv(
        noul,
        soft_dummy,
        human,
        strata,
        labels=labels,
        seed=seed + 7,
    )
    s3_noul["arm"] = "noul"
    s3 = {
        "schema": "jevbench.exp1_s3.v1",
        "prereg_clause": "Amendment 10.3 S3 Temperature scaling",
        "choice": s3_choice,
        "noul": s3_noul,
        "n_items": len(ids),
    }
    return s1, s2, s3


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="latest-exp1")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-boot", type=int, default=S1_N_BOOT)
    p.add_argument("--seed", type=int, default=S1_SEED)
    p.add_argument("--out-s1", type=Path, default=ROOT / "results" / "exp1_s1.json")
    p.add_argument("--out-s2", type=Path, default=ROOT / "results" / "exp1_s2.json")
    p.add_argument("--out-s3", type=Path, default=ROOT / "results" / "exp1_s3.json")
    args = p.parse_args()

    run_id, raw_path, _man = _resolve_exp1_run(ROOT, args.run)
    ds = load_dataset(datasets_root(ROOT), "chaosnli")
    labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
    label_dist_by_id = {
        li.id: list(li.label.label_dist)
        for li in ds.by_id.values()
        if li.label.label_dist is not None
    }
    print(f"loading {raw_path} …", flush=True)
    raw = load_raw_records(raw_path)
    rows = normalize_records(
        raw, labels_by_id=labels_by_id, label_dist_by_id=label_dist_by_id
    )
    print("computing S1/S2/S3 …", flush=True)
    s1, s2, s3 = build_s1_s2_s3(
        rows, n_boot=args.n_boot, seed=args.seed, quick=args.quick
    )
    for path, doc in (
        (args.out_s1, s1),
        (args.out_s2, s2),
        (args.out_s3, s3),
    ):
        doc = dict(doc)
        doc["run_id"] = run_id
        doc["raw"] = str(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
