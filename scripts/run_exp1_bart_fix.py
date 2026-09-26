#!/usr/bin/env python3
"""Analyse the post-data bart_mnli_nli fix run → results/exp1_bart_fix.json.

Does not modify results/exp1_baselines.json or the broken bart_mnli_ref arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.baseline_diagnostics import load_source_by_id  # noqa: E402
from jevbench.chaosnli import LABEL_ORDER  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp1 import (  # noqa: E402
    analyze_client,
    load_raw_records,
    normalize_records,
)
from jevbench.prereg import datasets_root  # noqa: E402
from jevbench.secondary import auroc_binary, top_uncertainty  # noqa: E402

CLIENT = "bart_mnli_nli"


def _confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = list(LABEL_ORDER)
    mat = {t: {p: 0 for p in labels} for t in labels}
    n = 0
    correct = 0
    for r in rows:
        if r.get("client") != CLIENT:
            continue
        if r.get("role", "primary") == "paraphrase":
            continue
        if int(r.get("pass", 0)) not in (0, 1):
            continue
        truth = str(r.get("label"))
        pred = str(r.get("choice"))
        if truth not in mat or pred not in mat[truth]:
            continue
        mat[truth][pred] += 1
        n += 1
        correct += int(truth == pred)
    return {
        "n": n,
        "accuracy_vs_majority": (correct / n) if n else float("nan"),
        "matrix": mat,
        "label_order": labels,
    }


def _s1_one_minus_top(rows: list[dict[str, Any]]) -> dict[str, Any]:
    items = [
        r
        for r in rows
        if r.get("client") == CLIENT
        and r.get("role", "primary") != "paraphrase"
        and isinstance(r.get("probabilities"), dict)
        and isinstance(r.get("label_dist"), (list, tuple))
    ]
    # one row per item
    by: dict[str, dict[str, Any]] = {}
    for r in sorted(items, key=lambda x: int(x.get("pass", 0))):
        by.setdefault(str(r["item_id"]), r)
    items = list(by.values())
    if not items:
        return {"n": 0}
    probs = np.asarray(
        [[float(r["probabilities"].get(lab, 0.0)) for lab in LABEL_ORDER] for r in items]
    )
    unc = top_uncertainty(probs)
    hard = np.asarray([1 if r.get("tier") in ("hard", "ambiguous") else 0 for r in items])
    return {
        "n": int(len(items)),
        "signal": "one_minus_top",
        "auroc_hard_vs_easy": float(auroc_binary(unc, hard)),
        "mean_unc_easy": float(unc[hard == 0].mean()) if np.any(hard == 0) else float("nan"),
        "mean_unc_hard": float(unc[hard == 1].mean()) if np.any(hard == 1) else float("nan"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="bart_fix run_id under runs/")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "exp1_bart_fix.json",
    )
    args = p.parse_args()
    n_boot = 200 if args.quick else args.n_boot

    rdir = ROOT / "runs" / args.run
    raw_path = rdir / "raw.jsonl"
    if not raw_path.is_file():
        print(f"missing {raw_path}", file=sys.stderr)
        return 2

    ds = load_dataset(datasets_root(ROOT), "chaosnli")
    labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
    label_dist_by_id = {
        li.id: list(li.label.label_dist)
        for li in ds.by_id.values()
        if li.label.label_dist is not None
    }
    raw = load_raw_records(raw_path)
    rows = normalize_records(
        raw, labels_by_id=labels_by_id, label_dist_by_id=label_dist_by_id
    )
    source_by_id = load_source_by_id(ROOT / "datasets" / "chaosnli" / "items.jsonl")
    for r in rows:
        iid = str(r.get("item_id"))
        if iid in source_by_id:
            r["source"] = source_by_id[iid]

    print("analyze_client bart_mnli_nli (all) …", flush=True)
    all_block = analyze_client(
        rows, client=CLIENT, seed=args.seed + 21, n_boot=n_boot, primitive_arm="choice"
    )
    mnli_rows = [r for r in rows if r.get("source") == "mnli"]
    snli_rows = [r for r in rows if r.get("source") == "snli"]
    print("analyze_client bart_mnli_nli MNLI-only …", flush=True)
    mnli_block = analyze_client(
        mnli_rows, client=CLIENT, seed=args.seed + 22, n_boot=n_boot, primitive_arm="choice"
    )
    print("analyze_client bart_mnli_nli SNLI-only …", flush=True)
    snli_block = analyze_client(
        snli_rows, client=CLIENT, seed=args.seed + 23, n_boot=n_boot, primitive_arm="choice"
    )

    conf_all = _confusion(rows)
    conf_mnli = _confusion(mnli_rows)
    conf_snli = _confusion(snli_rows)
    s1 = _s1_one_minus_top(rows)

    payload = {
        "schema": "jevbench.exp1_bart_fix.v1",
        "label": "POST-DATA BUG FIX: bart_mnli_nli",
        "note": (
            "Replaces the broken zero-shot str(state) bart_mnli_ref arm with true "
            "NLI on (premise, hypothesis). The original broken run stays in "
            "results/exp1_baselines.json and is still reported."
        ),
        "run_id": args.run,
        "raw": str(raw_path),
        "client": CLIENT,
        "role": "supervised_in_domain_reference",
        "n_boot": n_boot,
        "all": all_block,
        "by_source": {"mnli_only": mnli_block, "snli_only": snli_block},
        "accuracy_vs_majority": {
            "all": conf_all["accuracy_vs_majority"],
            "mnli": conf_mnli["accuracy_vs_majority"],
            "snli": conf_snli["accuracy_vs_majority"],
        },
        "confusion": {"all": conf_all, "mnli": conf_mnli, "snli": conf_snli},
        "s1_one_minus_top": s1,
        "n_raw_by_source": dict(Counter(r.get("source") for r in rows if r.get("client") == CLIENT)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"accuracy all={conf_all['accuracy_vs_majority']:.4f} "
        f"mnli={conf_mnli['accuracy_vs_majority']:.4f} "
        f"snli={conf_snli['accuracy_vs_majority']:.4f}"
    )
    print(
        f"corrected dECE all={all_block.get('delta_corrected')} "
        f"mnli={mnli_block.get('delta_corrected')} "
        f"snli={snli_block.get('delta_corrected')}"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
