#!/usr/bin/env python3
"""Step 15 B — baseline calibration via analyze_client (Amendments 10/11).

Writes results/exp1_baselines.json. Does not modify results/exp1.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.baseline_diagnostics import load_source_by_id  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp1 import (  # noqa: E402
    analyze_client,
    load_raw_records,
    normalize_records,
)
from jevbench.prereg import datasets_root  # noqa: E402

import importlib.util  # noqa: E402


def _resolve_exp1_run(repo: Path, run_arg: str):
    spec = importlib.util.spec_from_file_location(
        "run_exp1_analyze", repo / "scripts" / "run_exp1_analyze.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_exp1_run(repo, run_arg)

# Seed offsets mirror run_exp1_analysis (baseline = seed+1) with one slot
# per local arm so each arm is reproducible and distinct.
BASELINE_SEEDS = {
    "prefill_qwen15": 1,
    "gliclass": 11,
    "bart_mnli_ref": 21,
    "trivial": 31,
}


def _attach_source(
    rows: list[dict[str, Any]], source_by_id: dict[str, str]
) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        rr = dict(r)
        iid = str(r.get("item_id"))
        if iid in source_by_id:
            rr["source"] = source_by_id[iid]
        out.append(rr)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="latest-exp1")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument(
        "--quick",
        action="store_true",
        help="n_boot=200 for a smoke pass",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "exp1_baselines.json",
    )
    args = p.parse_args()
    n_boot = 200 if args.quick else args.n_boot

    run_id, raw_path, _man = _resolve_exp1_run(ROOT, args.run)
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
    rows = _attach_source(rows, source_by_id)

    arms: dict[str, Any] = {}
    for client, offset in BASELINE_SEEDS.items():
        print(f"analyze_client {client} (choice, seed+{offset}) …", flush=True)
        block = analyze_client(
            rows,
            client=client,
            seed=args.seed + offset,
            n_boot=n_boot,
            primitive_arm="choice",
        )
        arms[client] = block

    # Amendment 11: BART MNLI-only and SNLI-only
    bart_rows_mnli = [r for r in rows if r.get("source") == "mnli"]
    bart_rows_snli = [r for r in rows if r.get("source") == "snli"]
    print("analyze_client bart_mnli_ref MNLI-only …", flush=True)
    bart_mnli = analyze_client(
        bart_rows_mnli,
        client="bart_mnli_ref",
        seed=args.seed + 22,
        n_boot=n_boot,
        primitive_arm="choice",
    )
    print("analyze_client bart_mnli_ref SNLI-only …", flush=True)
    bart_snli = analyze_client(
        bart_rows_snli,
        client="bart_mnli_ref",
        seed=args.seed + 23,
        n_boot=n_boot,
        primitive_arm="choice",
    )

    payload = {
        "schema": "jevbench.exp1_baselines.v1",
        "prereg_clause": (
            "Amendments 10/11 — local baseline arms analysed with existing "
            "analyze_client (choice arm, soft primary, equal-n, bias correction, "
            "n_null_pval=2000); previously pointed at removed adapter_baseline"
        ),
        "run_id": run_id,
        "raw": str(raw_path),
        "seed": args.seed,
        "n_boot": n_boot,
        "seed_offsets": BASELINE_SEEDS,
        "arms": arms,
        "bart_mnli_ref_by_source": {
            "role": "supervised_in_domain_reference",
            "mnli_only": bart_mnli,
            "snli_only": bart_snli,
        },
        "note": (
            "Do not put bart_mnli_ref in the same comparison table as Jev. "
            "Descriptive_direction and divergence_to_humans are inside each "
            "analyze_client block when soft scoring is available."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
