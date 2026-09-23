#!/usr/bin/env python3
"""Run EXP-1 analysis → results/exp1.json (+ verdict).

Default source: offline fixture (no network). Pass --raw for a live run dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.agreement import compute_agreement  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp1 import run_exp1_analysis  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "runs" / "offline_fixture" / "raw.jsonl",
        help="Path to raw.jsonl (fixture or runs/<id>/raw.jsonl)",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "exp1.json")
    p.add_argument("--task", default="support_tickets")
    p.add_argument("--jev-client", default="jev")
    p.add_argument("--baseline-client", default="adapter_baseline")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument(
        "--quick",
        action="store_true",
        help="n_boot=500 for a smoke pass",
    )
    p.add_argument(
        "--scored",
        action="store_true",
        help="Mark as scored live run (enables underpowered gate in verdict)",
    )
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    n_boot = 500 if args.quick else args.n_boot
    raw = args.raw
    if not raw.is_file():
        print(f"missing raw.jsonl: {raw}", file=sys.stderr)
        return 2

    labels_by_id = None
    # Live runner records need labels joined from the dataset
    sample = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    if "label" not in sample:
        ds = load_dataset(datasets_root(ROOT), args.task)
        labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
        label_dist_by_id = {
            li.id: list(li.label.label_dist)
            for li in ds.by_id.values()
            if li.label.label_dist is not None
        }
    else:
        label_dist_by_id = None

    agreement = compute_agreement(
        ROOT / "datasets" / args.task, kappa_floor=0.6
    ).to_dict()
    power = None
    pa = ROOT / "results" / "power_analysis.json"
    if pa.is_file():
        power = json.loads(pa.read_text(encoding="utf-8"))

    payload = run_exp1_analysis(
        raw_path=raw,
        out_path=args.out,
        labels_by_id=labels_by_id,
        label_dist_by_id=label_dist_by_id or None,
        jev_client=args.jev_client,
        baseline_client=args.baseline_client,
        seed=args.seed,
        n_boot=n_boot,
        agreement=agreement,
        power=power,
        scored=args.scored,
        labelling_ceiling_applies=args.task != "chaosnli",
        contamination_arms=["jev", "adapter_baseline", "trivial"],
    )
    print(payload["verdict"])
    print(f"wrote {args.out}")
    if payload.get("gates", {}).get("block_reason") and args.scored:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
