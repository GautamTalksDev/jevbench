#!/usr/bin/env python3
"""Run EXP-2 decomposition analysis → results/exp2.json.

Honest claim (do not weaken):
  \"Jev plus N labelled examples outperforms Jev alone AND outperforms a
   zero-shot frontier LLM, at a fraction of the cost.\"

A vs C = zero-shot. B vs D = supervised. Never headline B vs C alone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp2 import run_exp2_analysis  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "runs" / "offline_fixture" / "raw.jsonl",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "exp2.json")
    p.add_argument("--task", default="support_tickets")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--quick", action="store_true", help="n_boot=500")
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--no-chart", action="store_true")
    args = p.parse_args()

    if not args.raw.is_file():
        print(f"missing {args.raw}", file=sys.stderr)
        return 2

    ds = load_dataset(datasets_root(ROOT), args.task)
    labels = {li.id: str(li.label.label) for li in ds.by_id.values()}
    tiers = {li.id: li.tier for li in ds.by_id.values()}

    payload = run_exp2_analysis(
        raw_path=args.raw,
        out_path=args.out,
        labels_by_id=labels,
        tiers_by_id=tiers,
        seed=args.seed,
        n_boot=500 if args.quick else args.n_boot,
        write_chart=not args.no_chart,
    )
    print(payload["verdict"])
    print(f"labels_to_beat_zero_shot_llm: {payload.get('labels_to_beat_zero_shot_llm')}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
