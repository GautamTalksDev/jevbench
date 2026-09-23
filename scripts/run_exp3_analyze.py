#!/usr/bin/env python3
"""Run EXP-3 moat analysis → results/exp3.json.

Deciding metric: paired ΔECE(Jev, prefill). Latency is flagged unfair.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp3 import run_exp3_analysis  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "runs" / "offline_fixture" / "raw.jsonl",
        help="Needs jev + prefill clients; fixture may only smoke-partial",
    )
    p.add_argument("--out", type=Path, default=ROOT / "results" / "exp3.json")
    p.add_argument("--task", default="support_tickets")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=20260922)
    args = p.parse_args()

    if not args.raw.is_file():
        print(f"missing {args.raw}", file=sys.stderr)
        return 2

    ds = load_dataset(datasets_root(ROOT), args.task)
    labels = {li.id: str(li.label.label) for li in ds.by_id.values()}

    payload = run_exp3_analysis(
        raw_path=args.raw,
        out_path=args.out,
        labels_by_id=labels,
        seed=args.seed,
        n_boot=500 if args.quick else args.n_boot,
    )
    print(payload["verdict"])
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
