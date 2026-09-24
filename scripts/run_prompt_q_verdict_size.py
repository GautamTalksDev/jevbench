#!/usr/bin/env python3
"""PROMPT Q — re-verify Amendment 9 verdict-rule size (offline, no Jev)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.verdict import verify_verdict_size  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "verdict_size.json")
    args = p.parse_args()

    out = verify_verdict_size(
        n_trials=args.trials, seed=args.seed, quick=args.quick
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(out["null"], indent=2))
    print(json.dumps(out["alternative_at_0_09"], indent=2))
    print(json.dumps(out["acceptance"], indent=2))
    print(f"wrote {args.out}", flush=True)
    acc = out["acceptance"]
    if not (
        acc["pvalue_fpr_ok"]
        and acc["holds_boundary_size_ok"]
        and acc["tracks_null_fpr_low"]
        and acc["holds_at_effect_rare"]
        and acc["pvalue_power_at_0_09_ok"]
    ):
        print("SIZE CHECK FAILED", file=sys.stderr)
        return 2
    print(acc["tracks_power_at_boundary_note"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
