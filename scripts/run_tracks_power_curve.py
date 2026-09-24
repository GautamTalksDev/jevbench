#!/usr/bin/env python3
"""Composite tracks-gate power curve across true ΔECE ∈ {0.09,0.11,0.13,0.15}."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.verdict import tracks_power_curve  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument(
        "--out", type=Path, default=ROOT / "results" / "tracks_power_curve.json"
    )
    args = p.parse_args()

    out = tracks_power_curve(
        n_trials=args.trials, seed=args.seed, quick=args.quick
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in out["curve"]:
        print(
            f"true Δ={row['true_delta_ece']:.2f}  "
            f"mean_corr={row['mean_corrected']:.4f}  "
            f"P(tracks)={row['power_tracks']:.3f}  "
            f"P(p<0.05)={row['power_pvalue']:.3f}",
            flush=True,
        )
    print(out["note"], flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
