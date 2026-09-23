#!/usr/bin/env python3
"""Run the offline EXP-1 power analysis. No network. No API key.

Prints the budget decision line. Do not label a single item until it exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.power import (  # noqa: E402
    N_BOOT_POWER,
    N_TRIALS_DEFAULT,
    format_recommendation,
    run_power_analysis,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT)
    p.add_argument("--n-boot", type=int, default=N_BOOT_POWER)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument("--quick", action="store_true", help="trials=50 for a smoke pass")
    p.add_argument(
        "--coverage-trials",
        type=int,
        default=5000,
        help="null FPR trials for coverage.json (Prompt F2)",
    )
    p.add_argument("--no-surfaces", action="store_true")
    p.add_argument("--no-coverage", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    trials = 50 if args.quick else args.trials
    payload = run_power_analysis(
        out_dir=args.out_dir,
        n_trials=trials,
        n_boot=args.n_boot,
        seed=args.seed,
        write_chart=True,
        run_surfaces=not args.quick and not args.no_surfaces,
        run_coverage=not args.quick and not args.no_coverage,
        coverage_trials=100 if args.quick else args.coverage_trials,
        max_workers=args.workers,
    )

    print()
    print("=" * 72)
    print(format_recommendation(payload))
    print("=" * 72)
    print(f"wrote {args.out_dir / 'power_analysis.json'}")
    if payload.get("charts"):
        print(f"charts: {payload['charts']}")
    print(f"elapsed {payload['elapsed_s']}s")

    # Fail loudly if FPR is broken (also raised inside run_power_analysis)
    if payload["mean_null_fpr"] > 0.10:
        sys.exit(2)
    if payload.get("underpowered"):
        sys.exit(4)
    if payload["chosen_n_per_stratum"] is None:
        sys.exit(3)


if __name__ == "__main__":
    main()
