#!/usr/bin/env python3
"""Prompt J offline diagnostics: asymmetric-noise power + BCa mean vs ΔECE.

No network. No API key. Writes:
  results/power_asymmetric.json
  results/bca_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.power import (  # noqa: E402
    N_BOOT_POWER,
    N_TRIALS_COVERAGE,
    run_asymmetric_power,
    run_bca_diagnostic,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=750)
    p.add_argument("--power-trials", type=int, default=400)
    p.add_argument("--coverage-trials", type=int, default=N_TRIALS_COVERAGE)
    p.add_argument("--n-boot", type=int, default=N_BOOT_POWER)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    p.add_argument("--skip-bca", action="store_true")
    p.add_argument("--skip-power", action="store_true")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    power_trials = 40 if args.quick else args.power_trials
    cov_trials = 100 if args.quick else args.coverage_trials

    if not args.skip_power:
        asym = run_asymmetric_power(
            n=args.n,
            n_trials=power_trials,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        path = out / "power_asymmetric.json"
        path.write_text(json.dumps(asym, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        print(
            f"power soft={asym['power_soft']:.3f} hard={asym['power_hard']:.3f} "
            f"null_mean soft={asym['null_mean_delta_ece_soft']:.4f} "
            f"hard={asym['null_mean_delta_ece_hard']:.4f}"
        )
        print(asym["note"])

    if not args.skip_bca:
        diag = run_bca_diagnostic(
            n=args.n,
            n_trials=cov_trials,
            n_boot=args.n_boot,
            seed=args.seed + 11,
            max_workers=args.workers,
        )
        path = out / "bca_diagnostic.json"
        path.write_text(json.dumps(diag, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        print(diag["explanation"]["text"])
        print(f"explanation={diag['explanation']['explanation']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
