#!/usr/bin/env python3
"""PROMPT O — soft-scoring null with Beta–Binomial item-level scatter.

Offline. No network. No Jev data.

Writes:
  results/soft_null_kappa.json
  results/power_asymmetric.json  (hard path unchanged; soft path uses new null)
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
    SOFT_NULL_OPERATING_KAPPA,
    run_asymmetric_power,
    run_soft_null_kappa_sweep,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=750)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--n-boot", type=int, default=N_BOOT_POWER)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-asymmetric", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    trials = 60 if args.quick else args.trials
    n_boot = 400 if args.quick else args.n_boot

    sweep = run_soft_null_kappa_sweep(
        n=args.n,
        n_trials=trials,
        n_boot=n_boot,
        seed=args.seed,
        max_workers=args.workers,
    )
    path = out / "soft_null_kappa.json"
    path.write_text(json.dumps(sweep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)
    proof = sweep["proof_of_cause"]
    print(f"proof_of_cause confirmed={proof['confirmed']} FPR={proof['observed_fpr']}", flush=True)
    print(proof["methods_sentence"], flush=True)
    op = sweep["operating"]
    print(
        f"operating κ={op['kappa']} FPR={op['null_fpr']:.3f} "
        f"power={op['power']:.3f}",
        flush=True,
    )
    print(op["reason"], flush=True)
    for k, row in sweep["by_kappa"].items():
        print(
            f"  κ={k:>3}: FPR={row['null_fpr']:.3f}±{row['null_fpr_se']:.3f}  "
            f"power={row['power']:.3f}",
            flush=True,
        )

    if not proof["confirmed"]:
        print("PROOF FAILED — refusing to refresh power_asymmetric soft cells.", file=sys.stderr)
        return 2

    # Pin operating κ into the module constant via the chosen value for the
    # asymmetric refresh (import already bound; pass via monkeypatch of cells).
    if not args.skip_asymmetric:
        # Use the sweep's operating κ for the soft branch.
        import jevbench.power as power_mod

        power_mod.SOFT_NULL_OPERATING_KAPPA = int(op["kappa"])
        asym = run_asymmetric_power(
            n=args.n,
            n_trials=min(trials, 400) if args.quick else 400,
            n_boot=n_boot,
            seed=args.seed,
        )
        asym["soft_null_kappa_ref"] = "results/soft_null_kappa.json"
        asym["operating_kappa_from_sweep"] = op
        path = out / "power_asymmetric.json"
        path.write_text(json.dumps(asym, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
        print(
            f"power soft={asym['power_soft']:.3f} hard={asym['power_hard']:.3f} "
            f"null_fpr soft={asym['null_fpr_soft']:.3f} hard={asym['null_fpr_hard']:.3f}"
        )

    if not sweep["acceptance"]["power_ok_at_operating"]:
        print(
            "Power < 0.80 at operating κ — amend detectable effect size before any Jev call.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
