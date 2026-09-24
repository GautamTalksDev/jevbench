#!/usr/bin/env python3
"""PROMPT P — structural ECE bias diagnosis + parametric correction.

Offline. No network. No Jev data.

Writes results/soft_bias_correction.json
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
    diagnose_raw_soft_null_tails,
    run_soft_bias_correction,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=750)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-e0", type=int, default=2000)
    p.add_argument("--n-null-pval", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()

    trials = 80 if args.quick else args.trials
    n_boot = 400 if args.quick else args.n_boot
    n_e0 = 400 if args.quick else args.n_e0
    n_null = 400 if args.quick else args.n_null_pval

    print("=== PROMPT P diagnosis (raw, before correction) ===", flush=True)
    diagnosis = diagnose_raw_soft_null_tails(
        n=args.n,
        n_trials=trials,
        n_boot=n_boot,
        kappa=20.0,
        seed=args.seed,
    )
    upper = float(diagnosis["null_fpr_upper"])
    lower = float(diagnosis["null_fpr_lower"])
    favours = upper > lower
    text = (
        f"At κ=20, raw null FPR={diagnosis['null_fpr']:.3f} splits into "
        f"upper-tail={upper:.3f} and lower-tail={lower:.3f}. "
        + (
            "Mostly upper-tail: structural ECE bias favours H1."
            if favours
            else "Not mostly upper-tail."
        )
        + f" Null mean raw ΔECE={diagnosis['mean_raw_delta_ece']:.4f}."
    )
    print(text, flush=True)

    print("=== Corrected FPR / power / κ sensitivity ===", flush=True)
    out = run_soft_bias_correction(
        n=args.n,
        n_trials=trials,
        n_boot=n_boot,
        n_e0=n_e0,
        n_null_pval=n_null,
        seed=args.seed,
        max_workers=args.workers,
        diagnosis=diagnosis,
    )
    for k, row in out["by_kappa"].items():
        print(
            f"  κ={k:>3}: raw_null_mean={row['null_mean_raw_delta_ece']:.4f}  "
            f"bias0={row['null_mean_bias0_delta_ece']:.4f}  "
            f"corr_null_mean={row['null_mean_corrected_delta_ece']:.4f}  "
            f"FPR_p={row['null_fpr_pvalue']:.3f}  "
            f"FPR_ci={row['null_fpr_corrected']:.3f}  "
            f"power_p={row['power_pvalue']:.3f}",
            flush=True,
        )
    op = out["operating"]
    print(
        f"operating κ={op['kappa']}: FPR_p={op['null_fpr_pvalue']:.3f}  "
        f"FPR_ci={op['null_fpr_ci']:.3f}  "
        f"power_p={op['power_pvalue']:.3f}  "
        f"null_mean raw={op['null_mean_raw_delta_ece']:.4f} "
        f"corrected={op['null_mean_corrected_delta_ece']:.4f}",
        flush=True,
    )
    print(
        f"sensitivity verdict_kappa_dependent="
        f"{out['sensitivity']['verdict_kappa_dependent']}",
        flush=True,
    )
    acc = out["acceptance"]
    print(
        f"acceptance fpr_band_ok={acc['fpr_band_ok']} power_ok={acc['power_ok']}",
        flush=True,
    )

    path = args.out_dir / "soft_bias_correction.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}", flush=True)

    if not acc["fpr_band_ok"]:
        print(
            "FAIL: corrected FPR outside [0.03, 0.08] — do not proceed to EXP-1.",
            file=sys.stderr,
        )
        return 2
    if not acc["power_ok"]:
        print(
            "FAIL: corrected power < 0.80 — amend detectable effect before Jev.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
