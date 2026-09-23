#!/usr/bin/env python3
"""Compute Cohen's kappa on labels.jsonl vs labels_pass2.jsonl.

Exit 2 if kappa < floor (default 0.6). Exit 3 if no paired labels yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.agreement import (  # noqa: E402
    compute_agreement,
    measured_label_noise,
    select_agreement_subset,
    write_agreement_report,
)
from jevbench.dataset import read_jsonl  # noqa: E402
from jevbench.power import (  # noqa: E402
    PESSIMISTIC_DIFFICULTY_SD,
    POWER_THRESHOLD,
    choose_n,
    run_power_curve,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="support_tickets")
    p.add_argument("--kappa-floor", type=float, default=0.6)
    p.add_argument("--fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument(
        "--recheck-power",
        action="store_true",
        help="Re-run F1 power at measured label_noise; print whether n still suffices",
    )
    p.add_argument("--power-trials", type=int, default=200)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "label_agreement.json")
    args = p.parse_args()

    task_dir = ROOT / "datasets" / args.task
    primary = read_jsonl(task_dir / "labels.jsonl")
    subset = select_agreement_subset(
        [str(r["id"]) for r in primary], fraction=args.fraction, seed=args.seed
    )
    subset_path = task_dir / "agreement_subset_ids.json"
    subset_path.write_text(
        json.dumps(
            {
                "fraction": args.fraction,
                "seed": args.seed,
                "ids": subset,
                "n": len(subset),
                "instruction": (
                    "Second labeler: run tools/label.py --pass 2 --labeler <other> "
                    f"--id <id> for each id (or label the queue filtered to these ids)."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"agreement subset ({len(subset)} / {len(primary)}): wrote {subset_path}")

    result = compute_agreement(task_dir, kappa_floor=args.kappa_floor)
    write_agreement_report(result, args.out)
    print(
        f"kappa={result.kappa}  n_paired={result.n_paired}  "
        f"disagreement_rate={result.disagreement_rate}  "
        f"passes_floor={result.passes_floor}"
    )
    print(f"wrote {args.out}")

    if result.n_paired == 0:
        print(
            "No pass-2 overlap yet. Label the subset with a different --labeler "
            "via tools/label.py --pass 2, then re-run."
        )
        return 3

    if not result.passes_floor:
        print(
            f"FAIL: kappa {result.kappa:.3f} < {args.kappa_floor}. "
            "Tighten LABEL_GUIDE.md, re-label, do not proceed to paid runs."
        )
        return 2

    if args.recheck_power:
        noise = measured_label_noise(result)
        print(f"Feeding measured label_noise={noise:.4f} into F1 power sweep…")
        power = run_power_curve(
            n_trials=args.power_trials,
            seed=args.seed,
            null=False,
            label_noise=float(noise),
            difficulty_sd=PESSIMISTIC_DIFFICULTY_SD,
            tier_leakage=0.0,
            max_workers=8,
        )
        chosen = choose_n(power["curve"], POWER_THRESHOLD)
        print(
            f"At measured noise, smallest n with power>={POWER_THRESHOLD}: {chosen}"
        )
        for row in power["curve"]:
            print(
                f"  n={row['n_per_stratum']:4d}  power={row['rate_exclude_zero']:.3f}"
            )
        # Compare to locked operating n
        locked = None
        pa = ROOT / "results" / "power_analysis.json"
        if pa.is_file():
            locked = json.loads(pa.read_text(encoding="utf-8")).get(
                "chosen_n_per_stratum"
            )
        if locked is not None and chosen is not None and chosen > locked:
            print(
                f"WARNING: measured noise requires n={chosen} > locked n={locked}. "
                "Raise sample size or do not claim the prior power."
            )
            return 4
        print(f"OK: measured-noise required n={chosen} ≤ locked n={locked}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
