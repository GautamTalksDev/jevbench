#!/usr/bin/env python3
"""Post-hoc EXP-1 pass sensitivity → results/exp1_sensitivity_passes.json.

Label: NOT PREREGISTERED — robustness only.
Does not modify results/exp1_jev_only.json or any primary analysis output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp1_sensitivity import (  # noqa: E402
    build_sensitivity_payload,
    write_sensitivity,
)
from jevbench.prereg import datasets_root  # noqa: E402


def _resolve_exp1_run(run_arg: str):
    path = ROOT / "scripts" / "run_exp1_analyze.py"
    spec = importlib.util.spec_from_file_location("run_exp1_analyze", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_exp1_run(ROOT, run_arg)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        required=True,
        help="EXP-1 run_id or latest-exp1 (same resolver as analyze-exp1)",
    )
    p.add_argument("--client", default="jev")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "exp1_sensitivity_passes.json",
    )
    p.add_argument("--n-null-pval", type=int, default=2000)
    args = p.parse_args()

    try:
        run_id, raw_path, _man = _resolve_exp1_run(args.run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    primary = ROOT / "results" / "exp1_jev_only.json"
    primary_hash_before = None
    if primary.is_file():
        primary_hash_before = hashlib.sha256(primary.read_bytes()).hexdigest()

    ds = load_dataset(datasets_root(ROOT), "chaosnli")
    labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
    label_dist_by_id = {
        li.id: list(li.label.label_dist)
        for li in ds.by_id.values()
        if li.label.label_dist is not None
    }

    print(f"run_id={run_id}")
    print("label=NOT PREREGISTERED — robustness only")
    payload = build_sensitivity_payload(
        raw_path=raw_path,
        run_id=run_id,
        labels_by_id=labels_by_id,
        label_dist_by_id=label_dist_by_id,
        client=args.client,
        n_null_pval=args.n_null_pval,
    )
    out = write_sensitivity(payload, args.out)

    if primary_hash_before is not None:
        after = hashlib.sha256(primary.read_bytes()).hexdigest()
        if after != primary_hash_before:
            print(
                "ERROR: results/exp1_jev_only.json changed during sensitivity run",
                file=sys.stderr,
            )
            return 3
        print("exp1_jev_only.json unchanged (sha256 match)")

    for arm, block in (payload.get("arms") or {}).items():
        vals = block.get("delta_corrected_all") or []
        print(
            f"{arm}: min={block.get('delta_corrected_min'):.6f} "
            f"max={block.get('delta_corrected_max'):.6f} "
            f"n={len(vals)} values={vals}"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
