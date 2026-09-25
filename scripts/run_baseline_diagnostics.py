#!/usr/bin/env python3
"""Step 15 A — baseline diagnostics (report only; no client fixes)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.baseline_diagnostics import run_baseline_diagnostics  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402

import importlib.util  # noqa: E402


def _resolve_exp1_run(repo: Path, run_arg: str):
    spec = importlib.util.spec_from_file_location(
        "run_exp1_analyze", repo / "scripts" / "run_exp1_analyze.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.resolve_exp1_run(repo, run_arg)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", default="latest-exp1")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "baseline_diagnostics.json",
    )
    args = p.parse_args()

    run_id, raw_path, _man = _resolve_exp1_run(ROOT, args.run)
    ds = load_dataset(datasets_root(ROOT), "chaosnli")
    labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
    label_dist_by_id = {
        li.id: list(li.label.label_dist)
        for li in ds.by_id.values()
        if li.label.label_dist is not None
    }
    payload = run_baseline_diagnostics(
        raw_path=raw_path,
        items_path=ROOT / "datasets" / "chaosnli" / "items.jsonl",
        labels_by_id=labels_by_id,
        label_dist_by_id=label_dist_by_id,
    )
    payload["run_id"] = run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.out}")
    for name, arm in (payload.get("arms") or {}).items():
        print(
            f"  {name}: n={arm.get('n_primary')} "
            f"acc={arm.get('accuracy_vs_majority')} "
            f"mean_top={arm.get('mean_top_prob')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
