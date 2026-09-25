#!/usr/bin/env python3
"""Run EXP-1 analysis → results/exp1.json (+ verdict).

Default source: offline fixture (no network). Pass --raw for a live run dir.

Completeness: refuses a final results/exp1.json unless every client declared
in the experiment YAML has all items × its repeats in raw.jsonl. Pass
``--clients jev`` for an interim write to results/exp1_jev_only.json labelled
"interim: baselines pending".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.agreement import compute_agreement  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.exp1 import run_exp1_analysis  # noqa: E402
from jevbench.experiment import load_experiment  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402
from jevbench.runner import (  # noqa: E402
    load_completed_keys,
    resolve_client_specs,
    work_units,
)


def _expected_keys_for_clients(
    *,
    repo_root: Path,
    experiment: str,
    client_names: list[str] | None,
) -> tuple[set[str], list[str]]:
    spec = load_experiment(repo_root, experiment)
    selected = resolve_client_specs(spec, client_names)
    dataset = load_dataset(datasets_root(repo_root), spec.task)
    items = sorted(dataset.by_id.values(), key=lambda x: x.id)
    units = work_units(items, selected, spec.effective_repeats())
    return {u.key for u in units}, [c.name for c in selected]


def assert_clients_complete(
    raw_path: Path,
    *,
    repo_root: Path,
    experiment: str,
    client_names: list[str] | None,
) -> list[str]:
    """Raise SystemExit-style error string if required client rows are missing."""
    expected, names = _expected_keys_for_clients(
        repo_root=repo_root,
        experiment=experiment,
        client_names=client_names,
    )
    done = load_completed_keys(raw_path)
    missing = sorted(expected - done)
    if missing:
        sample = ", ".join(missing[:5])
        more = f" (and {len(missing) - 5} more)" if len(missing) > 5 else ""
        raise RuntimeError(
            f"raw.jsonl is incomplete for clients [{', '.join(names)}]: "
            f"missing {len(missing)} unit key(s), e.g. {sample}{more}. "
            "Resume the run for those clients, or pass --clients jev for an "
            "interim jev-only write."
        )
    return names


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "runs" / "offline_fixture" / "raw.jsonl",
        help="Path to raw.jsonl (fixture or runs/<id>/raw.jsonl)",
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--task", default=None, help="Override dataset task (default: from YAML)")
    p.add_argument("--jev-client", default="jev")
    p.add_argument("--baseline-client", default="adapter_baseline")
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument(
        "--quick",
        action="store_true",
        help="n_boot=500 for a smoke pass",
    )
    p.add_argument(
        "--scored",
        action="store_true",
        help="Mark as scored live run (enables underpowered gate in verdict)",
    )
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument(
        "--clients",
        default=None,
        help="Comma-separated client subset for interim analysis (e.g. jev)",
    )
    p.add_argument(
        "--experiment",
        default="exp1_difficulty_calibration",
        help="Experiment YAML that defines the full client set",
    )
    p.add_argument(
        "--skip-completeness",
        action="store_true",
        help="Skip the YAML completeness gate (offline fixture / tests only).",
    )
    args = p.parse_args()

    n_boot = 500 if args.quick else args.n_boot
    raw = args.raw
    if not raw.is_file():
        print(f"missing raw.jsonl: {raw}", file=sys.stderr)
        return 2

    client_names: list[str] | None = None
    if args.clients is not None:
        client_names = [c.strip() for c in args.clients.split(",") if c.strip()]
        if not client_names:
            print("--clients was empty", file=sys.stderr)
            return 2

    interim = client_names is not None
    if interim and client_names == ["jev"]:
        out_path = args.out or (ROOT / "results" / "exp1_jev_only.json")
    elif interim:
        joined = "_".join(client_names)
        out_path = args.out or (ROOT / "results" / f"exp1_{joined}_only.json")
    else:
        out_path = args.out or (ROOT / "results" / "exp1.json")

    if not args.skip_completeness and "offline_fixture" not in str(raw):
        try:
            assert_clients_complete(
                raw,
                repo_root=ROOT,
                experiment=args.experiment,
                client_names=client_names,
            )
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 3
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    labels_by_id = None
    label_dist_by_id = None
    # Live runner records need labels joined from the dataset
    sample = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    spec = load_experiment(ROOT, args.experiment)
    task = args.task or spec.task
    if "label" not in sample:
        ds = load_dataset(datasets_root(ROOT), task)
        labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
        label_dist_by_id = {
            li.id: list(li.label.label_dist)
            for li in ds.by_id.values()
            if li.label.label_dist is not None
        }

    agreement = compute_agreement(
        ROOT / "datasets" / task, kappa_floor=0.6
    ).to_dict()
    power = None
    pa = ROOT / "results" / "power_analysis.json"
    if pa.is_file():
        power = json.loads(pa.read_text(encoding="utf-8"))

    contamination = (
        list(client_names)
        if client_names is not None
        else [c.name for c in spec.clients]
    )

    payload = run_exp1_analysis(
        raw_path=raw,
        out_path=out_path,
        labels_by_id=labels_by_id,
        label_dist_by_id=label_dist_by_id or None,
        jev_client=args.jev_client,
        baseline_client=args.baseline_client,
        seed=args.seed,
        n_boot=n_boot,
        agreement=agreement,
        power=power,
        scored=args.scored,
        labelling_ceiling_applies=task != "chaosnli",
        contamination_arms=contamination,
    )
    if interim:
        payload["status"] = "interim: baselines pending"
        payload["interim_clients"] = client_names
        payload["verdict"] = (
            "INTERIM (baselines pending). "
            + str(payload.get("verdict") or "")
        )
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(payload["verdict"])
    print(f"wrote {out_path}")
    if payload.get("gates", {}).get("block_reason") and args.scored:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
