#!/usr/bin/env python3
"""Run EXP-1 analysis → results/exp1.json (+ verdict).

Requires ``--run <run_id>`` or ``--run latest-exp1``. Never defaults to the
offline demo fixture for EXP-1 outputs.

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
from jevbench.exp1 import (  # noqa: E402
    EASY_TIERS,
    HARD_TIERS,
    load_raw_records,
    no_verdict_reason,
    normalize_records,
    run_exp1_analysis,
)
from jevbench.experiment import load_experiment  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402
from jevbench.runner import (  # noqa: E402
    load_completed_keys,
    resolve_client_specs,
    work_units,
)

EXPECTED_STRATUM_N = 750


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
    """Raise RuntimeError if required client rows are missing."""
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


def assert_exp1_manifest(manifest: dict[str, Any], *, run_id: str) -> None:
    """Refuse non-ChaosNLI / unscored demo sources for EXP-1 outputs."""
    task = manifest.get("task")
    if task != "chaosnli":
        raise RuntimeError(
            f"refusing run {run_id!r}: manifest.task={task!r} (need 'chaosnli'). "
            "analyze-exp1 never falls back to offline_fixture / support_tickets."
        )
    if manifest.get("scored") is False:
        raise RuntimeError(
            f"refusing run {run_id!r}: manifest.scored is false "
            "(demo / unscored source — not an EXP-1 analysis input)."
        )


def list_exp1_run_dirs(runs_root: Path) -> list[Path]:
    """Candidate EXP-1 run dirs (excludes ``_aborted_*``)."""
    if not runs_root.is_dir():
        return []
    out: list[Path] = []
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith("_aborted_"):
            continue
        if not d.name.startswith("exp1_"):
            continue
        if not (d / "manifest.json").is_file() or not (d / "raw.jsonl").is_file():
            continue
        out.append(d)
    return out


def resolve_exp1_run(repo_root: Path, run_arg: str) -> tuple[str, Path, dict[str, Any]]:
    """Resolve ``--run`` to (run_id, raw.jsonl path, manifest).

    ``latest-exp1`` → newest ``runs/exp1_*`` with task==chaosnli and scored!=false.
    """
    runs_root = repo_root / "runs"
    if run_arg == "latest-exp1":
        eligible: list[tuple[str, Path, dict[str, Any]]] = []
        for d in list_exp1_run_dirs(runs_root):
            man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            try:
                assert_exp1_manifest(man, run_id=d.name)
            except RuntimeError:
                continue
            eligible.append((d.name, d / "raw.jsonl", man))
        if not eligible:
            raise RuntimeError(
                "no eligible EXP-1 run under runs/exp1_* "
                "(need manifest.task=='chaosnli' and scored!=false; "
                "_aborted_* excluded)."
            )
        # Timestamped names sort lexicographically by UTC time.
        eligible.sort(key=lambda t: t[0], reverse=True)
        return eligible[0]

    run_dir = runs_root / run_arg
    if not run_dir.is_dir():
        raise RuntimeError(f"run dir not found: {run_dir}")
    if run_arg.startswith("_aborted_"):
        raise RuntimeError(f"refusing aborted run dir: {run_arg}")
    man_path = run_dir / "manifest.json"
    raw_path = run_dir / "raw.jsonl"
    if not man_path.is_file() or not raw_path.is_file():
        raise RuntimeError(f"run {run_arg!r} missing manifest.json or raw.jsonl")
    man = json.loads(man_path.read_text(encoding="utf-8"))
    assert_exp1_manifest(man, run_id=run_arg)
    return run_arg, raw_path, man


def primary_strata_sizes(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Unique primary-role item counts in hard vs easy analysis strata."""
    hard_ids: set[str] = set()
    easy_ids: set[str] = set()
    for r in rows:
        if r.get("role", "primary") == "paraphrase":
            continue
        iid = str(r.get("source_item_id") or r.get("item_id"))
        tier = r.get("tier")
        if tier in HARD_TIERS:
            hard_ids.add(iid)
        elif tier in EASY_TIERS:
            easy_ids.add(iid)
    return {"hard": len(hard_ids), "easy": len(easy_ids)}


def infer_scoring_primary(rows: list[dict[str, Any]]) -> str:
    primary = [
        r
        for r in rows
        if r.get("role", "primary") != "paraphrase"
        and isinstance(r.get("probabilities"), dict)
    ]
    if primary and all(
        isinstance(r.get("label_dist"), (list, tuple)) for r in primary
    ):
        return "soft"
    return "hard"


def preflight_print_and_check(
    *,
    run_id: str,
    raw_path: Path,
    rows: list[dict[str, Any]],
    n_records: int,
) -> str:
    """Print run summary; refuse if primary strata ≠ 750/750. Return scoring_primary."""
    strata = primary_strata_sizes(rows)
    n_items = strata["hard"] + strata["easy"]
    scoring = infer_scoring_primary(rows)
    print(f"run_id={run_id}")
    print(f"n_records={n_records}")
    print(f"n_items={n_items}")
    print(f"strata sizes: hard={strata['hard']} easy={strata['easy']}")
    print(f"scoring_primary={scoring}")
    if strata["hard"] != EXPECTED_STRATUM_N or strata["easy"] != EXPECTED_STRATUM_N:
        raise RuntimeError(
            f"refusing analysis: primary-role strata must be "
            f"{EXPECTED_STRATUM_N}/{EXPECTED_STRATUM_N} (hard/easy); "
            f"got hard={strata['hard']} easy={strata['easy']}."
        )
    return scoring


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        required=True,
        help=(
            "EXP-1 run_id under runs/, or 'latest-exp1' for the newest "
            "runs/exp1_* with task=chaosnli and scored!=false "
            "(excludes _aborted_*). Required — no offline_fixture default."
        ),
    )
    p.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Optional raw.jsonl override (tests only). Still requires --run for id.",
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
        help="Skip the YAML completeness gate (tests only).",
    )
    p.add_argument(
        "--skip-strata-check",
        action="store_true",
        help="Skip the 750/750 primary-strata gate (tests only).",
    )
    args = p.parse_args()

    try:
        run_id, raw_resolved, manifest = resolve_exp1_run(ROOT, args.run)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    raw = args.raw if args.raw is not None else raw_resolved
    if not raw.is_file():
        print(f"missing raw.jsonl: {raw}", file=sys.stderr)
        return 2

    # Belt-and-suspenders: never write EXP-1 outputs from the demo fixture path.
    if "offline_fixture" in str(raw.resolve()):
        print(
            "refusing: raw path is offline_fixture — not a valid EXP-1 analysis source",
            file=sys.stderr,
        )
        return 2

    n_boot = 500 if args.quick else args.n_boot

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

    if not args.skip_completeness:
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
    sample = json.loads(raw.read_text(encoding="utf-8").splitlines()[0])
    spec = load_experiment(ROOT, args.experiment)
    task = args.task or spec.task
    if task != "chaosnli":
        print(
            f"refusing: experiment/task={task!r} (EXP-1 analysis requires chaosnli)",
            file=sys.stderr,
        )
        return 2
    if "label" not in sample:
        ds = load_dataset(datasets_root(ROOT), task)
        labels_by_id = {li.id: str(li.label.label) for li in ds.by_id.values()}
        label_dist_by_id = {
            li.id: list(li.label.label_dist)
            for li in ds.by_id.values()
            if li.label.label_dist is not None
        }

    raw_recs = load_raw_records(raw)
    rows = normalize_records(
        raw_recs, labels_by_id=labels_by_id, label_dist_by_id=label_dist_by_id or None
    )
    try:
        if not args.skip_strata_check:
            preflight_print_and_check(
                run_id=run_id,
                raw_path=raw,
                rows=rows,
                n_records=len(raw_recs),
            )
        else:
            strata = primary_strata_sizes(rows)
            print(f"run_id={run_id}")
            print(f"n_records={len(raw_recs)}")
            print(f"n_items={strata['hard'] + strata['easy']}")
            print(f"strata sizes: hard={strata['hard']} easy={strata['easy']}")
            print(f"scoring_primary={infer_scoring_primary(rows)}")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

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

    scored_flag = bool(args.scored) or (manifest.get("scored") is True)

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
        scored=scored_flag,
        labelling_ceiling_applies=False,  # chaosnli only
        contamination_arms=contamination,
    )
    payload["source"] = dict(payload.get("source") or {})
    payload["source"]["run_id"] = run_id
    payload["source"]["manifest_task"] = manifest.get("task")
    payload["source"]["manifest_scored"] = manifest.get("scored")

    if interim:
        payload["status"] = "interim: baselines pending"
        payload["interim_clients"] = client_names
        if payload.get("verdict") is not None:
            payload["verdict"] = (
                "INTERIM (baselines pending). " + str(payload.get("verdict") or "")
            )

    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if payload.get("verdict") is None:
        reason = payload.get("no_verdict_reason") or no_verdict_reason(payload)
        print(f"NO VERDICT: {reason}")
    else:
        print(payload["verdict"])
    print(f"wrote {out_path}")
    if payload.get("gates", {}).get("block_reason") and scored_flag:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
