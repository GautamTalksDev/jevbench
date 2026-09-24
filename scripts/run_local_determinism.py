#!/usr/bin/env python3
"""Amendment 11 — local-baseline determinism check.

Greedy logprob / classifier scoring on CPU should be deterministic. Run 50
mid-entropy (or analysis) items × 3 repeats for each local client. If every
repeat matches exactly, record PASS and do not waste R=10 on locals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.chaosnli import hydrate_dataset  # noqa: E402
from jevbench.clients.base import SystemOneRequest  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.experiment import load_experiment  # noqa: E402
from jevbench.prereg import datasets_root  # noqa: E402
from jevbench.runner import build_client  # noqa: E402


def _fingerprint(decisions: list) -> str:
    parts = []
    for d in sorted(decisions, key=lambda x: x.question_key):
        if d.error:
            parts.append(f"{d.question_key}:ERR:{d.error}")
            continue
        if d.probabilities:
            probs = {k: round(float(v), 8) for k, v in sorted(d.probabilities.items())}
            parts.append(f"{d.question_key}:P:{probs}")
        else:
            parts.append(f"{d.question_key}:V:{round(float(d.value), 8)}")
    return "|".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument(
        "--clients",
        default="prefill_qwen15,gliclass,bart_mnli_ref",
        help="Comma-separated client names from exp1 YAML",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "local_determinism.json",
    )
    args = p.parse_args()

    spec = load_experiment(ROOT, "experiments/exp1_difficulty_calibration.yaml")
    dataset = load_dataset(datasets_root(ROOT), spec.task)
    if spec.task == "chaosnli":
        hydrate_dataset(dataset, dataset.root / "cache" / "text.jsonl")

    # Prefer mid-entropy excluded items (no analysis contamination)
    mid_ids = []
    uni = ROOT / "datasets" / "chaosnli" / "universe.jsonl"
    with uni.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stratum") == "mid" and not row.get("in_analysis"):
                mid_ids.append(str(row["id"]))
    rng = np.random.default_rng(args.seed)
    mid_ids = sorted(mid_ids)
    pick = rng.choice(len(mid_ids), size=min(args.n_items, len(mid_ids)), replace=False)
    item_ids = [mid_ids[int(i)] for i in pick]

    questions = spec.question_map()
    wanted = {c.strip() for c in args.clients.split(",") if c.strip()}
    report: dict = {
        "schema": "jevbench.local_determinism.v1",
        "n_items": len(item_ids),
        "repeats": args.repeats,
        "seed": args.seed,
        "item_ids": item_ids,
        "clients": {},
    }
    all_pass = True

    for cspec in spec.clients:
        if cspec.name not in wanted:
            continue
        if cspec.type not in ("prefill", "gliclass", "bart_mnli", "trivial"):
            continue
        print(f"determinism: {cspec.name} …", flush=True)
        client = build_client(cspec, experiment=spec)
        mismatches = 0
        checked = 0
        try:
            for iid in item_ids:
                if iid not in dataset.by_id:
                    # mid items are not in the analysis dataset — hydrate from cache
                    continue
            # Build minimal items from text cache for mid IDs
            text_by_id = {}
            with (ROOT / "datasets" / "chaosnli" / "cache" / "text.jsonl").open(
                encoding="utf-8"
            ) as f:
                for line in f:
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    text_by_id[str(obj["id"])] = obj

            for iid in item_ids:
                hit = text_by_id.get(iid)
                if hit is None:
                    continue
                state = {
                    "premise": hit["premise"],
                    "hypothesis": hit["hypothesis"],
                    "pair_id": iid,
                }
                fps = []
                for r in range(args.repeats):
                    dec = client.decide(
                        SystemOneRequest(
                            item_id=iid,
                            state=state,
                            questions=questions,
                            model=cspec.model or "",
                            pass_idx=r,
                        )
                    )
                    fps.append(_fingerprint(dec))
                checked += 1
                if len(set(fps)) != 1:
                    mismatches += 1
        finally:
            rel = getattr(client, "release", None)
            if callable(rel):
                rel()

        ok = mismatches == 0 and checked > 0
        all_pass = all_pass and ok
        report["clients"][cspec.name] = {
            "checked": checked,
            "mismatches": mismatches,
            "pass": ok,
            "type": cspec.type,
            "model": cspec.model,
            "role": cspec.role,
        }
        print(
            f"  checked={checked} mismatches={mismatches} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )

    report["all_pass"] = all_pass
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}", flush=True)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
