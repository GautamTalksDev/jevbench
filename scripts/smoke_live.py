#!/usr/bin/env python3
"""One-off live smoke: 2 mid-entropy pilot items + paraphrase preflight dry-run.

Never touches EXP-1 primary or paraphrase ids. Never prints sentence text or the
API key. Artifacts land under runs/smoke_<UTC>/ (gitignored via runs/*).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevbench.budget import BudgetExceeded, BudgetLedger, classify_run_kind  # noqa: E402
from jevbench.chaosnli import hydrate_dataset  # noqa: E402
from jevbench.dataset import load_dataset  # noqa: E402
from jevbench.envload import load_dotenv, require_typesafe_api_key  # noqa: E402
from jevbench.request_canon import build_request_body  # noqa: E402
from jevbench.runner import (  # noqa: E402
    Runner,
    RunnerConfig,
    assert_hydrated_inputs_safe,
)

SMOKE_HARD_CAP_USD = 0.001
SMOKE_TASK = "smoke_pilot"
SMOKE_EXP = "smoke_live"
MAX_PROJECTED_TOKENS_PER_CALL = 800  # abort if estimate implies more


def exp1_item_ids(items_path: Path) -> set[str]:
    """Primary ids, paraphrase row ids, and paraphrase source ids."""
    out: set[str] = set()
    with items_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out.add(str(row["id"]))
            state = row.get("state") or {}
            if isinstance(state, dict) and state.get("source_item_id"):
                out.add(str(state["source_item_id"]))
    return out


def select_live_ids(pilot_path: Path, *, n: int = 2) -> list[str]:
    data = json.loads(pilot_path.read_text(encoding="utf-8"))
    ids = sorted(str(x) for x in data["pilot_ids"])
    if len(ids) < n:
        raise RuntimeError(f"pilot/items.json has only {len(ids)} ids; need {n}")
    return ids[:n]


def assert_ids_excluded_from_exp1(ids: list[str], items_path: Path) -> None:
    forbidden = exp1_item_ids(items_path)
    overlap = sorted(set(ids) & forbidden)
    if overlap:
        raise RuntimeError(
            "Smoke refused: id(s) are in the EXP-1 primary or paraphrase set: "
            + ", ".join(overlap)
        )


def _universe_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


def _cache_by_id(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["id"])] = row
    return out


def write_smoke_task(repo_root: Path, live_ids: list[str]) -> Path:
    """Write a 2-item task for the runner. Text already marked fetched."""
    task = repo_root / "datasets" / SMOKE_TASK
    task.mkdir(parents=True, exist_ok=True)
    (task / "LABEL_GUIDE.md").write_text("# smoke pilot\n", encoding="utf-8")
    (task / "DISPUTED.md").write_text("# none\n", encoding="utf-8")

    universe = _universe_by_id(repo_root / "datasets" / "chaosnli" / "universe.jsonl")
    cache = _cache_by_id(repo_root / "datasets" / "chaosnli" / "cache" / "text.jsonl")
    items: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for iid in live_ids:
        if iid not in universe:
            raise RuntimeError(f"pilot id {iid} missing from universe.jsonl")
        if iid not in cache:
            raise RuntimeError(
                f"pilot id {iid} missing from cache/text.jsonl — "
                "run scripts/fetch_chaosnli.py"
            )
        uni = universe[iid]
        text = cache[iid]
        items.append(
            {
                "id": iid,
                "role": "primary",
                "tier": "easy",  # tier unused for this smoke; mid items not in EXP-1
                "state": {
                    "pair_id": iid,
                    "source": uni.get("source", text.get("source", "mnli")),
                    "text_status": "fetched",
                    "premise": text["premise"],
                    "hypothesis": text["hypothesis"],
                },
            }
        )
        labels.append(
            {
                "id": iid,
                "label": uni["majority_label"],
                "labeler": "chaosnli",
                "labeled_at": "2026-09-23T00:00:00+00:00",
                "label_dist": uni["label_dist"],
                "label_count": uni["label_count"],
            }
        )
    (task / "items.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items),
        encoding="utf-8",
    )
    (task / "labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels),
        encoding="utf-8",
    )
    return task


def write_smoke_experiment(repo_root: Path) -> Path:
    exp = {
        "name": SMOKE_EXP,
        "status": "ready",
        "model": "jev-1.13.0",
        "task": SMOKE_TASK,
        "serving_path": "native",
        "pricing_snapshot_date": "2026-09-19",
        "geography_note": "smoke live path check",
        "concurrency": 1,
        "repeats": 1,
        "description": "One-off live smoke on mid-entropy pilot ids. Not EXP-1.",
        "hypotheses": [
            {
                "id": "H_smoke",
                "statement": "Live path reaches Jev.",
                "falsified_when": "Call errors or wrong resolved model.",
            }
        ],
        "metrics": ["none"],
        "decision_rules": ["smoke only"],
        "sample_size": {"min_items": 1, "repeats": 1},
        "stopping_rule": "2 items then stop",
        "questions": {
            "relation": {
                "kind": "choice",
                "instructions": (
                    "Decide whether the hypothesis is entailed by the premise, "
                    "contradicted by the premise, or neither (neutral). Use only "
                    "the premise and the hypothesis in the state."
                ),
                "criteria": {
                    "entailment": "The hypothesis must be true if the premise is true.",
                    "neutral": "The hypothesis might be true, but the premise does not require it.",
                    "contradiction": "The hypothesis cannot be true if the premise is true.",
                },
            },
            "noul_entailment": {
                "kind": "noul",
                "instructions": (
                    "The hypothesis is entailed by the premise. Use only the "
                    "premise and hypothesis in the state."
                ),
            },
            "noul_neutral": {
                "kind": "noul",
                "instructions": (
                    "The hypothesis is neutral with respect to the premise. "
                    "Use only the premise and hypothesis in the state."
                ),
            },
            "noul_contradiction": {
                "kind": "noul",
                "instructions": (
                    "The hypothesis is contradicted by the premise. Use only "
                    "the premise and hypothesis in the state."
                ),
            },
        },
        "clients": [
            {"name": "jev", "type": "jev", "model": "jev-1.13.0", "serving_path": "native"}
        ],
    }
    path = repo_root / "experiments" / f"{SMOKE_EXP}.yaml"
    path.write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")
    return path


def projected_smoke_usd() -> float:
    # Same table as cost.py snapshot 2026-09-19: $0.042 / MTok input, $0 output.
    return 2 * MAX_PROJECTED_TOKENS_PER_CALL / 1_000_000 * 0.042


def run_live(repo_root: Path, out_dir: Path, live_ids: list[str]) -> dict[str, Any]:
    write_smoke_task(repo_root, live_ids)
    write_smoke_experiment(repo_root)

    projected = projected_smoke_usd()
    if projected > SMOKE_HARD_CAP_USD + 1e-12:
        raise RuntimeError(
            f"Smoke projected ${projected:.6f} exceeds hard cap "
            f"${SMOKE_HARD_CAP_USD:.3f}"
        )
    ledger = BudgetLedger.load(repo_root)
    kind = classify_run_kind(SMOKE_EXP)
    ledger.assert_can_start(
        kind=kind, projected_usd=min(projected, SMOKE_HARD_CAP_USD), run_id=out_dir.name
    )

    rdir = Runner(
        RunnerConfig(
            repo_root=repo_root,
            experiment=SMOKE_EXP,
            repeats=1,
            concurrency=1,
            confirm=True,
            progress=False,
            skip_lock_check=True,  # smoke task is not the EXP-1 lock
            skip_budget_guard=False,
            geography_note="smoke_live",
        )
    ).run()
    if not isinstance(rdir, Path):
        raise RuntimeError("smoke live expected a run directory, got dry-run estimate")

    rows = []
    with (rdir / "raw.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != 2:
        raise RuntimeError(f"expected 2 live calls, got {len(rows)}")

    summary = []
    for row in rows:
        decisions = row.get("decisions") or []
        resolved = None
        latency = None
        in_tok = int((row.get("usage") or {}).get("input_tokens") or row.get("input_tokens") or 0)
        out_tok = int((row.get("usage") or {}).get("output_tokens") or row.get("output_tokens") or 0)
        for d in decisions:
            if isinstance(d, dict):
                resolved = d.get("resolved_model") or resolved
                if d.get("latency_ms") is not None:
                    latency = float(d["latency_ms"])
                in_tok = max(in_tok, int(d.get("input_tokens") or 0))
                out_tok = max(out_tok, int(d.get("output_tokens") or 0))
        cost = float(row.get("est_cost_usd") or 0.0)
        if cost > SMOKE_HARD_CAP_USD:
            raise RuntimeError(
                f"single-call cost ${cost:.6f} exceeded smoke hard cap "
                f"${SMOKE_HARD_CAP_USD:.3f}"
            )
        summary.append(
            {
                "item_id": row.get("item_id"),
                "resolved_model": resolved,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": latency,
                "est_cost_usd": cost,
            }
        )
    total_cost = sum(float(s["est_cost_usd"]) for s in summary)
    if total_cost > SMOKE_HARD_CAP_USD + 1e-12:
        raise RuntimeError(
            f"smoke total ${total_cost:.6f} exceeded hard cap ${SMOKE_HARD_CAP_USD:.3f}"
        )
    (out_dir / "live_summary.json").write_text(
        json.dumps({"calls": summary, "run_dir": str(rdir)}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"calls": summary, "run_dir": rdir}


def run_paraphrase_dry(repo_root: Path) -> dict[str, Any]:
    """Hydrate + preflight + build bodies. SEND NOTHING."""
    calls: list[Any] = []

    def factory(_spec: Any) -> Any:
        calls.append(_spec)
        raise RuntimeError("smoke dry-run must never construct a client")

    dataset = load_dataset(repo_root / "datasets", "chaosnli")
    hydrate_dataset(dataset, dataset.root / "cache" / "text.jsonl")
    from jevbench.experiment import load_experiment

    # Questions from EXP-1 yaml for realistic body shape; no client.
    spec = load_experiment(repo_root, "exp1_difficulty_calibration")
    assert_hydrated_inputs_safe(
        repo_root,
        dataset,
        questions=spec.question_map(),
        model=spec.model,
    )
    para = [li for li in dataset.by_id.values() if li.item.role == "paraphrase"]
    if len(para) != 100:
        raise RuntimeError(f"expected 100 paraphrase items, got {len(para)}")
    n_ok = 0
    for li in para:
        assert isinstance(li.item.state, dict)
        body = build_request_body(
            item_id=li.id,
            client="jev",
            model=spec.model,
            state=li.item.state,
            questions=spec.question_map(),
        )
        blob = json.dumps(body, ensure_ascii=False)
        for token in (
            "text_status",
            "local_paraphrase_required",
            "fetch_required",
            "frozen_before_any_model_run",
        ):
            if token in blob:
                raise RuntimeError(f"dry body for {li.id} contains {token}")
        n_ok += 1
    if calls:
        raise RuntimeError(f"client factory was called {len(calls)} time(s) during dry-run")
    # Touch factory binding so the assert above is the contract; never invoke Runner.
    _ = factory
    return {"n_safe": n_ok, "n_total": 100, "client_factory_calls": len(calls)}


def main() -> int:
    load_dotenv(ROOT / ".env")
    require_typesafe_api_key()  # clear error if missing; value never printed

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "runs" / f"smoke_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    live_ids = select_live_ids(ROOT / "pilot" / "items.json", n=2)
    assert_ids_excluded_from_exp1(live_ids, ROOT / "datasets" / "chaosnli" / "items.jsonl")

    print(f"smoke_dir={out_dir.name}")
    print(f"live_ids={live_ids[0]},{live_ids[1]}")

    try:
        live = run_live(ROOT, out_dir, live_ids)
        for call in live["calls"]:
            print(
                f"live item_id={call['item_id']} "
                f"resolved_model={call['resolved_model']} "
                f"input_tokens={call['input_tokens']} "
                f"output_tokens={call['output_tokens']} "
                f"latency_ms={call['latency_ms']} "
                f"est_cost_usd={call['est_cost_usd']:.8f}"
            )

        dry = run_paraphrase_dry(ROOT)
        print(f"paraphrase dry-run {dry['n_safe']}/{dry['n_total']} safe")

        ledger = BudgetLedger.load(ROOT)
        print(f"budget_ledger_spent_usd={ledger.spent_usd():.8f}")
        print(f"budget_ledger_remaining_usd={ledger.remaining_global():.8f}")
        (out_dir / "smoke_report.json").write_text(
            json.dumps(
                {
                    "live_ids": live_ids,
                    "live": live["calls"],
                    "paraphrase_dry": dry,
                    "budget_spent_usd": ledger.spent_usd(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except (BudgetExceeded, RuntimeError, FileNotFoundError) as exc:
        print(f"SMOKE FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
