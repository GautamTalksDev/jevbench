#!/usr/bin/env python3
"""PROMPT S / R — Pilot. First paid Jev calls + local baselines. NOT analysed for ΔECE.

ITEM GUARD: mid-entropy only. Budget cap $0.05. Baselines: prefill Qwen2.5-1.5B + GLiClass.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.budget import (  # noqa: E402
    BudgetExceeded,
    BudgetLedger,
    cost_for_call,
)
from jevbench.clients.base import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    RetryPolicy,
    SystemOneRequest,
)
from jevbench.clients.gliclass import GLiClassClient, GLiClassClientConfig  # noqa: E402
from jevbench.clients.jev import JevClient, JevClientConfig  # noqa: E402
from jevbench.clients.prefill import PrefillClient, PrefillClientConfig  # noqa: E402
from jevbench.clients.trivial import (  # noqa: E402
    TrivialClient,
    TrivialClientConfig,
    TrivialQuestionSpec,
)
from jevbench.cost import DEFAULT_PILOT  # noqa: E402
from jevbench.exp1 import NOUL_SUM_FLOOR, normalize_noul_triplet  # noqa: E402
from jevbench.request_canon import (  # noqa: E402
    build_request_body,
    request_body_sha256,
    strip_row_for_public,
)

PILOT_SEED = 20260924
PILOT_N = 20
ANALYSIS_N = 750
JEV_MODEL = "jev-1.13.0"
PREFILL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
GLICLASS_MODEL = "knowledgator/gliclass-base-v1.0"
PILOT_CAP_USD = 0.05
SNAPSHOT = "2026-09-19"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'").strip('"')
        if k and k not in os.environ:
            os.environ[k] = v


def analysis_ids(repo: Path) -> set[str]:
    ids: set[str] = set()
    with (repo / "datasets" / "chaosnli" / "items.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("role", "primary") == "primary":
                ids.add(str(row["id"]))
    if len(ids) != 2 * ANALYSIS_N:
        raise RuntimeError(f"expected {2 * ANALYSIS_N} analysis ids, got {len(ids)}")
    return ids


def mid_pool(repo: Path, forbidden: set[str]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    with (repo / "datasets" / "chaosnli" / "universe.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stratum") != "mid":
                continue
            iid = str(row["id"])
            if iid in forbidden or row.get("in_analysis"):
                raise RuntimeError(f"ITEM GUARD FAIL: {iid}")
            pool.append(row)
    return pool


def sample_pilot_ids(pool: list[dict[str, Any]], *, n: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    ordered = sorted(pool, key=lambda r: r["id"])
    idx = rng.choice(len(ordered), size=n, replace=False)
    return sorted(ordered[int(i)]["id"] for i in idx)


def load_text_cache(repo: Path) -> dict[str, dict[str, str]]:
    path = repo / "datasets" / "chaosnli" / "cache" / "text.jsonl"
    by_id: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            by_id[str(obj["id"])] = obj
    return by_id


def questions_from_yaml(spec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, q in spec["questions"].items():
        if q["kind"] == "choice":
            out[key] = ChoiceQuestion(
                instructions=q["instructions"], criteria=dict(q["criteria"])
            )
        elif q["kind"] == "noul":
            out[key] = NoulQuestion(instructions=q["instructions"])
    return out


def network_floor_ms(host: str = "https://api.typesafe.ai", n: int = 5) -> dict[str, float]:
    times: list[float] = []
    with httpx.Client(timeout=30.0) as client:
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                client.get(host)
            except Exception:  # noqa: BLE001
                pass
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "p50_ms": float(statistics.median(times)),
        "p95_ms": float(times[max(0, int(np.ceil(0.95 * len(times)) - 1))]),
    }


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    arr = sorted(xs)
    return float(arr[max(0, min(len(arr) - 1, int(np.ceil(q * len(arr)) - 1)))])


def summarize_decisions(decisions: list[Any]) -> dict[str, Any]:
    rows = [d.to_dict() if hasattr(d, "to_dict") else dict(d) for d in decisions]
    choice = next((r for r in rows if r.get("kind") == "choice"), None)
    nouls = [r for r in rows if r.get("kind") == "noul"]
    noul_pack = normalize_noul_triplet(rows)
    return {
        "rows": rows,
        "resolved_model": (choice or (rows[0] if rows else {})).get("resolved_model"),
        "choice_probs": None if choice is None else choice.get("probabilities"),
        "noul_has_confidence": any(r.get("confidence") is not None for r in nouls),
        "noul_pack": noul_pack,
        "errors": [r["error"] for r in rows if r.get("error")],
        "latency_ms": max((float(r.get("latency_ms") or 0.0) for r in rows), default=0.0),
        "input_tokens": max((int(r.get("input_tokens") or 0) for r in rows), default=0),
        "output_tokens": max((int(r.get("output_tokens") or 0) for r in rows), default=0),
    }


def adapter_note_removed() -> str:
    return (
        "Amendment 10: paid adapter baselines removed. Prefill probabilities "
        "are logprob-derived (TransformersPrefillBackend / single constrained "
        "token). GLiClass scores are classifier logits→softmax, not verbalised."
    )


def write_report(path: Path, report: dict[str, Any]) -> None:
    c = report["checks"]
    lines = [
        "# Pilot REPORT — PROMPT S / R",
        "",
        f"**Run id:** `{report['run_id']}`  ",
        f"**UTC:** {report['utc']}  ",
        f"**Pinned Jev:** `{JEV_MODEL}` native `api.typesafe.ai`  ",
        f"**Local baselines:** prefill `{PREFILL_MODEL}`, GLiClass `{GLICLASS_MODEL}`  ",
        f"**Items:** {report['n_items']} mid-entropy only (seed `{PILOT_SEED}`)  ",
        f"**Outputs analysed for ΔECE:** NO  ",
        f"**Credit expiry (fill from console):** {report.get('credit_expiry', 'NOT RECORDED — check TypeSafe dashboard')}  ",
        f"**Console balance Δ vs ledger:** {report.get('balance_check', 'pending — compare manually')}",
        "",
        "## Item guard",
        "",
        f"- Pilot ∩ analysis: **{report['pilot_analysis_overlap']}** (must be 0)",
        f"- Assert passed: **{report['item_guard_ok']}**",
        "",
        "```",
        ", ".join(report["pilot_ids"]),
        "```",
        "",
        "## Budget",
        "",
        f"- Pilot cap: **${PILOT_CAP_USD:.2f}**",
        f"- Ledger spent this run: **${report['ledger_spent']:.6f}**",
        f"- Global remaining: **${report['ledger_remaining']:.4f}**",
        "",
        "## Checks",
        "",
        "### 1. Resolved `model` on every Jev row",
        f"- OK: **{c['1_resolved_model']['ok']}** — `{c['1_resolved_model']['values']}`",
        "",
        "### 2. Choice vs Noul shapes",
        f"- Choice all 3 labels: **{c['2_shapes']['choice_all_three']}**",
        f"- Noul, no confidence: **{c['2_shapes']['noul_no_confidence']}**",
        "",
        "### 3. Noul near-zero sum",
        f"- Floor `{NOUL_SUM_FLOOR:g}`; triggers: **{c['3_noul_floor']['triggers']}** / {c['3_noul_floor']['noul_calls']}",
        "",
        "### 4. Probability source",
        f"- {c['4_probability_source']}",
        "",
        "### 5. Tokens vs cost.py",
        f"- Jev mean in/out: **{c['5_tokens']['jev_mean_in']:.1f} / {c['5_tokens']['jev_mean_out']:.1f}**",
        f"- Prior DEFAULT_PILOT jev: {DEFAULT_PILOT['jev']}",
        f"- Defaults updated: **{c['5_tokens']['defaults_updated']}**",
        "",
        "### 6. Latency",
        f"- Jev p50/p95: **{c['6_latency']['jev_p50']:.0f} / {c['6_latency']['jev_p95']:.0f}** ms",
        f"- Prefill p50/p95: **{c['6_latency']['prefill_p50']:.0f} / {c['6_latency']['prefill_p95']:.0f}** ms",
        f"- GLiClass p50/p95: **{c['6_latency']['gliclass_p50']:.0f} / {c['6_latency']['gliclass_p95']:.0f}** ms",
        f"- Network floor api.typesafe.ai p50/p95: **{c['6_latency']['network_p50']:.0f} / {c['6_latency']['network_p95']:.0f}** ms",
        "",
        "### 7. Errors (max 3 retries; never 4xx)",
        f"- `{c['7_errors']}`",
        "",
        "### 8. Baseline model pins",
        f"- Prefill resolved: `{c['8_baselines']['prefill_resolved']}`",
        f"- GLiClass resolved: `{c['8_baselines']['gliclass_resolved']}`",
        "",
        "## Blockers",
        "",
    ]
    if report.get("blockers"):
        lines.extend(f"- {b}" for b in report["blockers"])
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Amendment 10 gate",
            "",
            "Code changes after this pilot prompted by Jev output → cite this "
            "pilot run id and mid-entropy-only items.",
            "",
            f"Raw: `{report['raw_path']}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=PILOT_N)
    p.add_argument("--seed", type=int, default=PILOT_SEED)
    p.add_argument("--skip-prefill", action="store_true", help="Skip HF download (guard+Jev only)")
    p.add_argument("--credit-expiry", type=str, default="", help="From TypeSafe console")
    p.add_argument("--console-balance-delta", type=str, default="", help="USD change observed in console")
    args = p.parse_args()

    _load_dotenv(ROOT / ".env")
    # Amendment 10: refuse paid LLM keys in this process
    for bad in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if os.environ.pop(bad, None):
            print(f"stripped {bad} from environment (Amendment 10)", flush=True)

    out_dir = ROOT / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_dir = out_dir / "full"
    stripped_dir = out_dir / "stripped"
    full_dir.mkdir(parents=True, exist_ok=True)
    stripped_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("pilot_%Y%m%dT%H%M%SZ")
    raw_path = full_dir / f"{run_id}.jsonl"
    stripped_path = stripped_dir / f"{run_id}.jsonl"

    forbidden = analysis_ids(ROOT)
    pool = mid_pool(ROOT, forbidden)
    pilot_ids = sample_pilot_ids(pool, n=args.n, seed=args.seed)
    overlap = sorted(set(pilot_ids) & forbidden)
    if overlap:
        raise RuntimeError(f"ITEM GUARD FAIL: {overlap}")

    (out_dir / "items.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "pilot_ids": pilot_ids,
                "n_analysis": len(forbidden),
                "n_mid_pool": len(pool),
                "overlap": overlap,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    blockers: list[str] = []
    if not os.environ.get("TYPESAFE_API_KEY"):
        blockers.append("TYPESAFE_API_KEY missing — put it in .env (gitignored), never in chat")

    text = load_text_cache(ROOT)
    if any(i not in text for i in pilot_ids):
        blockers.append("text cache missing some pilot ids — run scripts/fetch_chaosnli.py")

    spec = yaml.safe_load(
        (ROOT / "experiments" / "exp1_difficulty_calibration.yaml").read_text(encoding="utf-8")
    )
    questions = questions_from_yaml(spec)

    pilot_yaml = {
        "name": "pilot_prompt_s",
        "status": "ready",
        "model": JEV_MODEL,
        "serving_path": "native",
        "clients": [
            {"name": "jev", "type": "jev", "model": JEV_MODEL},
            {
                "name": "prefill_qwen15",
                "type": "prefill",
                "model": PREFILL_MODEL,
                "prefill_backend": "transformers",
            },
            {"name": "gliclass", "type": "gliclass", "model": GLICLASS_MODEL},
        ],
        "noul_sum_floor": NOUL_SUM_FLOOR,
        "budget_cap_usd": PILOT_CAP_USD,
    }
    (ROOT / "experiments" / "pilot_prompt_r.yaml").write_text(
        yaml.safe_dump(pilot_yaml, sort_keys=False), encoding="utf-8"
    )

    ledger = BudgetLedger.load(ROOT)
    # Projected: 20 Jev calls × ~600 tokens × $0.042/MTok
    projected = 20 * 600 / 1_000_000 * 0.042
    try:
        ledger.assert_can_start(kind="pilot", projected_usd=projected, run_id=run_id)
    except BudgetExceeded as exc:
        blockers.append(str(exc))

    no_retry_4xx = RetryPolicy(max_attempts=4, retry_4xx=False, retry_429=False)
    clients: dict[str, Any] = {}
    if os.environ.get("TYPESAFE_API_KEY") and not blockers:
        clients["jev"] = JevClient(
            JevClientConfig(model=JEV_MODEL, serving_path="native", retry=no_retry_4xx)
        )
    elif os.environ.get("TYPESAFE_API_KEY"):
        # Still allow Jev if only budget soft-warn — but BudgetExceeded is hard
        if not any("budget" in b.lower() or "cap" in b.lower() for b in blockers):
            clients["jev"] = JevClient(
                JevClientConfig(model=JEV_MODEL, serving_path="native", retry=no_retry_4xx)
            )

    if not args.skip_prefill:
        try:
            clients["prefill_qwen15"] = PrefillClient(
                PrefillClientConfig(
                    model=PREFILL_MODEL, backend="transformers", device="cpu"
                )
            )
        except Exception as exc:  # noqa: BLE001
            blockers.append(f"prefill init failed: {exc}")
    try:
        clients["gliclass"] = GLiClassClient(
            GLiClassClientConfig(model_id=GLICLASS_MODEL, device="cpu")
        )
    except Exception as exc:  # noqa: BLE001
        # Fall back to trivial so the pilot can still exercise shapes
        blockers.append(f"gliclass unavailable ({exc}); using trivial floor")
        clients["trivial"] = TrivialClient(
            TrivialClientConfig(
                specs={
                    "relation": TrivialQuestionSpec(majority_class="neutral"),
                    "noul_entailment": TrivialQuestionSpec(noul_default=0.33),
                    "noul_neutral": TrivialQuestionSpec(noul_default=0.34),
                    "noul_contradiction": TrivialQuestionSpec(noul_default=0.33),
                }
            )
        )

    net = network_floor_ms()
    jev_models: list[str] = []
    prefill_models: list[str] = []
    gliclass_models: list[str] = []
    lat_jev: list[float] = []
    lat_prefill: list[float] = []
    lat_gli: list[float] = []
    tok_in: list[float] = []
    tok_out: list[float] = []
    choice_ok = choice_n = noul_ok = noul_n = floor_triggers = noul_calls = 0
    jev_model_ok = True
    err_classes: Counter[str] = Counter()
    run_spent = 0.0

    # Only proceed with live Jev if key present
    if "jev" not in clients and not os.environ.get("TYPESAFE_API_KEY"):
        pass

    with raw_path.open("w", encoding="utf-8") as raw_f:
        for iid in pilot_ids:
            hit = text.get(iid)
            if hit is None:
                err_classes["missing_text"] += 1
                continue
            state = {
                "premise": hit["premise"],
                "hypothesis": hit["hypothesis"],
                "pair_id": iid,
                "source": hit.get("source", ""),
            }
            for cname in ("jev", "prefill_qwen15", "gliclass", "trivial"):
                if cname not in clients:
                    continue
                client = clients[cname]
                model = {
                    "jev": JEV_MODEL,
                    "prefill_qwen15": PREFILL_MODEL,
                    "gliclass": GLICLASS_MODEL,
                    "trivial": "trivial",
                }[cname]
                ctype = {
                    "jev": "jev",
                    "prefill_qwen15": "prefill",
                    "gliclass": "gliclass",
                    "trivial": "trivial",
                }[cname]
                try:
                    decisions = client.decide(
                        SystemOneRequest(
                            item_id=iid,
                            state=state,
                            questions=questions,
                            model=model,
                            pass_idx=0,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    err_classes[type(exc).__name__] += 1
                    raw_f.write(
                        json.dumps(
                            {
                                "item_id": iid,
                                "client": cname,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        + "\n"
                    )
                    continue
                summary = summarize_decisions(decisions)
                for e in summary["errors"]:
                    err_classes[(e or "error").split(":", 1)[0]] += 1
                cost = cost_for_call(
                    snapshot_date=SNAPSHOT,
                    client_type=ctype,
                    model=model,
                    input_tokens=summary["input_tokens"],
                    output_tokens=summary["output_tokens"],
                )
                try:
                    if ctype == "jev":
                        ledger.record(
                            run_id=run_id,
                            run_kind="pilot",
                            client=cname,
                            model=model,
                            item_id=iid,
                            input_tokens=summary["input_tokens"],
                            output_tokens=summary["output_tokens"],
                            cost_usd=cost,
                        )
                        ledger.check_after_call(run_id=run_id, kind="pilot")
                        run_spent = ledger.spent_for_run(run_id)
                except BudgetExceeded as exc:
                    err_classes["BudgetExceeded"] += 1
                    blockers.append(str(exc))
                    raw_f.write(json.dumps({"item_id": iid, "stopped": str(exc)}) + "\n")
                    break

                raw_f.write(
                    json.dumps(
                        {
                            "item_id": iid,
                            "client": cname,
                            "model_requested": model,
                            "summary": {
                                k: summary[k]
                                for k in (
                                    "resolved_model",
                                    "choice_probs",
                                    "noul_pack",
                                    "errors",
                                    "latency_ms",
                                    "input_tokens",
                                    "output_tokens",
                                )
                            },
                            "decisions": summary["rows"],
                            "est_cost_usd": cost,
                            "stratum": "mid",
                            "outputs_analysed": False,
                        }
                    )
                    + "\n"
                )
                raw_f.flush()

                if cname == "jev" and not summary["errors"]:
                    rm = summary["resolved_model"]
                    if not rm:
                        jev_model_ok = False
                    else:
                        jev_models.append(str(rm))
                    lat_jev.append(summary["latency_ms"])
                    tok_in.append(summary["input_tokens"])
                    tok_out.append(summary["output_tokens"])
                    if summary["choice_probs"] is not None:
                        choice_n += 1
                        if set(summary["choice_probs"]) >= {
                            "entailment",
                            "neutral",
                            "contradiction",
                        }:
                            choice_ok += 1
                    if summary["noul_pack"] is not None:
                        noul_calls += 1
                        noul_n += 1
                        if not summary["noul_has_confidence"]:
                            noul_ok += 1
                        if summary["noul_pack"]["near_zero_sum"]:
                            floor_triggers += 1
                elif cname == "prefill_qwen15":
                    if summary["resolved_model"]:
                        prefill_models.append(str(summary["resolved_model"]))
                    lat_prefill.append(summary["latency_ms"])
                elif cname in ("gliclass", "trivial"):
                    if summary["resolved_model"]:
                        gliclass_models.append(str(summary["resolved_model"]))
                    lat_gli.append(summary["latency_ms"])
            else:
                continue
            break  # budget stop

    # Enrich full log with request bodies; write stripped public twin.
    if raw_path.is_file():
        enriched: list[dict[str, Any]] = []
        stripped_rows: list[dict[str, Any]] = []
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            iid = str(row.get("item_id") or "")
            hit = text.get(iid)
            if hit is None:
                enriched.append(row)
                continue
            state = {
                "premise": hit["premise"],
                "hypothesis": hit["hypothesis"],
                "pair_id": iid,
                "source": hit.get("source", ""),
            }
            body = build_request_body(
                item_id=iid,
                client=str(row.get("client") or ""),
                model=str(row.get("model_requested") or ""),
                state=state,
                questions=questions,
            )
            digest = request_body_sha256(body)
            full_row = dict(row)
            full_row["request_body"] = body
            full_row["request_body_sha256"] = digest
            enriched.append(full_row)
            stripped_rows.append(
                strip_row_for_public(row, request_body_sha256_hex=digest)
            )
        raw_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in enriched),
            encoding="utf-8",
        )
        stripped_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in stripped_rows),
            encoding="utf-8",
        )

    defaults_updated = False
    if tok_in:
        mean_in, mean_out = float(np.mean(tok_in)), float(np.mean(tok_out))
        cost_path = ROOT / "jevbench" / "cost.py"
        text_c = cost_path.read_text(encoding="utf-8")
        old = (
            '    "jev": {"input_tokens": 180.0, "output_tokens": 24.0},'
        )
        # Also match previously updated values
        import re

        new = f'    "jev": {{"input_tokens": {mean_in:.1f}, "output_tokens": {mean_out:.1f}}},'
        text2, nsub = re.subn(
            r'    "jev": \{"input_tokens": [0-9.]+, "output_tokens": [0-9.]+\},',
            new,
            text_c,
            count=1,
        )
        if nsub:
            # annotate pilot source nearby if default block comment exists
            text2 = text2.replace(
                "# Default pilot means (from a notional 20-call structured-output pilot).",
                f"# Default pilot means — measured {run_id} (mid-entropy PROMPT S).",
            )
            cost_path.write_text(text2, encoding="utf-8")
            defaults_updated = True
        (out_dir / "measured_tokens.json").write_text(
            json.dumps(
                {"jev": {"input_tokens": mean_in, "output_tokens": mean_out}, "run_id": run_id},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731
    balance_check = "pending — compare TypeSafe console Δ to ledger"
    if args.console_balance_delta:
        try:
            console_d = abs(float(args.console_balance_delta))
            led = max(run_spent, 1e-12)
            pct_m = abs(console_d - led) / led * 100.0
            balance_check = (
                f"console Δ=${console_d:.6f} vs ledger ${led:.6f} "
                f"({pct_m:.1f}% mismatch)"
                + (" — STOP (>20%)" if pct_m > 20 else " — OK")
            )
            if pct_m > 20:
                blockers.append(balance_check)
        except ValueError:
            balance_check = f"unparseable console delta {args.console_balance_delta!r}"

    report = {
        "run_id": run_id,
        "utc": datetime.now(UTC).isoformat(),
        "n_items": len(pilot_ids),
        "pilot_ids": pilot_ids,
        "pilot_analysis_overlap": len(overlap),
        "item_guard_ok": len(overlap) == 0,
        "raw_path": str(stripped_path.relative_to(ROOT)),
        "raw_full_path": str(raw_path.relative_to(ROOT)),
        "blockers": blockers,
        "ledger_spent": run_spent,
        "ledger_remaining": ledger.remaining_global(),
        "credit_expiry": args.credit_expiry or "NOT RECORDED — check TypeSafe dashboard → credit details",
        "balance_check": balance_check,
        "checks": {
            "1_resolved_model": {
                "ok": jev_model_ok and bool(jev_models),
                "values": sorted(set(jev_models)),
            },
            "2_shapes": {
                "choice_all_three": choice_n > 0 and choice_ok == choice_n,
                "noul_no_confidence": noul_n > 0 and noul_ok == noul_n,
            },
            "3_noul_floor": {"triggers": floor_triggers, "noul_calls": noul_calls},
            "4_probability_source": adapter_note_removed(),
            "5_tokens": {
                "jev_mean_in": mean(tok_in),
                "jev_mean_out": mean(tok_out),
                "defaults_updated": defaults_updated,
            },
            "6_latency": {
                "jev_p50": pct(lat_jev, 0.5),
                "jev_p95": pct(lat_jev, 0.95),
                "prefill_p50": pct(lat_prefill, 0.5),
                "prefill_p95": pct(lat_prefill, 0.95),
                "gliclass_p50": pct(lat_gli, 0.5),
                "gliclass_p95": pct(lat_gli, 0.95),
                "network_p50": net["p50_ms"],
                "network_p95": net["p95_ms"],
            },
            "7_errors": dict(err_classes),
            "8_baselines": {
                "prefill_resolved": sorted(set(prefill_models)),
                "gliclass_resolved": sorted(set(gliclass_models)),
            },
        },
    }
    (out_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_report(out_dir / "REPORT.md", report)
    print(f"wrote {out_dir / 'REPORT.md'}", flush=True)
    if blockers:
        print("BLOCKERS:", *blockers, sep="\n  ", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
