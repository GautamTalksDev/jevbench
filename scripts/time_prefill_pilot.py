#!/usr/bin/env python3
"""Time PrefillClient on 20 mid-entropy pilot items (never EXP-1 ids).

Reports seconds/item and peak RSS. Loads one model, then releases it.
"""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jevbench.clients.base import ChoiceQuestion, SystemOneRequest  # noqa: E402
from jevbench.clients.prefill import PrefillClient, PrefillClientConfig  # noqa: E402


def _peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux: ru_maxrss is kilobytes; macOS: bytes.
    rss = float(usage.ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def main() -> int:
    pilot = json.loads((ROOT / "pilot" / "items.json").read_text(encoding="utf-8"))
    ids = sorted(str(x) for x in pilot["pilot_ids"])[:20]
    # Guard: never touch EXP-1 analysis ids.
    exp1 = {
        json.loads(line)["id"]
        for line in (ROOT / "datasets" / "chaosnli" / "items.jsonl").open()
        if line.strip()
    }
    overlap = sorted(set(ids) & exp1)
    if overlap:
        print(f"refusing: pilot ids in EXP-1 set: {overlap[:5]}", file=sys.stderr)
        return 2

    cache: dict[str, dict] = {}
    with (ROOT / "datasets" / "chaosnli" / "cache" / "text.jsonl").open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in ids:
                cache[row["id"]] = row
    missing = [i for i in ids if i not in cache]
    if missing:
        print(f"cache missing {missing[:3]} — run fetch_chaosnli.py", file=sys.stderr)
        return 2

    questions = {
        "relation": ChoiceQuestion(
            instructions="entailment, neutral, or contradiction",
            criteria={
                "entailment": "entailed",
                "neutral": "neither",
                "contradiction": "contradicted",
            },
        )
    }
    client = PrefillClient(
        PrefillClientConfig(
            model="Qwen/Qwen2.5-1.5B-Instruct",
            backend="transformers",
            device="cpu",
        )
    )
    t0 = time.perf_counter()
    n = 0
    try:
        for iid in ids:
            text = cache[iid]
            state = {
                "pair_id": iid,
                "source": text.get("source", "mnli"),
                "text_status": "fetched",
                "premise": text["premise"],
                "hypothesis": text["hypothesis"],
            }
            req = SystemOneRequest(
                item_id=iid,
                state=state,
                questions=questions,
                model="Qwen/Qwen2.5-1.5B-Instruct",
                pass_idx=0,
            )
            client.decide(req)
            n += 1
    finally:
        client.release()
    elapsed = time.perf_counter() - t0
    s_per = elapsed / max(n, 1)
    peak = _peak_rss_mb()
    print(f"prefill_pilot_n={n}")
    print(f"prefill_s_per_item={s_per:.3f}")
    print(f"prefill_peak_rss_mb={peak:.1f}")
    print(f"prefill_dtype=float32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
