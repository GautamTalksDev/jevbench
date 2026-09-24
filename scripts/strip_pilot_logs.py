#!/usr/bin/env python3
"""Strip pilot raw logs for public commit.

Keeps responses, item IDs, and SHA-256 of the full request body.
Moves unstripped logs to ``pilot/full/`` (gitignored).
Writes stripped logs to ``pilot/stripped/`` (safe to commit).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jevbench.experiment import load_experiment  # noqa: E402
from jevbench.request_canon import (  # noqa: E402
    build_request_body,
    request_body_sha256,
    strip_row_for_public,
)


def load_text_cache(root: Path) -> dict[str, dict]:
    path = root / "datasets" / "chaosnli" / "cache" / "text.jsonl"
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[str(obj["id"])] = obj
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Full jsonl paths (default: pilot/*.jsonl excluding stripped/full)",
    )
    args = p.parse_args()

    full_dir = ROOT / "pilot" / "full"
    stripped_dir = ROOT / "pilot" / "stripped"
    full_dir.mkdir(parents=True, exist_ok=True)
    stripped_dir.mkdir(parents=True, exist_ok=True)

    inputs = list(args.inputs)
    if not inputs:
        inputs = sorted(
            q
            for q in (ROOT / "pilot").glob("*.jsonl")
            if q.is_file()
        )

    spec = load_experiment(ROOT, "experiments/exp1_difficulty_calibration.yaml")
    questions = spec.question_map()
    text = load_text_cache(ROOT)

    if not inputs:
        print("no pilot jsonl to strip", flush=True)
        return 0

    for src in inputs:
        src = src.resolve()
        rows_in = [
            json.loads(line)
            for line in src.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        out_rows: list[dict] = []
        for row in rows_in:
            iid = str(row["item_id"])
            hit = text.get(iid)
            if hit is None:
                raise SystemExit(f"missing text cache for {iid} — run fetch_chaosnli.py")
            state = {
                "premise": hit["premise"],
                "hypothesis": hit["hypothesis"],
                "pair_id": iid,
                "source": hit.get("source", ""),
            }
            model = str(row.get("model_requested") or "")
            client = str(row.get("client") or "")
            body = build_request_body(
                item_id=iid,
                client=client,
                model=model,
                state=state,
                questions=questions,
            )
            digest = request_body_sha256(body)
            # Preserve a full copy that still has whatever was logged, plus body.
            full_row = dict(row)
            full_row["request_body"] = body
            full_row["request_body_sha256"] = digest
            out_rows.append((full_row, strip_row_for_public(row, request_body_sha256_hex=digest)))

        dest_full = full_dir / src.name
        dest_stripped = stripped_dir / src.name
        with dest_full.open("w", encoding="utf-8") as ff, dest_stripped.open(
            "w", encoding="utf-8"
        ) as sf:
            for full_row, stripped in out_rows:
                ff.write(json.dumps(full_row, ensure_ascii=False) + "\n")
                sf.write(json.dumps(stripped, ensure_ascii=False) + "\n")

        # Remove original from pilot/ root if it lived there
        if src.parent == (ROOT / "pilot") and src.exists():
            src.unlink()
            print(f"moved {src.name} → pilot/full/ + pilot/stripped/", flush=True)
        else:
            print(f"wrote {dest_full} and {dest_stripped}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
