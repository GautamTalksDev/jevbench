#!/usr/bin/env python3
"""Re-fetch ChaosNLI text, rebuild request bodies, verify stored SHA-256.

Reads ``pilot/stripped/*.jsonl`` (public). Uses local
``datasets/chaosnli/cache/text.jsonl`` (gitignored). Does not call Jev.
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
from jevbench.request_canon import build_request_body, request_body_sha256  # noqa: E402


def load_text_cache(root: Path) -> dict[str, dict]:
    path = root / "datasets" / "chaosnli" / "cache" / "text.jsonl"
    if not path.is_file():
        raise SystemExit(
            f"missing {path} — run: .venv/bin/python scripts/fetch_chaosnli.py"
        )
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
        "--stripped-dir",
        type=Path,
        default=ROOT / "pilot" / "stripped",
    )
    p.add_argument(
        "--fetch",
        action="store_true",
        help="Run scripts/fetch_chaosnli.py before verifying",
    )
    args = p.parse_args()

    if args.fetch:
        import subprocess

        rc = subprocess.call(
            [sys.executable, str(ROOT / "scripts" / "fetch_chaosnli.py")],
            cwd=str(ROOT),
        )
        if rc != 0:
            return rc

    text = load_text_cache(ROOT)
    spec = load_experiment(ROOT, "experiments/exp1_difficulty_calibration.yaml")
    questions = spec.question_map()

    files = sorted(args.stripped_dir.glob("*.jsonl"))
    if not files:
        print(f"no stripped logs in {args.stripped_dir}", flush=True)
        return 2

    n_ok = n_bad = n_rows = 0
    for path in files:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            iid = str(row["item_id"])
            expected = row.get("request_body_sha256")
            if not expected:
                print(f"FAIL {path.name}:{line_no} missing request_body_sha256")
                n_bad += 1
                continue
            hit = text.get(iid)
            if hit is None:
                print(f"FAIL {path.name}:{line_no} no cache text for {iid}")
                n_bad += 1
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
            got = request_body_sha256(body)
            if got != expected:
                print(
                    f"FAIL {path.name}:{line_no} item={iid} "
                    f"client={row.get('client')} expected={expected[:12]}… got={got[:12]}…"
                )
                n_bad += 1
            else:
                n_ok += 1

    print(f"verified ok={n_ok} bad={n_bad} rows={n_rows} files={len(files)}")
    return 0 if n_bad == 0 and n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
