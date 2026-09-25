#!/usr/bin/env python3
"""Fail if the frozen preregistration text or locked dataset hashes drift.

preregistration.lock.json is itself frozen, so it cannot grow a new field.
The markdown digest below is the SHA-256 of PREREGISTRATION.md at the lock
this repository already published. Dataset hashes are read from the lock.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "preregistration.lock.json"
PREREG_PATH = ROOT / "PREREGISTRATION.md"

# SHA-256 of the frozen PREREGISTRATION.md. Do not edit that file to "fix" this.
EXPECTED_PREREG_SHA256 = "e19aecb696d780d16d05363b34d850b9a58282097cd84b0b207df7eb77146014"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    problems: list[str] = []
    if not PREREG_PATH.is_file():
        problems.append("PREREGISTRATION.md is missing")
    else:
        current = sha256_file(PREREG_PATH)
        if current != EXPECTED_PREREG_SHA256:
            problems.append(
                "PREREGISTRATION.md SHA-256 does not match the published digest\n"
                f"  expected: {EXPECTED_PREREG_SHA256}\n"
                f"  current:  {current}"
            )
    if not LOCK_PATH.is_file():
        problems.append("preregistration.lock.json is missing")
    else:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        recorded = lock.get("preregistration_md_sha256")
        if recorded and recorded != EXPECTED_PREREG_SHA256:
            problems.append(
                "preregistration.lock.json preregistration_md_sha256 "
                f"{recorded} != {EXPECTED_PREREG_SHA256}"
            )
        task = lock.get("task")
        pairs = (
            ("items_sha256", "items.jsonl"),
            ("labels_sha256", "labels.jsonl"),
        )
        for field, name in pairs:
            expected = lock.get(field)
            path = ROOT / "datasets" / str(task) / name
            if not expected:
                problems.append(f"lock is missing {field}")
                continue
            if not path.is_file():
                problems.append(f"missing {path}")
                continue
            got = sha256_file(path)
            if got != expected:
                problems.append(f"{path} hash {got} != lock {expected}")
    if problems:
        print("PREREGISTRATION INTEGRITY FAILED", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("preregistration integrity ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
