#!/usr/bin/env python3
"""Fail CI on unhedged novelty language or suspicious uncited numeric claims.

This is a first-pass grep linter; tighten as paper/main.md grows.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

NOVELTY = re.compile(
    r"\b(nobody has|no one has|first ever|never been done|nothing exists)\b",
    re.I,
)
ARCHITECTURE_ASSERT = re.compile(
    r"\b(RLCD (is|was|means)|trained on|architecture is|based on (an )?open[- ]weight)\b",
    re.I,
)
# Bare percentages / ECE-like floats without a nearby citation marker — soft check
NUMBER_CLAIM = re.compile(r"\b(ECE|Brier|AUROC|accuracy)\b[^.!?\n]{0,40}\b0?\.\d+\b", re.I)
CITE = re.compile(r"(\[[^\]]+\]|\([A-Z][^\)]*\d{4}|run_id|raw\.jsonl|RELATED_WORK)")


def main(path: Path) -> int:
    if not path.exists():
        print(f"skip: {path} missing")
        return 0
    text = path.read_text(encoding="utf-8")
    failed = 0
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        if NOVELTY.search(line):
            print(f"{path}:{i}: unhedged novelty — cite RELATED_WORK.md / rephrase")
            print(f"  {line.strip()}")
            failed += 1
        if ARCHITECTURE_ASSERT.search(line):
            print(f"{path}:{i}: architecture/RLCD assertion — UNKNOWN is required")
            print(f"  {line.strip()}")
            failed += 1
        if NUMBER_CLAIM.search(line) and not CITE.search(line):
            print(f"{path}:{i}: metric number without nearby citation/run_id marker")
            print(f"  {line.strip()}")
            failed += 1
    if failed:
        print(f"\n{failed} claim-lint failure(s)")
        return 1
    print("claim lint OK")
    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.md")
    raise SystemExit(main(target))
