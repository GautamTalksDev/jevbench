#!/usr/bin/env python3
"""Fail if non-frozen Markdown uses em dashes, en dashes, or spaced double hyphens."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {
    ROOT / "PREREGISTRATION.md",
    ROOT / "results" / "harness_fixture" / "README.md",
    # Historical records restored with original em/en dashes; do not rewrite.
    ROOT / "docs" / "HISTORY_REWRITE.md",
    ROOT / "pilot" / "REPORT.md",
}
SKIP_PARTS = {".git", ".venv", "node_modules"}
EM = "\u2014"
EN = "\u2013"
INLINE = re.compile(r"`[^`]*`")


def _prose_outside_code(text: str) -> list[tuple[int, str]]:
    lines_out: list[tuple[int, str]] = []
    fence = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        prose = INLINE.sub(" ", line)
        lines_out.append((lineno, prose))
    return lines_out


def main() -> int:
    problems: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        if path.resolve() in {p.resolve() for p in FROZEN}:
            continue
        for lineno, prose in _prose_outside_code(path.read_text(encoding="utf-8")):
            if EM in prose or EN in prose or " -- " in prose:
                rel = path.relative_to(ROOT)
                problems.append(f"{rel}:{lineno}: {prose.strip()}")
    if problems:
        print("dash check failed", file=sys.stderr)
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("dash check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
