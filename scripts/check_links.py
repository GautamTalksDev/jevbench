#!/usr/bin/env python3
"""Fail if a relative Markdown link or image does not resolve to a file."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "node_modules"}
LINK = re.compile(r"!\[[^\]]*\]\(([^)]+)\)|\[[^\]]+\]\(([^)]+)\)")


def main() -> int:
    missing: list[str] = []
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in SKIP_PARTS for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        fence = False
        for line in text.splitlines():
            if line.lstrip().startswith("```"):
                fence = not fence
                continue
            if fence:
                continue
            for match in LINK.finditer(line):
                raw = match.group(1) or match.group(2)
                target = raw.strip().split()[0]
                if target.startswith(("#", "http://", "https://", "mailto:")):
                    continue
                target = target.split("#", 1)[0]
                if not target:
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    missing.append(f"{path.relative_to(ROOT)} -> {raw}")
    if missing:
        print("link check failed", file=sys.stderr)
        print("\n".join(missing), file=sys.stderr)
        return 1
    print("link check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
