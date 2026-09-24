#!/usr/bin/env python3
"""Fail if cited commit hashes are missing from this repository.

Scans PREREGISTRATION.md, METHODS_NOTES.md, RELATED_WORK.md, docs/, and
paper/ for git-like hex digests (7–40 chars). Skips 64-char SHA-256 file
digests and hashes that appear only inside third-party GitHub URLs.

Usage:
  .venv/bin/python scripts/verify_citations.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN_FILES = [
    ROOT / "PREREGISTRATION.md",
    ROOT / "paper" / "METHODS_NOTES.md",
    ROOT / "paper" / "RELATED_WORK.md",
]
SCAN_GLOBS = [
    "docs/**/*.md",
    "paper/**/*.md",
]

# Backtick digests, AMENDMENT_*=hex, and bare 40-char hashes on their own.
HASH_RE = re.compile(
    r"(?:`([0-9a-f]{7,40})`)|(?:(?:AMENDMENT(?:_\d+)?_COMMIT|prereg_commit)[=:\"'\s]+([0-9a-f]{7,40}))",
    re.IGNORECASE,
)
# Also catch table cells with bare hashes (no backticks) of length 7–40
# when surrounded by | or whitespace — but not 64-char sha256.
BARE_RE = re.compile(r"(?<![0-9a-f])([0-9a-f]{7,40})(?![0-9a-f])", re.IGNORECASE)

EXTERNAL_URL_RE = re.compile(
    r"https?://github\.com/(?!GautamTalksDev/jevbench)[^\s)]+",
    re.IGNORECASE,
)


def _git_exists(digest: str) -> bool:
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{digest}^{{commit}}"],
        cwd=ROOT,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True
    # Also allow annotated tags / any object that resolves
    proc2 = subprocess.run(
        ["git", "rev-parse", "--verify", digest],
        cwd=ROOT,
        capture_output=True,
    )
    return proc2.returncode == 0


def _collect_paths() -> list[Path]:
    paths: list[Path] = []
    for p in SCAN_FILES:
        if p.is_file():
            paths.append(p)
    for pattern in SCAN_GLOBS:
        paths.extend(ROOT.glob(pattern))
    # de-dupe
    return sorted({p.resolve() for p in paths if p.is_file()})


def _external_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in EXTERNAL_URL_RE.finditer(text)]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in spans)


def main() -> int:
    missing: list[tuple[str, str, str]] = []  # file, digest, context
    checked: set[str] = set()

    for path in _collect_paths():
        text = path.read_text(encoding="utf-8")
        ext = _external_spans(text)
        found: list[tuple[int, str]] = []

        # HISTORY_REWRITE.md intentionally lists pre-rewrite (absent) hashes
        # in the Old column — only verify the New column.
        if path.name == "HISTORY_REWRITE.md":
            for line in text.splitlines():
                if not line.startswith("|") or line.startswith("|---") or "Subject" in line:
                    continue
                ticks = re.findall(r"`([0-9a-f]{7,40})`", line, re.I)
                if len(ticks) >= 2:
                    # last digest on the row is the post-rewrite hash
                    found.append((0, ticks[-1].lower()))
                elif len(ticks) == 1 and "unchanged" in line.lower():
                    found.append((0, ticks[0].lower()))
            # Also check prose hashes outside the table (none expected)
        else:
            for m in HASH_RE.finditer(text):
                dig = m.group(1) or m.group(2)
                if dig:
                    found.append((m.start(), dig.lower()))
            for m in BARE_RE.finditer(text):
                dig = m.group(1).lower()
                if len(dig) == 64:
                    continue
                window = text[max(0, m.start() - 80) : m.end() + 40].lower()
                if not any(
                    k in window
                    for k in ("commit", "amendment", "prereg", "history_rewrite")
                ):
                    continue
                found.append((m.start(), dig))

        for pos, dig in found:
            if len(dig) == 64:
                continue
            # Pure decimal strings (e.g. seed `20260924`) are not git hashes.
            if len(dig) < 40 and not re.search(r"[a-f]", dig):
                continue
            if _in_spans(pos, ext):
                continue
            if dig in checked:
                continue
            checked.add(dig)
            if not _git_exists(dig):
                rel = path.relative_to(ROOT)
                ctx = text[max(0, pos - 30) : pos + 50].replace("\n", " ")
                missing.append((str(rel), dig, ctx))

    print(f"checked {len(checked)} unique commit-like digests")
    if missing:
        print("MISSING from this repository:")
        for rel, dig, ctx in missing:
            print(f"  {rel}: {dig}")
            print(f"    …{ctx}…")
        return 1
    print("all cited commit hashes resolve in this repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
