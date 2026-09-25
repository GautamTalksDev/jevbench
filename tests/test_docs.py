"""Documentation style and link checks."""

from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_prose_dashes() -> None:
    ns = runpy.run_path(str(ROOT / "scripts" / "check_prose_dashes.py"))
    assert ns["main"]() == 0


def test_internal_links() -> None:
    ns = runpy.run_path(str(ROOT / "scripts" / "check_links.py"))
    assert ns["main"]() == 0
