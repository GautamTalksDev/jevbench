"""Offline reproduce + checksum gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevbench.reproduce import FIXTURE_DIR, RESULTS, reproduce


def test_offline_fixture_exists():
    assert (FIXTURE_DIR / "raw.jsonl").is_file()
    assert (FIXTURE_DIR / "manifest.json").is_file()


def test_reproduce_and_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Smoke: full reproduce against repo fixture, then --check
    summary = reproduce(check=False)
    assert summary["elapsed_s"] < 120
    harness = RESULTS / "harness_fixture"
    assert (harness / "metrics.json").is_file()
    assert (harness / "finding.json").is_file()
    assert (RESULTS / "SHA256SUMS").is_file()
    money = harness / "charts" / "ece_vs_accuracy_by_tier_1080p.png"
    assert money.is_file()
    reproduce(check=True)
