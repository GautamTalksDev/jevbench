"""smoke_live must refuse EXP-1 analysis ids."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_smoke():
    path = ROOT / "scripts" / "smoke_live.py"
    spec = importlib.util.spec_from_file_location("smoke_live", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_live_refuses_exp1_id(tmp_path: Path) -> None:
    smoke = _load_smoke()
    items = tmp_path / "items.jsonl"
    items.write_text(
        json.dumps(
            {
                "id": "exp1-primary-id",
                "role": "primary",
                "tier": "easy",
                "state": {"pair_id": "exp1-primary-id", "text_status": "fetch_required"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "id": "src1::paraphrase",
                "role": "paraphrase",
                "tier": "easy",
                "state": {
                    "source_item_id": "src1",
                    "text_status": "local_paraphrase_required",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="EXP-1"):
        smoke.assert_ids_excluded_from_exp1(["exp1-primary-id"], items)
    with pytest.raises(RuntimeError, match="EXP-1"):
        smoke.assert_ids_excluded_from_exp1(["src1"], items)
    smoke.assert_ids_excluded_from_exp1(["109443n"], items)


def test_select_live_ids_are_sorted_prefix(tmp_path: Path) -> None:
    smoke = _load_smoke()
    pilot = tmp_path / "items.json"
    pilot.write_text(
        json.dumps({"pilot_ids": ["c", "a", "b", "d"]}),
        encoding="utf-8",
    )
    assert smoke.select_live_ids(pilot, n=2) == ["a", "b"]
