"""Budget guard unit tests (Amendment 10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jevbench.budget import (
    GLOBAL_CAP_USD,
    BudgetExceeded,
    BudgetLedger,
    classify_run_kind,
    cost_for_call,
)


def test_classify_run_kind() -> None:
    assert classify_run_kind("pilot_prompt_s") == "pilot"
    assert classify_run_kind("exp1_difficulty_calibration") == "exp1"


def test_local_clients_free() -> None:
    assert cost_for_call(
        snapshot_date="2026-09-19",
        client_type="prefill",
        model="Qwen/Qwen2.5-1.5B-Instruct",
        input_tokens=1000,
        output_tokens=10,
    ) == 0.0


def test_ledger_caps(tmp_path: Path) -> None:
    # Point ledger at tmp by constructing path
    led = BudgetLedger(path=tmp_path / "budget.json")
    assert led.global_cap_usd == GLOBAL_CAP_USD
    led.assert_can_start(kind="pilot", projected_usd=0.04, run_id="t")
    with pytest.raises(BudgetExceeded):
        led.assert_can_start(kind="pilot", projected_usd=0.10, run_id="t")
    led.record(
        run_id="r1",
        run_kind="pilot",
        client="jev",
        model="jev-1.13.0",
        item_id="x",
        input_tokens=1000,
        output_tokens=0,
        cost_usd=0.04,
    )
    led.check_after_call(run_id="r1", kind="pilot")
    led.record(
        run_id="r1",
        run_kind="pilot",
        client="jev",
        model="jev-1.13.0",
        item_id="y",
        input_tokens=1000,
        output_tokens=0,
        cost_usd=0.02,
    )
    with pytest.raises(BudgetExceeded):
        led.check_after_call(run_id="r1", kind="pilot")
