"""Cost model guards — never apply Jev rates to baselines."""

from __future__ import annotations

import pytest

from jevbench.cost import (
    assert_jev_rate_only_on_jev,
    estimate,
    resolve_price,
)


def test_adapter_requires_model():
    with pytest.raises(ValueError, match="explicit model"):
        resolve_price("2026-09-19", "adapter")


def test_jev_rate_not_applied_to_adapter():
    jev = resolve_price("2026-09-19", "jev")
    base = resolve_price("2026-09-19", "adapter", "openai/gpt-4o-mini")
    assert jev.model_id == "jev"
    assert base.model_id != "jev"
    assert base.input_usd_per_mtok != jev.input_usd_per_mtok
    assert_jev_rate_only_on_jev(jev, "jev")
    assert_jev_rate_only_on_jev(base, "adapter")
    with pytest.raises(AssertionError):
        assert_jev_rate_only_on_jev(jev, "adapter")


def test_estimate_split_baseline_dominates():
    """1200 adapter calls ≈ 600k in + output ≫ Jev at $0.042/Mtok."""
    split = estimate(
        snapshot_date="2026-09-19",
        n_items=400,  # 200 hard + 200 easy
        repeats=3,
        baseline_model="openai/gpt-4o-mini",
    )
    assert split.n_calls_jev == 1200
    assert split.n_calls_baseline == 1200
    # ~600k input tokens on baseline at $0.15/M + output
    assert split.baseline_usd > 0.05
    assert split.baseline_usd > split.jev_usd
    assert split.total_usd == pytest.approx(split.jev_usd + split.baseline_usd)
    assert "Jev $" in split.to_dict()["split_line"]
    # Must not look like the old ~$0.03 claim
    assert split.total_usd > 0.10


def test_jev_output_not_billed():
    jev = resolve_price("2026-09-19", "jev")
    assert jev.output_usd_per_mtok == 0.0
    assert jev.cost_usd(1_000_000, 1_000_000) == pytest.approx(0.042)
