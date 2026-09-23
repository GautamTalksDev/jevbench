"""Tests for video chart exports — tokens, sizes, footer, transparent."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jevbench.charts import (
    cost_per_1000_correct,
    ece_vs_accuracy_by_tier,
    invariant_scatter,
    latency_distribution,
    load_tokens,
    make_context,
    reliability_diagram,
    selective_automation_curve,
    three_arm_comparison,
)
from jevbench.metrics import expected_calibration_error


@pytest.fixture
def tokens_path() -> Path:
    return Path(__file__).resolve().parents[1] / "arena" / "tokens.json"


def test_tokens_json_valid_and_colourblind_fields(tokens_path: Path):
    data = load_tokens(tokens_path)
    assert data["typography"]["min_px"] >= 28
    assert "1080p" in data["export"]["sizes"]
    assert "1440p" in data["export"]["sizes"]
    assert data["export"]["sizes"]["1080p"] == [1920, 1080]
    assert data["export"]["sizes"]["1440p"] == [2560, 1440]
    for s in data["series"]:
        assert "marker" in s and "dash" in s and "color" in s
    for t in data["tiers"]:
        assert "marker" in t and "dash" in t


def test_reliability_embeds_occupancy_and_footer(tmp_path: Path, tokens_path: Path):
    rng = np.random.default_rng(0)
    probs = rng.uniform(0.1, 0.9, 200)
    labels = (rng.uniform(0, 1, 200) < probs).astype(int)
    ece = expected_calibration_error(probs, labels, n_bins=8)
    ctx = make_context(
        run_id="run-test-001",
        resolved_model="jev-1.13.0",
        out_dir=tmp_path,
        tokens_path=tokens_path,
    )
    paths = reliability_diagram(ctx, ece=ece, tier_label="hard")
    assert paths["svg"].is_file()
    assert paths["1080p"].is_file()
    assert paths["1440p"].is_file()
    Image = pytest.importorskip("PIL.Image")
    im1080 = Image.open(paths["1080p"])
    im1440 = Image.open(paths["1440p"])
    assert im1080.size == (1920, 1080)
    assert im1440.size == (2560, 1440)
    svg = paths["svg"].read_text(encoding="utf-8")
    assert "run_id=run-test-001" in svg
    assert "model=jev-1.13.0" in svg
    assert "occupancy" in svg.lower() or "Count" in svg


def test_money_chart_and_transparent(tmp_path: Path, tokens_path: Path):
    ctx = make_context(
        run_id="r2",
        resolved_model="jev-1.13.0",
        out_dir=tmp_path / "opaque",
        tokens_path=tokens_path,
        transparent=False,
    )
    paths = ece_vs_accuracy_by_tier(
        ctx,
        tier_names=["trivial", "easy", "hard", "ambiguous"],
        accuracies=[0.95, 0.88, 0.7, 0.55],
        eces=[0.04, 0.07, 0.12, 0.18],
    )
    assert paths["1080p"].stat().st_size > 1000

    ctx_t = make_context(
        run_id="r2",
        resolved_model="jev-1.13.0",
        out_dir=tmp_path / "transparent",
        tokens_path=tokens_path,
        transparent=True,
    )
    paths_t = ece_vs_accuracy_by_tier(
        ctx_t,
        tier_names=["trivial", "easy", "hard", "ambiguous"],
        accuracies=[0.95, 0.88, 0.7, 0.55],
        eces=[0.04, 0.07, 0.12, 0.18],
        stem="ece_vs_accuracy_by_tier_t",
    )
    # Transparent PNG should have an alpha channel
    Image = pytest.importorskip("PIL.Image", reason="Pillow needed for alpha check")
    im = Image.open(paths_t["1080p"])
    assert im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)


def test_remaining_chart_types_smoke(tmp_path: Path, tokens_path: Path):
    ctx = make_context(
        run_id="smoke",
        resolved_model="jev-1.13.0",
        out_dir=tmp_path,
        tokens_path=tokens_path,
    )
    three_arm_comparison(
        ctx,
        arms=["jev", "adapter", "trivial"],
        accuracy=[0.8, 0.7, 0.6],
        cost=[0.1, 2.0, 0.0],
        latency_p50=[80, 300, 1],
        accuracy_err=[0.02, 0.03, 0.04],
    )
    latency_distribution(
        ctx,
        series_latencies={"jev": np.random.default_rng(0).normal(90, 15, 100)},
        kind="ecdf",
    )
    cost_per_1000_correct(ctx, arms=["jev", "adapter"], cost_per_1000=[0.2, 5.0])
    selective_automation_curve(
        ctx,
        coverage=np.linspace(0.3, 1, 15),
        accuracy=np.linspace(0.97, 0.8, 15),
        target_accuracy=0.9,
    )
    invariant_scatter(
        ctx,
        noul_probs=np.linspace(0.1, 0.9, 30),
        choice_probs=np.linspace(0.15, 0.85, 30),
    )
    # Ensure files landed
    assert list(tmp_path.glob("*.png"))
    assert list(tmp_path.glob("*.svg"))
