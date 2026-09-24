"""ChaosNLI lock, soft/hard scoring, and asymmetric noise."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jevbench.chaosnli import (
    ALPHANLI_EXCLUSION,
    POPULATION_N,
    TARGET_N,
    load_population,
    shannon_entropy,
)
from jevbench.dataset import load_dataset
from jevbench.metrics import (
    expected_calibration_error,
    hard_correctness,
    soft_correctness,
    subsample_to_equal_n,
)
from jevbench.power import majority_share_from_entropy, noise_rate_from_entropy

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "datasets" / "chaosnli"


def test_hard_scoring_docstring_states_the_bias():
    assert hard_correctness.__doc__ is not None
    assert (
        "Hard scoring puts label noise in the hard stratum only, which inflates "
        "ΔECE in the direction of the hypothesis."
    ) in hard_correctness.__doc__


def test_soft_and_hard_correctness():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    dist = np.array([[0.5, 0.4, 0.1], [0.2, 0.2, 0.6]])
    soft = soft_correctness(probs, dist)
    assert soft[0] == pytest.approx(0.5)
    assert soft[1] == pytest.approx(0.6)
    hard = hard_correctness(probs, np.array([0, 1]))
    assert hard.tolist() == [1.0, 0.0]


def test_soft_ece_bin_accuracy_is_mean_soft_correctness():
    probs = np.tile(np.array([[0.9, 0.05, 0.05]]), (4, 1))
    labels = np.zeros(4, dtype=int)
    soft = np.array([1.0, 0.5, 0.5, 0.0])
    result = expected_calibration_error(probs, labels, n_bins=10, correctness=soft)
    occupied = [b for b in result.bins if b.count]
    assert len(occupied) == 1
    assert occupied[0].accuracy == pytest.approx(0.5)
    assert result.ece == pytest.approx(abs(0.9 - 0.5))


def test_subsample_target_n_draws_from_both():
    sub = subsample_to_equal_n(
        np.zeros(10),
        np.zeros(10, dtype=int),
        [f"a{i}" for i in range(10)],
        np.zeros(8),
        np.zeros(8, dtype=int),
        [f"b{i}" for i in range(8)],
        seed=1,
        target_n=5,
    )
    assert sub.target_n == 5
    assert len(sub.item_ids_a) == len(sub.item_ids_b) == 5
    assert len(sub.discarded_item_ids) == 8


def test_noise_rate_rises_with_entropy():
    easy = noise_rate_from_entropy([0.0, 0.2])
    hard = noise_rate_from_entropy([1.5, np.log2(3)])
    assert easy[0] == 0.0
    assert hard[-1] == pytest.approx(0.5)
    assert float(hard.mean()) > float(easy.mean())
    shares = majority_share_from_entropy([0.0, np.log2(3)])
    assert shares[0] == pytest.approx(1.0)
    assert shares[1] == pytest.approx(0.5)


def test_locked_population_has_no_sentences_and_equal_n():
    thresholds = json.loads((TASK / "thresholds.json").read_text(encoding="utf-8"))
    assert thresholds["n_population"] == POPULATION_N
    assert thresholds["target_n_per_stratum"] == TARGET_N
    assert "abductive" in ALPHANLI_EXCLUSION
    assert thresholds["alphanli_excluded"] is True
    universe = (TASK / "universe.jsonl").read_text(encoding="utf-8")
    assert '"premise"' not in universe and '"hypothesis"' not in universe
    ds = load_dataset(ROOT / "datasets", "chaosnli")
    primary = [li for li in ds.by_id.values() if li.item.role == "primary"]
    para = [li for li in ds.by_id.values() if li.item.role == "paraphrase"]
    assert len(primary) == 1500
    assert sum(li.tier == "easy" for li in primary) == 750
    assert sum(li.tier == "hard" for li in primary) == 750
    assert len(para) == 100
    for li in primary:
        state = li.item.state
        assert isinstance(state, dict)
        assert "premise" not in state and "hypothesis" not in state
        assert li.label.label_dist is not None
        assert abs(sum(li.label.label_dist) - 1.0) < 1e-9
    for li in para:
        state = li.item.state
        assert isinstance(state, dict)
        assert "premise" not in state and "hypothesis" not in state
        assert state.get("text_status") == "local_paraphrase_required"


def test_entropy_matches_readme_example():
    h = shannon_entropy([76, 20, 4])
    assert h == pytest.approx(0.9510456605801273, abs=1e-12)


def test_load_population_rejects_short_files(tmp_path: Path):
    snli = tmp_path / "s.jsonl"
    mnli = tmp_path / "m.jsonl"
    snli.write_text("{}\n", encoding="utf-8")
    mnli.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_population(snli, mnli)
