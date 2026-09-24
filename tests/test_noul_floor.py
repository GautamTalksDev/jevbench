"""PROMPT R — noul near-zero floor + mid-item guard helpers."""

from __future__ import annotations

from jevbench.exp1 import NOUL_SUM_FLOOR, normalize_noul_triplet


def test_noul_normalize_ordinary() -> None:
    decisions = [
        {"question_key": "noul_entailment", "value": 0.2},
        {"question_key": "noul_neutral", "value": 0.3},
        {"question_key": "noul_contradiction", "value": 0.5},
    ]
    pack = normalize_noul_triplet(decisions)
    assert pack is not None
    assert pack["near_zero_sum"] is False
    assert abs(sum(pack["probabilities"].values()) - 1.0) < 1e-9
    assert pack["probabilities"]["contradiction"] == 0.5


def test_noul_near_zero_floor() -> None:
    decisions = [
        {"question_key": "noul_entailment", "value": 0.0},
        {"question_key": "noul_neutral", "value": 0.0},
        {"question_key": "noul_contradiction", "value": NOUL_SUM_FLOOR / 10},
    ]
    pack = normalize_noul_triplet(decisions)
    assert pack is not None
    assert pack["near_zero_sum"] is True
    assert pack["probabilities"] == {
        "entailment": 1 / 3,
        "neutral": 1 / 3,
        "contradiction": 1 / 3,
    }


def test_noul_incomplete_returns_none() -> None:
    assert (
        normalize_noul_triplet(
            [{"question_key": "noul_entailment", "value": 0.5}]
        )
        is None
    )
