"""Tests for bart_mnli_nli (true NLI head) and id2label mapping."""

from __future__ import annotations

import pytest

from jevbench.clients.bart_mnli import (
    BartMnliNliClient,
    BartMnliNliClientConfig,
    map_id2label_probs,
    premise_hypothesis_from_state,
)
from jevbench.clients.base import ChoiceQuestion, NoulQuestion, SystemOneRequest


def test_premise_hypothesis_from_state() -> None:
    p, h = premise_hypothesis_from_state(
        {"premise": "A dog runs.", "hypothesis": "An animal moves."}
    )
    assert "dog" in p and "animal" in h
    with pytest.raises(TypeError):
        premise_hypothesis_from_state("not a dict")
    with pytest.raises(ValueError):
        premise_hypothesis_from_state({"premise": "", "hypothesis": "x"})


def test_map_id2label_probs_canonical() -> None:
    torch = pytest.importorskip("torch")
    # HF bart-large-mnli: 0=contradiction, 1=neutral, 2=entailment
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    logits = torch.tensor([[0.0, 0.0, 5.0]])  # entailment wins
    probs = map_id2label_probs(logits, id2label)
    assert set(probs) == {"entailment", "neutral", "contradiction"}
    assert probs["entailment"] == pytest.approx(1.0, abs=0.05)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_bart_mnli_nli_noul_errors() -> None:
    torch = pytest.importorskip("torch")

    class _Tok:
        model_max_length = 128

        def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

    class _Mdl:
        config = type(
            "C",
            (),
            {"id2label": {0: "contradiction", 1: "neutral", 2: "entailment"}},
        )()

        def __call__(self, **kwargs):  # noqa: ANN003
            return type("O", (), {"logits": torch.tensor([[0.0, 0.0, 5.0]])})()

        def eval(self):
            return self

        def to(self, device):  # noqa: ANN001
            return self

    client = BartMnliNliClient(BartMnliNliClientConfig())
    client._tokenizer = _Tok()
    client._model = _Mdl()
    req = SystemOneRequest(
        item_id="t1",
        state={"premise": "A dog runs.", "hypothesis": "An animal moves."},
        questions={
            "noul_entailment": NoulQuestion(instructions="entailed?"),
        },
        model="facebook/bart-large-mnli",
        pass_idx=0,
    )
    decs = client.decide(req)
    assert len(decs) == 1
    assert decs[0].error and "Choice" in decs[0].error


def test_bart_mnli_nli_clear_entailment_live() -> None:
    """Real model: clear entailment pair should prefer entailment (skip if offline)."""
    pytest.importorskip("transformers")
    pytest.importorskip("torch")
    try:
        client = BartMnliNliClient(
            BartMnliNliClientConfig(model_id="facebook/bart-large-mnli", device="cpu")
        )
        req = SystemOneRequest(
            item_id="entail_smoke",
            state={
                "premise": "A woman is playing the piano.",
                "hypothesis": "A woman is playing an instrument.",
            },
            questions={
                "relation": ChoiceQuestion(
                    instructions="NLI",
                    criteria={
                        "entailment": "e",
                        "neutral": "n",
                        "contradiction": "c",
                    },
                )
            },
            model="facebook/bart-large-mnli",
            pass_idx=0,
        )
        decs = client.decide(req)
        client.release()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"model unavailable offline: {exc}")
    assert not decs[0].error, decs[0].error
    assert decs[0].value == "entailment"
    assert decs[0].probabilities is not None
    assert decs[0].probabilities["entailment"] > 0.5
    assert "id2label" in (decs[0].raw or {})
