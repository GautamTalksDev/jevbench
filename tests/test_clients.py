"""Offline unit tests for the client layer. No network."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from jevbench.clients import (
    AdapterClient,
    AdapterClientConfig,
    ChoiceQuestion,
    JevClient,
    JevClientConfig,
    KeywordRule,
    NoulQuestion,
    PrefillClient,
    PrefillClientConfig,
    PrefillCompletion,
    ScoreQuestion,
    SystemOneRequest,
    TrivialClient,
    TrivialClientConfig,
    TrivialQuestionSpec,
    build_sentinel_mapping,
)
from jevbench.clients.base import SENTINEL_DOC_KEY, RetryPolicy
from jevbench.clients.parse import parse_response
from jevbench.clients.prefill import softmax_logprobs

FIXTURES = Path(__file__).parent / "fixtures"


def _ticket_request(item_id: str = "item-1") -> SystemOneRequest:
    return SystemOneRequest(
        item_id=item_id,
        state="I was charged twice for order A-104. Please refund the duplicate.",
        questions={
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={
                    "billing": "Charges, invoices, refunds",
                    "technical": "Bugs and integration problems",
                    "other": None,
                },
            ),
            "refund_requested": NoulQuestion(
                instructions="Does the customer ask for money back?",
            ),
            "urgency": ScoreQuestion(
                instructions="How urgent is this ticket?",
                criteria=["can wait", "this week", "today"],
            ),
        },
        model="jev-1.13.0",
    )


def test_parse_preserves_score_float_and_noul_confidence_none():
    raw = json.loads((FIXTURES / "systemone_success.json").read_text())
    req = _ticket_request()
    decisions = parse_response(
        req,
        raw,
        latency_ms=12.5,
        serving_path="native",
        attempt=1,
        fallback_model="jev-1.13.0",
    )
    by_key = {d.question_key: d for d in decisions}

    noul = by_key["refund_requested"]
    assert noul.kind == "noul"
    assert noul.value == pytest.approx(0.999)
    assert noul.confidence is None
    assert noul.probabilities is None

    score = by_key["urgency"]
    assert score.kind == "score"
    assert score.value == pytest.approx(1.035)
    assert isinstance(score.value, float)
    assert score.value != int(score.value)  # not rounded
    assert score.raw["answer"]["legend"]["1"] == "this week"

    choice = by_key["department"]
    assert choice.value == "billing"
    assert choice.probabilities["billing"] == pytest.approx(0.84)
    assert choice.confidence == pytest.approx(0.596)
    assert choice.resolved_model == "jev-1.13.0"
    assert choice.latency_ms == 12.5
    assert choice.input_tokens == 312
    assert choice.output_tokens == 48


def test_jev_client_native_path_uses_transport_fixture():
    raw = json.loads((FIXTURES / "systemone_success.json").read_text())
    calls: list[SystemOneRequest] = []

    def transport(request: SystemOneRequest) -> dict[str, Any]:
        calls.append(request)
        return raw

    client = JevClient(
        JevClientConfig(serving_path="native", model="jev-1.13.0"),
        transport=transport,
        sleep=lambda _s: None,
    )
    decisions = client.decide(_ticket_request())
    assert client.serving_path == "native"
    assert all(d.serving_path == "native" for d in decisions)
    assert all(d.attempt == 1 for d in decisions)
    assert all(d.error is None for d in decisions)
    assert len(calls) == 1


def test_jev_client_gateway_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        JevClient(JevClientConfig(serving_path="gateway"))


def test_jev_client_gateway_records_path():
    raw = json.loads((FIXTURES / "systemone_success.json").read_text())
    client = JevClient(
        JevClientConfig(
            serving_path="gateway",
            base_url="https://gateway.example/v1",
        ),
        transport=lambda _r: raw,
        sleep=lambda _s: None,
    )
    decisions = client.decide(_ticket_request())
    assert all(d.serving_path == "gateway" for d in decisions)


class RateLimitError(Exception):
    def __init__(self, retry_after: float | None = None):
        super().__init__("rate limited")
        self.status_code = 429
        self.headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
        self.retry_after = retry_after


def test_jev_client_does_not_retry_429():
    """Amendment 10: never retry 4xx including 429."""
    raw = json.loads((FIXTURES / "systemone_success.json").read_text())
    sleeps: list[float] = []
    n = {"i": 0}

    def transport(_request: SystemOneRequest) -> dict[str, Any]:
        n["i"] += 1
        if n["i"] < 3:
            raise RateLimitError(retry_after=1.25)
        return raw

    client = JevClient(
        JevClientConfig(retry=RetryPolicy(max_attempts=5)),
        transport=transport,
        sleep=sleeps.append,
    )
    decisions = client.decide(_ticket_request())
    assert n["i"] == 1  # no retries
    assert sleeps == []
    assert all(d.attempt == 1 for d in decisions)
    assert all(d.error and "RateLimitError" in d.error for d in decisions)


def test_jev_client_records_error_after_non_retryable_429():
    def transport(_request: SystemOneRequest) -> dict[str, Any]:
        raise RateLimitError(retry_after=0.01)

    client = JevClient(
        JevClientConfig(retry=RetryPolicy(max_attempts=3, base_delay_s=0.01)),
        transport=transport,
        sleep=lambda _s: None,
    )
    decisions = client.decide(_ticket_request())
    assert len(decisions) == 3
    assert all(d.error and "RateLimitError" in d.error for d in decisions)
    assert all(d.attempt == 1 for d in decisions)


def test_adapter_client_exposes_llm_answer_mode_ablation():
    raw = json.loads((FIXTURES / "adapter_success.json").read_text())

    def transport(_request: SystemOneRequest) -> dict[str, Any]:
        return raw

    for mode in ("probabilities", "discrete"):
        client = AdapterClient(
            AdapterClientConfig(
                provider="anthropic",
                model="claude-haiku-like",
                llm_answer_mode=mode,  # type: ignore[arg-type]
            ),
            transport=transport,
            sleep=lambda _s: None,
        )
        req = SystemOneRequest(
            item_id="a1",
            state="This book was a delight to read.",
            questions={
                "positive": NoulQuestion(instructions="The book review is positive."),
                "topic": ChoiceQuestion(
                    instructions="Topic?",
                    criteria={"billing": None, "technical": None},
                ),
            },
            model="claude-haiku-like",
        )
        decisions = client.decide(req)
        assert all(d.serving_path == "adapter" for d in decisions)
        assert all(d.raw["adapter_config"]["llm_answer_mode"] == mode for d in decisions)
        noul = next(d for d in decisions if d.kind == "noul")
        assert noul.confidence is None
        assert noul.value == pytest.approx(0.91)


def test_adapter_rejects_bad_mode():
    with pytest.raises(ValueError, match="llm_answer_mode"):
        AdapterClient(AdapterClientConfig(llm_answer_mode="softmax"))  # type: ignore[arg-type]


def test_prefill_sentinel_mapping_and_softmax():
    options = ["billing", "technical", "other"]
    mapping = build_sentinel_mapping(options)
    assert set(mapping.keys()) == set(options)
    assert len(set(mapping.values())) == 3
    assert all(len(s) == 1 for s in mapping.values())

    logprobs = {"1": -0.1, "2": -1.0, "3": -2.0}
    probs = softmax_logprobs(logprobs, ["1", "2", "3"])
    assert probs["1"] > probs["2"] > probs["3"]
    assert math.isclose(sum(probs.values()), 1.0, rel_tol=1e-9)


def test_prefill_client_with_injected_backend():
    options = ["billing", "technical", "other"]

    class FakeBackend:
        def get_tokenizer(self):
            from jevbench.clients.prefill import CharTokenizer

            return CharTokenizer()

        def complete_one_token(self, *, messages, allowed_tokens, model):
            assert messages[-1]["role"] == "assistant"
            assert messages[-1]["content"].startswith('{"choice": "') or messages[
                -1
            ]["content"].startswith('{"answer": "')
            # Put mass on the first allowed sentinel (stable for both choice & noul)
            chosen = allowed_tokens[0]
            lps = {t: -100.0 for t in allowed_tokens}
            lps[chosen] = 0.0
            return PrefillCompletion(
                token=chosen,
                logprobs=lps,
                resolved_model="fake-qwen",
                raw={"fake": True, "allowed": list(allowed_tokens)},
                input_tokens=50,
                output_tokens=1,
                compute_only_ms=1.5,
            )

    client = PrefillClient(
        PrefillClientConfig(model="fake-qwen", require_tokenizer_assert=True),
        backend=FakeBackend(),
    )
    client.ensure_tokenizer_ready(options)
    req = SystemOneRequest(
        item_id="p1",
        state="refund please",
        questions={
            "department": ChoiceQuestion(
                instructions="Team?",
                criteria={k: None for k in options},
            ),
            "refund": NoulQuestion(instructions="Refund asked?"),
        },
        pass_idx=0,
    )
    decisions = client.decide(req)
    by_key = {d.question_key: d for d in decisions}
    # First sentinel maps to first option ("billing") at pass_idx=0
    assert by_key["department"].value == "billing"
    assert by_key["department"].probabilities is not None
    assert by_key["department"].probabilities["billing"] == pytest.approx(1.0, abs=1e-6)
    assert SENTINEL_DOC_KEY in by_key["department"].raw
    assert by_key["department"].raw["latency"]["fair_comparison"] is False
    assert by_key["refund"].kind == "noul"
    assert by_key["refund"].confidence is None
    # First noul sentinel is "yes" → p_true ≈ 1
    assert by_key["refund"].value == pytest.approx(1.0, abs=1e-6)
    manifest = client.sentinel_mappings_for_manifest()
    assert "department" in manifest[SENTINEL_DOC_KEY]
    assert "refund" in manifest[SENTINEL_DOC_KEY]


def test_prefill_order_permutation_changes_mapping():
    from jevbench.clients.prefill import build_sentinel_mapping

    options = ["billing", "technical", "other"]
    m0 = build_sentinel_mapping(options, pass_idx=0, permutation_seed=1)
    m1 = build_sentinel_mapping(options, pass_idx=1, permutation_seed=1)
    # Same sentinels, possibly different assignment
    assert set(m0.values()) == set(m1.values())
    # With seed+pass shuffle, pass 1 should differ from pass 0 for N=3
    assert m0 != m1


def test_sentinel_assert_fails_on_multitoken():
    from jevbench.clients.prefill import (
        SentinelTokenError,
        assert_sentinels_are_single_tokens,
    )

    class BadTok:
        def encode(self, text, add_special_tokens=False):
            # Everything becomes two tokens
            return [1, 2]

    with pytest.raises(SentinelTokenError, match="single-token"):
        assert_sentinels_are_single_tokens(BadTok(), {"billing": "1"})


def test_gliclass_injected_scorer():
    from jevbench.clients.gliclass import GLiClassClient, GLiClassClientConfig

    class FakeScorer:
        def score(self, text, labels):
            return {lab: (1.0 if lab == "billing" else 0.1) for lab in labels}

    client = GLiClassClient(GLiClassClientConfig(), scorer=FakeScorer())
    req = SystemOneRequest(
        item_id="g1",
        state="refund please",
        questions={
            "department": ChoiceQuestion(
                instructions="Team?",
                criteria={"billing": None, "technical": None, "other": None},
            ),
        },
    )
    decs = client.decide(req)
    assert len(decs) == 1
    assert decs[0].value == "billing"
    assert decs[0].probabilities is not None
    assert math.isclose(sum(decs[0].probabilities.values()), 1.0, rel_tol=1e-9)


def test_trivial_regex_clears_refund():
    client = TrivialClient(
        TrivialClientConfig(
            specs={
                "refund_requested": TrivialQuestionSpec(
                    regex=r"refund",
                ),
                "department": TrivialQuestionSpec(
                    keywords=(
                        KeywordRule("billing", (r"charg", r"refund", r"invoice")),
                        KeywordRule("technical", (r"bug", r"crash")),
                    ),
                    majority_class="other",
                ),
                "urgency": TrivialQuestionSpec(
                    majority_class="this week",
                ),
            }
        )
    )
    decisions = client.decide(_ticket_request("t1"))
    by_key = {d.question_key: d for d in decisions}
    assert by_key["refund_requested"].value == pytest.approx(1.0)
    assert by_key["refund_requested"].confidence is None
    assert by_key["department"].value == "billing"
    assert by_key["urgency"].value == pytest.approx(1.0)
    assert all(d.input_tokens == 0 and d.output_tokens == 0 for d in decisions)
    assert all(d.serving_path == "trivial" for d in decisions)
    assert all(d.latency_ms >= 0 for d in decisions)


def test_trivial_majority_when_regex_misses():
    client = TrivialClient(
        TrivialClientConfig(
            specs={
                "department": TrivialQuestionSpec(
                    regex=r"xyzzy-never",
                    majority_class="other",
                )
            }
        )
    )
    req = SystemOneRequest(
        item_id="m1",
        state="hello world",
        questions={
            "department": ChoiceQuestion(
                instructions="Team?",
                criteria={"billing": None, "technical": None, "other": None},
            )
        },
    )
    d = client.decide(req)[0]
    assert d.value == "other"
    assert d.raw["trivial"]["strategy"] == "majority_class"


def test_all_clients_share_decision_fields():
    """Structural contract: every Decision has the exact field set."""
    expected = {
        "item_id",
        "question_key",
        "kind",
        "value",
        "probabilities",
        "confidence",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "resolved_model",
        "serving_path",
        "attempt",
        "error",
        "raw",
    }
    raw = json.loads((FIXTURES / "systemone_success.json").read_text())
    jev = JevClient(transport=lambda _r: raw, sleep=lambda _s: None)
    for d in jev.decide(_ticket_request()):
        assert set(d.to_dict().keys()) == expected
