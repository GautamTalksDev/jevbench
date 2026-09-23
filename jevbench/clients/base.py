"""Shared Decision shape and client protocol.

Every client in ``jevbench.clients`` returns ``Decision`` rows — nothing else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol, runtime_checkable

Kind = Literal["noul", "choice", "score"]
ServingPath = Literal["native", "gateway", "adapter", "prefill", "trivial"]
LlmAnswerMode = Literal["probabilities", "discrete"]
LlmProvider = Literal["openai", "anthropic"]


@dataclass(frozen=True)
class Decision:
    """One answer for one question on one item.

    Exact shared interface — do not add fields without updating every client
    and the runner schema.
    """

    item_id: str
    question_key: str
    kind: Kind
    value: str | float  # choice name | score expectation | noul prob
    probabilities: dict[str, float] | None
    confidence: float | None  # None for noul — expected, not a bug
    latency_ms: float  # wall clock around the call ONLY
    input_tokens: int
    output_tokens: int
    resolved_model: str
    serving_path: str
    attempt: int
    error: str | None
    raw: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChoiceQuestion:
    instructions: Any  # str or structured JSON per API
    criteria: dict[str, str | None]


@dataclass(frozen=True)
class NoulQuestion:
    instructions: Any


@dataclass(frozen=True)
class ScoreQuestion:
    instructions: Any
    criteria: list[str]  # ordered rubric; never a dict


Question = ChoiceQuestion | NoulQuestion | ScoreQuestion


@dataclass
class SystemOneRequest:
    """Common request envelope passed to every client."""

    item_id: str
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question]
    model: str = "jev-1.13.0"
    # Repeat index — prefill uses this to shuffle sentinel↔option assignment
    pass_idx: int = 0


@runtime_checkable
class DecisionClient(Protocol):
    """Shared client interface: one request → one Decision per question."""

    serving_path: str

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        """Evaluate all questions. Never raises for API/business failures —

        failures become Decision rows with ``error`` set and ``value`` left
        in a safe default (empty string / 0.0) only when no answer exists.
        """
        ...


def question_kind(q: Question) -> Kind:
    if isinstance(q, NoulQuestion):
        return "noul"
    if isinstance(q, ChoiceQuestion):
        return "choice"
    if isinstance(q, ScoreQuestion):
        return "score"
    raise TypeError(f"unknown question type: {type(q)!r}")


def error_decision(
    *,
    item_id: str,
    question_key: str,
    kind: Kind,
    latency_ms: float,
    resolved_model: str,
    serving_path: str,
    attempt: int,
    error: str,
    raw: dict | None = None,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Decision:
    """Build a Decision row for a failed call (never drop the failure)."""
    default: str | float = 0.0 if kind in ("noul", "score") else ""
    return Decision(
        item_id=item_id,
        question_key=question_key,
        kind=kind,
        value=default,
        probabilities=None,
        confidence=None,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        resolved_model=resolved_model,
        serving_path=serving_path,
        attempt=attempt,
        error=error,
        raw=raw if raw is not None else {},
    )


@dataclass
class RetryPolicy:
    max_attempts: int = 5
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0


# Re-export helper used by manifests
SENTINEL_DOC_KEY = "prefill_sentinel_mapping"
