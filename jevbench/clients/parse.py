"""Parse System One response objects/dicts into Decision rows.

Shared by the native Jev client and the LLM adapter — same wire shape.
"""

from __future__ import annotations

from typing import Any

from jevbench.clients.base import (
    ChoiceQuestion,
    Decision,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    error_decision,
    question_kind,
)


def _as_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion of SDK objects to plain dicts for ``raw``."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        out: dict[str, Any] = {}
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            out[k] = _as_dict(v) if hasattr(v, "__dict__") and not isinstance(
                v, (str, bytes, int, float, bool, type(None))
            ) else v
        return out
    return {"repr": repr(obj)}


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _probabilities(answer: Any) -> dict[str, float] | None:
    probs = _get(answer, "probabilities")
    if probs is None:
        return None
    if isinstance(probs, dict):
        return {str(k): float(v) for k, v in probs.items()}
    return None


def parse_answer(
    *,
    item_id: str,
    question_key: str,
    question: Question,
    answer: Any,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    resolved_model: str,
    serving_path: str,
    attempt: int,
    full_raw: dict[str, Any],
) -> Decision:
    kind = question_kind(question)
    raw_answer = _as_dict(answer)
    # Attach full response envelope for provenance; keep answer under key.
    raw = {
        "answer": raw_answer,
        "response": full_raw,
    }

    if answer is None:
        return error_decision(
            item_id=item_id,
            question_key=question_key,
            kind=kind,
            latency_ms=latency_ms,
            resolved_model=resolved_model,
            serving_path=serving_path,
            attempt=attempt,
            error=f"missing answer for question {question_key!r}",
            raw=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    if kind == "noul":
        noul = _get(answer, "noul")
        if noul is None:
            return error_decision(
                item_id=item_id,
                question_key=question_key,
                kind=kind,
                latency_ms=latency_ms,
                resolved_model=resolved_model,
                serving_path=serving_path,
                attempt=attempt,
                error="noul answer missing 'noul' field",
                raw=raw,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        # Noul has NO confidence field. Never substitute.
        return Decision(
            item_id=item_id,
            question_key=question_key,
            kind="noul",
            value=float(noul),
            probabilities=None,
            confidence=None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            resolved_model=resolved_model,
            serving_path=serving_path,
            attempt=attempt,
            error=None,
            raw=raw,
        )

    if kind == "choice":
        choice = _get(answer, "choice")
        conf = _get(answer, "confidence")
        return Decision(
            item_id=item_id,
            question_key=question_key,
            kind="choice",
            value=str(choice) if choice is not None else "",
            probabilities=_probabilities(answer),
            confidence=float(conf) if conf is not None else None,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            resolved_model=resolved_model,
            serving_path=serving_path,
            attempt=attempt,
            error=None if choice is not None else "choice answer missing 'choice' field",
            raw=raw,
        )

    # score — FLOAT expectation. NEVER round to int. NEVER interpolate levels.
    score = _get(answer, "score")
    conf = _get(answer, "confidence")
    legend = _get(answer, "legend")
    if legend is not None and "legend" not in raw_answer:
        raw_answer["legend"] = legend
        raw["answer"] = raw_answer
    if score is None:
        return error_decision(
            item_id=item_id,
            question_key=question_key,
            kind="score",
            latency_ms=latency_ms,
            resolved_model=resolved_model,
            serving_path=serving_path,
            attempt=attempt,
            error="score answer missing 'score' field",
            raw=raw,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    score_f = float(score)
    # Guard: refuse integer-looking coercion markers in raw for auditors
    raw_answer["score_as_float"] = score_f
    raw["answer"] = raw_answer
    return Decision(
        item_id=item_id,
        question_key=question_key,
        kind="score",
        value=score_f,
        probabilities=_probabilities(answer),
        confidence=float(conf) if conf is not None else None,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        resolved_model=resolved_model,
        serving_path=serving_path,
        attempt=attempt,
        error=None,
        raw=raw,
    )


def parse_response(
    request: SystemOneRequest,
    response: Any,
    *,
    latency_ms: float,
    serving_path: str,
    attempt: int,
    fallback_model: str,
) -> list[Decision]:
    full_raw = _as_dict(response)
    resolved = str(_get(response, "model", default=fallback_model) or fallback_model)
    usage = _get(response, "usage", default={}) or {}
    input_tokens = int(_get(usage, "input_tokens", default=0) or 0)
    output_tokens = int(_get(usage, "output_tokens", default=0) or 0)

    answers = _get(response, "answers", default={}) or {}
    if not isinstance(answers, dict):
        # SDK may expose typed accessors; try building from request keys
        answers = {}
        for key in request.questions:
            answers[key] = _get(response, key)

    # Prefer response.answers[key]; also try typed maps if present
    typed_maps = {
        "noul": _get(response, "nouls", default=None),
        "choice": _get(response, "choices", default=None),
        "score": _get(response, "scores", default=None),
    }

    decisions: list[Decision] = []
    for key, question in request.questions.items():
        kind = question_kind(question)
        answer = None
        if isinstance(answers, dict) and key in answers:
            answer = answers[key]
        else:
            typed = typed_maps.get(kind)
            if typed is not None:
                answer = _get(typed, key) if not isinstance(typed, dict) else typed.get(key)

        decisions.append(
            parse_answer(
                item_id=request.item_id,
                question_key=key,
                question=question,
                answer=answer,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                resolved_model=resolved,
                serving_path=serving_path,
                attempt=attempt,
                full_raw=full_raw,
            )
        )
    return decisions


def build_sdk_questions(questions: dict[str, Question]) -> dict[str, Any]:
    """Convert our Question objects to typesafe_sdk / adapter constructors.

    Imports are deferred so unit tests never need the SDKs on the path.
    """
    from typesafe_sdk import Choice, Noul, Score  # type: ignore

    out: dict[str, Any] = {}
    for key, q in questions.items():
        if isinstance(q, ChoiceQuestion):
            out[key] = Choice(instructions=q.instructions, criteria=q.criteria)
        elif isinstance(q, NoulQuestion):
            out[key] = Noul(instructions=q.instructions)
        elif isinstance(q, ScoreQuestion):
            out[key] = Score(instructions=q.instructions, criteria=list(q.criteria))
        else:
            raise TypeError(f"unknown question: {type(q)!r}")
    return out


def build_adapter_questions(questions: dict[str, Question]) -> dict[str, Any]:
    from system_one_adapter import Choice, Noul, Score  # type: ignore

    out: dict[str, Any] = {}
    for key, q in questions.items():
        if isinstance(q, ChoiceQuestion):
            out[key] = Choice(instructions=q.instructions, criteria=q.criteria)
        elif isinstance(q, NoulQuestion):
            out[key] = Noul(instructions=q.instructions)
        elif isinstance(q, ScoreQuestion):
            out[key] = Score(instructions=q.instructions, criteria=list(q.criteria))
        else:
            raise TypeError(f"unknown question: {type(q)!r}")
    return out
