"""Non-AI floor: regex, keyword rules, and majority-class.

Same ``Decision`` shape and wall-clock ``latency_ms`` measurement. Zero cost.
If a regex clears a task, the task measured the dataset — not the model.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from jevbench.clients.base import (
    ChoiceQuestion,
    Decision,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    error_decision,
    question_kind,
)

RuleFn = Callable[[str], str | float | None]


@dataclass
class KeywordRule:
    """If any pattern matches (case-insensitive), return ``label``."""

    label: str | float
    patterns: Sequence[str]
    flags: int = re.IGNORECASE

    def match(self, text: str) -> bool:
        return any(re.search(p, text, self.flags) for p in self.patterns)


@dataclass
class TrivialQuestionSpec:
    """Per-question non-AI strategy.

    Exactly one of ``regex``, ``keywords``, ``majority_class``, or ``rule``
    should drive the prediction. Priority: ``rule`` > ``regex`` > ``keywords``
    > ``majority_class``.
    """

    # Full-match or search regex → group(1) or whole match as choice/label;
    # for noul, pattern presence → 1.0 else 0.0 unless ``noul_default`` set.
    regex: str | None = None
    regex_flags: int = re.IGNORECASE
    keywords: Sequence[KeywordRule] = field(default_factory=tuple)
    majority_class: str | float | None = None
    noul_default: float = 0.0
    # Arbitrary callable for tests / complex floors
    rule: RuleFn | None = None
    # Optional fixed probabilities when we want a sharp one-hot
    emit_one_hot: bool = True


@dataclass
class TrivialClientConfig:
    serving_path: str = "trivial"
    resolved_model: str = "trivial/regex-keyword-majority"
    # question_key → strategy
    specs: Mapping[str, TrivialQuestionSpec] = field(default_factory=dict)
    # Fallback when a question has no spec: majority / default noul 0.0
    default_majority: str | float | None = None


def _state_text(state: str | dict[str, Any] | list[Any]) -> str:
    if isinstance(state, str):
        return state
    import json

    return json.dumps(state, ensure_ascii=False)


class TrivialClient:
    """Regex / keyword / majority-class floor behind the shared interface."""

    def __init__(self, config: TrivialClientConfig | None = None) -> None:
        self.config = config or TrivialClientConfig()

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def _predict(
        self,
        text: str,
        question: Question,
        spec: TrivialQuestionSpec | None,
    ) -> tuple[str | float, dict[str, float] | None, dict[str, Any]]:
        meta: dict[str, Any] = {"strategy": None}
        kind = question_kind(question)

        if spec and spec.rule is not None:
            meta["strategy"] = "rule"
            val = spec.rule(text)
            if val is None:
                raise ValueError("rule returned None")
            return val, self._one_hot(question, val, spec), meta

        if spec and spec.regex:
            meta["strategy"] = "regex"
            m = re.search(spec.regex, text, spec.regex_flags)
            if kind == "noul":
                val = 1.0 if m else float(spec.noul_default)
                return val, None, meta
            if m:
                captured = m.group(1) if m.lastindex else m.group(0)
                if kind == "score":
                    # If criteria are labels, map match to index expectation;
                    # if the capture is numeric, use float directly (still not
                    # interpolating between levels — raw parse only).
                    try:
                        return float(captured), self._one_hot(question, float(captured), spec), meta
                    except ValueError:
                        if isinstance(question, ScoreQuestion) and captured in question.criteria:
                            idx = float(question.criteria.index(captured))
                            return idx, self._one_hot(question, captured, spec), meta
                        return captured, None, meta
                return str(captured), self._one_hot(question, str(captured), spec), meta
            # regex miss → fall through

        if spec and spec.keywords:
            meta["strategy"] = "keywords"
            for rule in spec.keywords:
                if rule.match(text):
                    return (
                        rule.label,
                        self._one_hot(question, rule.label, spec),
                        meta,
                    )

        majority = (
            spec.majority_class
            if spec and spec.majority_class is not None
            else self.config.default_majority
        )
        if majority is not None:
            meta["strategy"] = "majority_class"
            if kind == "noul":
                return float(majority), None, meta
            return majority, self._one_hot(question, majority, spec), meta

        if kind == "noul":
            meta["strategy"] = "noul_default"
            default = spec.noul_default if spec else 0.0
            return float(default), None, meta

        if isinstance(question, ChoiceQuestion):
            first = next(iter(question.criteria.keys()))
            meta["strategy"] = "first_option_fallback"
            return first, self._one_hot(question, first, spec), meta

        if isinstance(question, ScoreQuestion):
            meta["strategy"] = "score_zero_fallback"
            return 0.0, self._one_hot(question, question.criteria[0], spec), meta

        raise ValueError(f"no trivial strategy for question kind={kind}")

    def _one_hot(
        self,
        question: Question,
        value: str | float,
        spec: TrivialQuestionSpec | None,
    ) -> dict[str, float] | None:
        if spec is not None and not spec.emit_one_hot:
            return None
        if isinstance(question, ChoiceQuestion):
            keys = list(question.criteria.keys())
            label = str(value)
            return {k: (1.0 if k == label else 0.0) for k in keys}
        if isinstance(question, ScoreQuestion):
            if isinstance(value, str) and value in question.criteria:
                return {lbl: (1.0 if lbl == value else 0.0) for lbl in question.criteria}
            # numeric expectation — one-hot nearest index without claiming magnitude
            if isinstance(value, (int, float)):
                # Exact integer index only — no rounding toward a level.
                fv = float(value)
                if fv != int(fv):
                    return None
                idx = int(fv)
                if idx < 0 or idx >= len(question.criteria):
                    return None
                return {
                    lbl: (1.0 if i == idx else 0.0)
                    for i, lbl in enumerate(question.criteria)
                }
        return None

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        text = _state_text(request.state)
        decisions: list[Decision] = []
        for key, question in request.questions.items():
            kind = question_kind(question)
            spec = self.config.specs.get(key)
            t0 = time.perf_counter()
            try:
                value, probs, meta = self._predict(text, question, spec)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                conf: float | None
                if kind == "noul":
                    conf = None
                    value = float(value)
                elif kind == "score":
                    # Expectation over level indices — never interpolate magnitudes.
                    if isinstance(value, str) and isinstance(question, ScoreQuestion):
                        if value not in question.criteria:
                            raise ValueError(
                                f"score label {value!r} not in criteria {question.criteria}"
                            )
                        value = float(question.criteria.index(value))
                    else:
                        value = float(value)
                    conf = (
                        1.0
                        if probs and max(probs.values()) >= 1.0 - 1e-12
                        else (max(probs.values()) if probs else None)
                    )
                else:
                    value = str(value)
                    conf = (
                        1.0
                        if probs and max(probs.values()) >= 1.0 - 1e-12
                        else (max(probs.values()) if probs else None)
                    )
                decisions.append(
                    Decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=kind,
                        value=value,
                        probabilities=probs,
                        confidence=conf,
                        latency_ms=latency_ms,
                        input_tokens=0,
                        output_tokens=0,
                        resolved_model=self.config.resolved_model,
                        serving_path=self.serving_path,
                        attempt=1,
                        error=None,
                        raw={"trivial": meta, "state_chars": len(text)},
                    )
                )
            except Exception as exc:  # noqa: BLE001
                latency_ms = (time.perf_counter() - t0) * 1000.0
                decisions.append(
                    error_decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=kind,
                        latency_ms=latency_ms,
                        resolved_model=self.config.resolved_model,
                        serving_path=self.serving_path,
                        attempt=1,
                        error=f"{type(exc).__name__}: {exc}",
                        raw={"exception": repr(exc)},
                    )
                )
        return decisions
