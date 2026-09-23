"""GLiClass zero-shot classifier arm for EXP-3 (moat control).

Off-the-shelf ``knowledgator/gliclass`` (or compatible) — not a Jev clone.
Maps label scores into the shared ``Decision`` shape so ECE can be compared
on identical items.

Optional dependency: if transformers / gliclass weights are unavailable,
inject a fake scorer for tests. Live EXP-3 requires a real local model.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from jevbench.clients.base import (
    ChoiceQuestion,
    Decision,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
    error_decision,
    question_kind,
)

logger = logging.getLogger(__name__)


class GLiClassScorer(Protocol):
    def score(
        self, text: str, labels: Sequence[str]
    ) -> dict[str, float]:
        """Return unnormalized scores or probabilities per label."""
        ...


@dataclass
class GLiClassClientConfig:
    model_id: str = "knowledgator/gliclass-base-v1.0"
    serving_path: str = "gliclass"
    resolved_model: str | None = None
    device: str = "cpu"


@dataclass
class GLiClassClient:
    """Zero-shot multi-label/multi-class classifier control (EXP-3)."""

    config: GLiClassClientConfig = field(default_factory=GLiClassClientConfig)
    scorer: GLiClassScorer | None = None

    def __post_init__(self) -> None:
        self._pipeline: Any = None

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def _get_scorer(self) -> GLiClassScorer:
        if self.scorer is not None:
            return self.scorer
        # Lazy load — heavy
        try:
            from transformers import pipeline  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "GLiClass arm requires transformers. Install it for live EXP-3, "
                "or inject scorer= for tests."
            ) from exc

        pipe = pipeline(
            "zero-shot-classification",
            model=self.config.model_id,
            device=self.config.device,
        )

        class _HFScorer:
            def score(self, text: str, labels: Sequence[str]) -> dict[str, float]:
                out = pipe(text, candidate_labels=list(labels), multi_label=False)
                labs = out["labels"]
                scores = out["scores"]
                return {str(l): float(s) for l, s in zip(labs, scores)}

        self.scorer = _HFScorer()
        return self.scorer

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        scorer = self._get_scorer()
        text = request.state if isinstance(request.state, str) else str(request.state)
        decisions: list[Decision] = []
        resolved = self.config.resolved_model or self.config.model_id

        for key, question in request.questions.items():
            kind = question_kind(question)
            try:
                if isinstance(question, ChoiceQuestion):
                    labels = list(question.criteria.keys())
                elif isinstance(question, NoulQuestion):
                    labels = ["yes", "no"]
                elif isinstance(question, ScoreQuestion):
                    labels = list(question.criteria)
                else:
                    raise TypeError(type(question))

                t0 = time.perf_counter()
                scores = scorer.score(text, labels)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                # Softmax-normalize if not already a distribution
                total = sum(max(0.0, float(v)) for v in scores.values()) or 1.0
                probs = {k: max(0.0, float(scores.get(k, 0.0))) / total for k in labels}
                chosen = max(probs, key=probs.get)

                if kind == "noul":
                    p_true = probs.get("yes", 0.0)
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind="noul",
                            value=float(p_true),
                            probabilities=None,
                            confidence=None,
                            latency_ms=latency_ms,
                            input_tokens=0,
                            output_tokens=0,
                            resolved_model=resolved,
                            serving_path=self.serving_path,
                            attempt=1,
                            error=None,
                            raw={"scores": scores, "probs": probs},
                        )
                    )
                else:
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind=kind,
                            value=chosen if kind == "choice" else float(
                                labels.index(chosen)
                            ),
                            probabilities=probs,
                            confidence=max(probs.values()) if probs else None,
                            latency_ms=latency_ms,
                            input_tokens=0,
                            output_tokens=0,
                            resolved_model=resolved,
                            serving_path=self.serving_path,
                            attempt=1,
                            error=None,
                            raw={"scores": scores},
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("GLiClass failed key=%s err=%s", key, exc)
                decisions.append(
                    error_decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=kind,
                        latency_ms=0.0,
                        resolved_model=resolved,
                        serving_path=self.serving_path,
                        attempt=1,
                        error=f"{type(exc).__name__}: {exc}",
                        raw={"exception": repr(exc)},
                    )
                )
        return decisions
