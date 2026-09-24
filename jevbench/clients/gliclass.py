"""GLiClass zero-shot classifier arm.

``knowledgator/gliclass-base-v1.0`` was trained on synthetic zero-shot data
(``MoritzLaurer/synthetic_zeroshot_mixtral_v0.1``). **The model card doesn't
list MNLI or SNLI** (verified). That is not a claim that it was not trained
on them. Do not silently substitute BART-MNLI (that arm is a separate
supervised reference).
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
    """Zero-shot multi-label/multi-class classifier control."""

    config: GLiClassClientConfig = field(default_factory=GLiClassClientConfig)
    scorer: GLiClassScorer | None = None
    _model: Any = field(default=None, repr=False)
    _tokenizer: Any = field(default=None, repr=False)

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def release(self) -> None:
        self.scorer = None
        self._model = None
        self._tokenizer = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _get_scorer(self) -> GLiClassScorer:
        if self.scorer is not None:
            return self.scorer
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "GLiClass arm requires `pip install gliclass transformers`. "
                "Do not substitute facebook/bart-large-mnli here — that is the "
                "separate supervised_in_domain_reference arm (Amendment 11)."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        model = GLiClassModel.from_pretrained(self.config.model_id)
        device = self.config.device
        model.to(device)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        pipe = ZeroShotClassificationPipeline(
            model, tokenizer, classification_type="multi-label", device=device
        )

        class _GLiScorer:
            def score(self, text: str, labels: Sequence[str]) -> dict[str, float]:
                results = pipe(text, list(labels), threshold=0.0)[0]
                out = {str(r["label"]): float(r["score"]) for r in results}
                for lab in labels:
                    out.setdefault(str(lab), 0.0)
                return out

        self.scorer = _GLiScorer()
        self.config.resolved_model = self.config.model_id
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
                            raw={
                                "scores": scores,
                                "probs": probs,
                                "training_note": (
                                    "Synthetic zero-shot mix "
                                    "(MoritzLaurer/synthetic_zeroshot_mixtral_v0.1). "
                                    "The model card doesn't list MNLI or SNLI "
                                    "(verified — not a claim it was untrained on them)."
                                ),
                            },
                        )
                    )
                else:
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind=kind,
                            value=chosen
                            if kind == "choice"
                            else float(labels.index(chosen)),
                            probabilities=probs,
                            confidence=max(probs.values()) if probs else None,
                            latency_ms=latency_ms,
                            input_tokens=0,
                            output_tokens=0,
                            resolved_model=resolved,
                            serving_path=self.serving_path,
                            attempt=1,
                            error=None,
                            raw={
                                "scores": scores,
                                "training_note": (
                                    "Synthetic zero-shot mix "
                                    "(MoritzLaurer/synthetic_zeroshot_mixtral_v0.1). "
                                    "The model card doesn't list MNLI or SNLI "
                                    "(verified — not a claim it was untrained on them)."
                                ),
                            },
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
                    )
                )
        return decisions
