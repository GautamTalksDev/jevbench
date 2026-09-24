"""BART-large-MNLI supervised in-domain reference (Amendment 11).

``facebook/bart-large-mnli`` is fine-tuned on MultiNLI. ChaosNLI's MNLI items
come from MNLI's development set, so on those items this model is a
**supervised, in-domain NLI reference** — not a zero-shot baseline.

Report MNLI and SNLI items separately. Do **not** put this arm in the same
comparison table as Jev. It answers: does a model trained for this task do
better?
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

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

DEFAULT_MODEL = "facebook/bart-large-mnli"
PAPER_ROLE = "supervised_in_domain_reference"


@dataclass
class BartMnliClientConfig:
    model_id: str = DEFAULT_MODEL
    serving_path: str = "bart_mnli"
    resolved_model: str | None = None
    device: str = "cpu"
    role: str = PAPER_ROLE


@dataclass
class BartMnliClient:
    """MNLI-supervised BART used only as an in-domain reference arm."""

    config: BartMnliClientConfig = field(default_factory=BartMnliClientConfig)
    _pipeline: Any = field(default=None, repr=False)

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def _get_pipeline(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        from transformers import pipeline  # type: ignore

        device = -1 if self.config.device == "cpu" else 0
        self._pipeline = pipeline(
            "zero-shot-classification",
            model=self.config.model_id,
            device=device,
        )
        return self._pipeline

    def release(self) -> None:
        """Drop weights so WSL2 can load the next local model."""
        self._pipeline = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        pipe = self._get_pipeline()
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
                out = pipe(text, candidate_labels=labels, multi_label=False)
                latency_ms = (time.perf_counter() - t0) * 1000.0
                labs: Sequence[str] = out["labels"]
                scores: Sequence[float] = out["scores"]
                probs = {str(l): float(s) for l, s in zip(labs, scores, strict=False)}
                # Ensure every requested label is present
                for lab in labels:
                    probs.setdefault(lab, 0.0)
                total = sum(probs[l] for l in labels) or 1.0
                probs = {l: probs[l] / total for l in labels}
                chosen = max(probs, key=probs.get)

                if kind == "noul":
                    decisions.append(
                        Decision(
                            item_id=request.item_id,
                            question_key=key,
                            kind="noul",
                            value=float(probs.get("yes", 0.0)),
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
                                "probs": probs,
                                "role": self.config.role,
                                "caveat": (
                                    "Fine-tuned on MultiNLI; ChaosNLI-MNLI items "
                                    "are in-domain. Report MNLI vs SNLI separately; "
                                    "do not compare head-to-head with Jev."
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
                                "probs": probs,
                                "role": self.config.role,
                                "caveat": (
                                    "Fine-tuned on MultiNLI; ChaosNLI-MNLI items "
                                    "are in-domain. Report MNLI vs SNLI separately; "
                                    "do not compare head-to-head with Jev."
                                ),
                            },
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("BartMnli failed key=%s err=%s", key, exc)
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
