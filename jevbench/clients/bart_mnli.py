"""BART-large-MNLI supervised in-domain reference (Amendment 11).

``facebook/bart-large-mnli`` is fine-tuned on MultiNLI. ChaosNLI's MNLI items
come from MNLI's development set, so on those items this model is a
**supervised, in-domain NLI reference** — not a zero-shot baseline.

Report MNLI and SNLI items separately. Do **not** put this arm in the same
comparison table as Jev. It answers: does a model trained for this task do
better?

Two clients live here:

- ``BartMnliClient`` (type ``bart_mnli``): the original zero-shot-classification
  pipeline that incorrectly scored ``str(state)``. Retained for provenance of
  the broken EXP-1 baseline run.
- ``BartMnliNliClient`` (type ``bart_mnli_nli``): post-data fix that runs true
  sequence-classification NLI on ``(premise, hypothesis)`` from the hydrated
  state.
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


def premise_hypothesis_from_state(state: Any) -> tuple[str, str]:
    """Extract premise/hypothesis the same way hydrated ChaosNLI items carry them."""
    if not isinstance(state, dict):
        raise TypeError(
            f"bart_mnli_nli requires a dict state with premise/hypothesis; "
            f"got {type(state).__name__}"
        )
    premise = state.get("premise")
    hypothesis = state.get("hypothesis")
    if not isinstance(premise, str) or not premise.strip():
        raise ValueError("state missing non-empty premise")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("state missing non-empty hypothesis")
    return premise, hypothesis


def map_id2label_probs(
    logits: Any, id2label: dict[Any, str]
) -> dict[str, float]:
    """Softmax logits and map via ``model.config.id2label`` to NLI label names."""
    import torch

    if not isinstance(logits, torch.Tensor):
        logits = torch.asarray(logits)
    if logits.ndim == 2:
        logits = logits[0]
    probs_t = torch.softmax(logits.float(), dim=-1)
    out: dict[str, float] = {}
    for idx, p in enumerate(probs_t.tolist()):
        name = str(id2label.get(idx, id2label.get(str(idx), idx))).lower()
        out[name] = float(p)
    canon = {
        "entailment": out.get("entailment", 0.0),
        "neutral": out.get("neutral", 0.0),
        "contradiction": out.get("contradiction", 0.0),
    }
    total = sum(canon.values()) or 1.0
    return {k: v / total for k, v in canon.items()}


@dataclass
class BartMnliClientConfig:
    model_id: str = DEFAULT_MODEL
    serving_path: str = "bart_mnli"
    resolved_model: str | None = None
    device: str = "cpu"
    role: str = PAPER_ROLE


@dataclass
class BartMnliClient:
    """MNLI-supervised BART used only as an in-domain reference arm.

    PROVENANCE ONLY - this class incorrectly feeds ``str(state)`` into the
    zero-shot-classification pipeline. Do not use for new scored runs.
    Prefer :class:`BartMnliNliClient`.
    """

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


@dataclass
class BartMnliNliClientConfig:
    model_id: str = DEFAULT_MODEL
    serving_path: str = "bart_mnli_nli"
    resolved_model: str | None = None
    device: str = "cpu"
    role: str = PAPER_ROLE


@dataclass
class BartMnliNliClient:
    """True NLI head: classify (premise, hypothesis) with bart-large-mnli."""

    config: BartMnliNliClientConfig = field(default_factory=BartMnliNliClientConfig)
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model
        import torch
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_id,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self._model.to(self.config.device)
        self._model.eval()
        return self._tokenizer, self._model

    def release(self) -> None:
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

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        import torch

        tok, mdl = self._load()
        resolved = self.config.resolved_model or self.config.model_id
        decisions: list[Decision] = []
        try:
            premise, hypothesis = premise_hypothesis_from_state(request.state)
        except (TypeError, ValueError) as exc:
            for key, question in request.questions.items():
                decisions.append(
                    error_decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=question_kind(question),
                        latency_ms=0.0,
                        resolved_model=resolved,
                        serving_path=self.serving_path,
                        attempt=1,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            return decisions

        id2label = {int(k): str(v) for k, v in dict(mdl.config.id2label).items()}

        for key, question in request.questions.items():
            kind = question_kind(question)
            if not isinstance(question, ChoiceQuestion):
                decisions.append(
                    error_decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=kind,
                        latency_ms=0.0,
                        resolved_model=resolved,
                        serving_path=self.serving_path,
                        attempt=1,
                        error=(
                            "bart_mnli_nli supports Choice questions only "
                            "(Noul/Score not applicable to the 3-way NLI head)"
                        ),
                    )
                )
                continue
            labels = list(question.criteria.keys())
            try:
                t0 = time.perf_counter()
                max_len = getattr(tok, "model_max_length", 1024) or 1024
                if max_len > 4096:
                    max_len = 1024
                encoded = tok(
                    premise,
                    hypothesis,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                )
                encoded = {k: v.to(self.config.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    logits = mdl(**encoded).logits
                latency_ms = (time.perf_counter() - t0) * 1000.0
                probs = map_id2label_probs(logits, id2label)
                for lab in labels:
                    probs.setdefault(lab, 0.0)
                total = sum(probs[l] for l in labels) or 1.0
                probs = {l: probs[l] / total for l in labels}
                chosen = max(probs, key=probs.get)
                decisions.append(
                    Decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind="choice",
                        value=chosen,
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
                            "id2label": {str(k): v for k, v in id2label.items()},
                            "role": self.config.role,
                            "encoding": "premise_hypothesis_pair",
                            "caveat": (
                                "Fine-tuned on MultiNLI; ChaosNLI-MNLI items "
                                "are in-domain. Report MNLI vs SNLI separately; "
                                "do not compare head-to-head with Jev."
                            ),
                        },
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("BartMnliNli failed key=%s err=%s", key, exc)
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
