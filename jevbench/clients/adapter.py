"""Fair LLM baseline via TypeSafe's system-one-adapter.

Exposes ``llm_answer_mode`` ("probabilities" | "discrete") for the required
ablation. Supports openai and anthropic providers.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jevbench.clients.base import (
    Decision,
    LlmAnswerMode,
    LlmProvider,
    RetryPolicy,
    SystemOneRequest,
    error_decision,
    question_kind,
)
from jevbench.clients.parse import build_adapter_questions, parse_response

logger = logging.getLogger(__name__)


@dataclass
class AdapterClientConfig:
    provider: LlmProvider = "openai"
    model: str = "gpt-4o-mini"
    llm_answer_mode: LlmAnswerMode = "probabilities"
    structured_outputs: bool = True
    normalize_probabilities: bool = True
    serving_path: str = "adapter"
    retry: RetryPolicy | None = None
    # Extra kwargs forwarded to SystemOneAdapterClient / system_one()
    extra: dict[str, Any] | None = None


class AdapterClient:
    """Wraps ``system_one_adapter.SystemOneAdapterClient``."""

    def __init__(
        self,
        config: AdapterClientConfig | None = None,
        *,
        transport: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or AdapterClientConfig()
        self.retry = self.config.retry or RetryPolicy()
        self._sleep = sleep or time.sleep
        self._transport = transport
        self._client: Any | None = None

        if self.config.llm_answer_mode not in ("probabilities", "discrete"):
            raise ValueError(
                f"llm_answer_mode must be 'probabilities' or 'discrete', "
                f"got {self.config.llm_answer_mode!r}"
            )
        if self.config.provider not in ("openai", "anthropic"):
            raise ValueError(
                f"provider must be 'openai' or 'anthropic', got {self.config.provider!r}"
            )

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from system_one_adapter import SystemOneAdapterClient  # type: ignore

        kwargs: dict[str, Any] = {
            "structured_outputs": self.config.structured_outputs,
            "llm_answer_mode": self.config.llm_answer_mode,
            "normalize_probabilities": self.config.normalize_probabilities,
        }
        if self.config.extra:
            kwargs.update(self.config.extra)
        self._client = SystemOneAdapterClient(**kwargs)
        return self._client

    def _call_once(self, request: SystemOneRequest) -> Any:
        if self._transport is not None:
            return self._transport(request)
        client = self._get_client()
        questions = build_adapter_questions(request.questions)
        return client.system_one(
            state=request.state,
            questions=questions,
            provider=self.config.provider,
            model=request.model or self.config.model,
        )

    def _is_rate_limit(self, exc: BaseException) -> bool:
        name = type(exc).__name__
        if "RateLimit" in name or name == "RateLimitError":
            return True
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status == 429

    def _retry_after_s(self, exc: BaseException, attempt: int) -> float:
        header = getattr(exc, "retry_after", None)
        headers = getattr(exc, "headers", None) or {}
        if header is None and isinstance(headers, dict):
            header = headers.get("retry-after") or headers.get("Retry-After")
        if header is not None:
            try:
                return float(header)
            except (TypeError, ValueError):
                pass
        return min(
            self.retry.max_delay_s,
            self.retry.base_delay_s * (2 ** (attempt - 1)),
        )

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        t0 = time.perf_counter()
        attempt = 0
        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                call_t0 = time.perf_counter()
                response = self._call_once(request)
                latency_ms = (time.perf_counter() - call_t0) * 1000.0
                decisions = parse_response(
                    request,
                    response,
                    latency_ms=latency_ms,
                    serving_path=self.serving_path,
                    attempt=attempt,
                    fallback_model=request.model or self.config.model,
                )
                # Stash ablation config on every raw payload for the manifest
                for d in decisions:
                    d.raw.setdefault("adapter_config", {})
                    d.raw["adapter_config"] = {
                        "provider": self.config.provider,
                        "llm_answer_mode": self.config.llm_answer_mode,
                        "structured_outputs": self.config.structured_outputs,
                        "normalize_probabilities": self.config.normalize_probabilities,
                        "model": request.model or self.config.model,
                    }
                return decisions
            except Exception as exc:  # noqa: BLE001
                latency_so_far = (time.perf_counter() - t0) * 1000.0
                if self._is_rate_limit(exc) and attempt < self.retry.max_attempts:
                    delay = self._retry_after_s(exc, attempt)
                    logger.warning(
                        "Adapter rate limit attempt %s/%s; sleep %.2fs item_id=%s",
                        attempt,
                        self.retry.max_attempts,
                        delay,
                        request.item_id,
                    )
                    self._sleep(delay)
                    continue
                logger.error(
                    "AdapterClient failed attempt=%s item_id=%s err=%s",
                    attempt,
                    request.item_id,
                    exc,
                )
                return [
                    error_decision(
                        item_id=request.item_id,
                        question_key=key,
                        kind=question_kind(q),
                        latency_ms=latency_so_far,
                        resolved_model=request.model or self.config.model,
                        serving_path=self.serving_path,
                        attempt=attempt,
                        error=f"{type(exc).__name__}: {exc}",
                        raw={
                            "exception": repr(exc),
                            "adapter_config": {
                                "provider": self.config.provider,
                                "llm_answer_mode": self.config.llm_answer_mode,
                            },
                        },
                    )
                    for key, q in request.questions.items()
                ]

        return []
