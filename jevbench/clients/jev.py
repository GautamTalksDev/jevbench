"""Native TypeSafe Jev client (typesafe-sdk).

Supports native ``api.typesafe.ai`` and a configurable gateway base URL.
Scored latency must use native; gateway path is recorded in ``serving_path``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jevbench.clients.base import (
    Decision,
    RetryPolicy,
    SystemOneRequest,
    error_decision,
    question_kind,
)
from jevbench.clients.parse import build_sdk_questions, parse_response

logger = logging.getLogger(__name__)

NATIVE_BASE_URL = "https://api.typesafe.ai"


@dataclass
class JevClientConfig:
    """Path selection and retry policy for scored / exploratory runs."""

    model: str = "jev-1.13.0"
    # "native" → api.typesafe.ai; "gateway" → base_url override (e.g. Vercel)
    serving_path: str = "native"
    base_url: str | None = None  # required when serving_path == "gateway"
    api_key: str | None = None
    retry: RetryPolicy | None = None


class JevClient:
    """Wraps ``typesafe_sdk.TypeSafeClient`` behind the shared Decision interface."""

    def __init__(
        self,
        config: JevClientConfig | None = None,
        *,
        transport: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config or JevClientConfig()
        self.retry = self.config.retry or RetryPolicy()
        self._sleep = sleep or time.sleep
        self._transport = transport
        self._sdk_client: Any | None = None

        if self.config.serving_path == "gateway" and not self.config.base_url:
            raise ValueError(
                "JevClientConfig.base_url is required when serving_path='gateway'"
            )
        if self.config.serving_path not in ("native", "gateway"):
            raise ValueError(
                f"serving_path must be 'native' or 'gateway', got {self.config.serving_path!r}"
            )

    @property
    def serving_path(self) -> str:
        return self.config.serving_path

    def _resolve_base_url(self) -> str | None:
        if self.config.serving_path == "native":
            return NATIVE_BASE_URL
        return self.config.base_url

    def _get_sdk(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        from typesafe_sdk import TypeSafeClient  # type: ignore

        kwargs: dict[str, Any] = {}
        base = self._resolve_base_url()
        if base is not None:
            kwargs["base_url"] = base
        if self.config.api_key is not None:
            kwargs["api_key"] = self.config.api_key
        else:
            from jevbench.envload import require_typesafe_api_key

            kwargs["api_key"] = require_typesafe_api_key()
        self._sdk_client = TypeSafeClient(**kwargs)
        return self._sdk_client

    def _call_once(self, request: SystemOneRequest) -> Any:
        if self._transport is not None:
            return self._transport(request)
        client = self._get_sdk()
        questions = build_sdk_questions(request.questions)
        return client.system_one(
            state=request.state,
            questions=questions,
            model=request.model or self.config.model,
        )

    def _is_rate_limit(self, exc: BaseException) -> bool:
        name = type(exc).__name__
        if name == "RateLimitError":
            return True
        # Duck-type HTTP 429
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        return status == 429

    def _http_status(self, exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
        try:
            return int(status) if status is not None else None
        except (TypeError, ValueError):
            return None

    def _is_retryable(self, exc: BaseException) -> bool:
        """Amendment 10: never retry 4xx (incl. 429). Retry 5xx / connection only."""
        status = self._http_status(exc)
        if status is not None and 400 <= status < 500:
            return False
        if self._is_rate_limit(exc):
            return bool(self.retry.retry_429)
        name = type(exc).__name__.lower()
        if any(k in name for k in ("timeout", "connection", "network")):
            return True
        if status is not None and status >= 500:
            return True
        return False

    def _retry_after_s(self, exc: BaseException, attempt: int) -> float:
        header = None
        for attr in ("retry_after", "retry-after"):
            if hasattr(exc, attr):
                header = getattr(exc, attr)
                break
        headers = getattr(exc, "headers", None) or {}
        if header is None and isinstance(headers, dict):
            header = headers.get("retry-after") or headers.get("Retry-After")
        if header is not None:
            try:
                return float(header)
            except (TypeError, ValueError):
                pass
        # Exponential backoff
        delay = min(
            self.retry.max_delay_s,
            self.retry.base_delay_s * (2 ** (attempt - 1)),
        )
        return delay

    def decide(self, request: SystemOneRequest) -> list[Decision]:
        last_exc: BaseException | None = None
        t0 = time.perf_counter()
        attempt = 0

        for attempt in range(1, self.retry.max_attempts + 1):
            try:
                call_t0 = time.perf_counter()
                response = self._call_once(request)
                latency_ms = (time.perf_counter() - call_t0) * 1000.0
                return parse_response(
                    request,
                    response,
                    latency_ms=latency_ms,
                    serving_path=self.serving_path,
                    attempt=attempt,
                    fallback_model=request.model or self.config.model,
                )
            except Exception as exc:  # noqa: BLE001 — recorded, never silent
                last_exc = exc
                latency_so_far = (time.perf_counter() - t0) * 1000.0
                if self._is_retryable(exc) and attempt < self.retry.max_attempts:
                    delay = self._retry_after_s(exc, attempt)
                    logger.warning(
                        "Retryable error on attempt %s/%s; sleeping %.2fs "
                        "(4xx never retried). item_id=%s err=%s",
                        attempt,
                        self.retry.max_attempts,
                        delay,
                        request.item_id,
                        exc,
                    )
                    self._sleep(delay)
                    continue
                # Non-retryable, or attempts exhausted — emit error Decisions
                logger.error(
                    "JevClient call failed attempt=%s item_id=%s err=%s",
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
                        raw={"exception": repr(exc)},
                    )
                    for key, q in request.questions.items()
                ]

        # Unreachable, but keep mypy happy
        assert last_exc is not None
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return [
            error_decision(
                item_id=request.item_id,
                question_key=key,
                kind=question_kind(q),
                latency_ms=latency_ms,
                resolved_model=request.model or self.config.model,
                serving_path=self.serving_path,
                attempt=attempt,
                error=f"{type(last_exc).__name__}: {last_exc}",
                raw={"exception": repr(last_exc)},
            )
            for key, q in request.questions.items()
        ]
