"""Token-bucket rate limiter for System One scored runs.

Published ceilings (TypeSafe docs): 250_000 tokens/sec, 1_200 requests/min.
Defaults run at 80% of those ceilings because docs warn limits shift under load.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

# Published ceilings
PUBLISHED_TOKENS_PER_SEC = 250_000.0
PUBLISHED_REQUESTS_PER_MIN = 1_200.0
DEFAULT_HEADROOM = 0.80  # 20% under ceiling


@dataclass
class RateLimitConfig:
    tokens_per_sec: float = PUBLISHED_TOKENS_PER_SEC * DEFAULT_HEADROOM
    requests_per_min: float = PUBLISHED_REQUESTS_PER_MIN * DEFAULT_HEADROOM
    # Burst capacity as multiples of the sustained rate (1s / 1s of budget)
    token_burst: float | None = None
    request_burst: float | None = None

    def __post_init__(self) -> None:
        if self.token_burst is None:
            self.token_burst = self.tokens_per_sec
        if self.request_burst is None:
            # Allow a short burst of ~1 second of request budget
            self.request_burst = self.requests_per_min / 60.0


class TokenBucketLimiter:
    """Thread-safe dual bucket: request rate + token rate."""

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self._lock = threading.Lock()
        now = time.monotonic()
        self._tokens = float(self.config.token_burst or self.config.tokens_per_sec)
        self._requests = float(self.config.request_burst or (self.config.requests_per_min / 60.0))
        self._last = now

    def _refill(self, now: float) -> None:
        elapsed = max(0.0, now - self._last)
        self._last = now
        self._tokens = min(
            float(self.config.token_burst or self.config.tokens_per_sec),
            self._tokens + elapsed * self.config.tokens_per_sec,
        )
        req_rate = self.config.requests_per_min / 60.0
        self._requests = min(
            float(self.config.request_burst or req_rate),
            self._requests + elapsed * req_rate,
        )

    def acquire(self, *, tokens: int = 0, requests: int = 1) -> float:
        """Block until budget is available. Returns seconds waited."""
        waited = 0.0
        need_tok = max(0.0, float(tokens))
        need_req = max(0.0, float(requests))
        while True:
            with self._lock:
                now = time.monotonic()
                self._refill(now)
                if self._tokens >= need_tok and self._requests >= need_req:
                    self._tokens -= need_tok
                    self._requests -= need_req
                    return waited
                # Time until enough of each resource
                waits: list[float] = []
                if self._tokens < need_tok:
                    deficit = need_tok - self._tokens
                    waits.append(deficit / max(self.config.tokens_per_sec, 1e-9))
                if self._requests < need_req:
                    deficit = need_req - self._requests
                    req_rate = self.config.requests_per_min / 60.0
                    waits.append(deficit / max(req_rate, 1e-9))
                sleep_for = max(waits) if waits else 0.01
            time.sleep(min(sleep_for, 0.5))
            waited += min(sleep_for, 0.5)
