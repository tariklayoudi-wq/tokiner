from __future__ import annotations
from typing import Optional, Protocol, List, Callable, Any, Dict
from pydantic import BaseModel
from datetime import datetime, timedelta
import time


class QuotaState(BaseModel):
    remaining: Optional[int] = None
    limit: Optional[int] = None
    reset: Optional[datetime] = None


class Provider(Protocol):
    id: str

    def get_quota(self) -> QuotaState: ...

    def call(self, *args, **kwargs) -> Any: ...


class CircuitBreaker:
    """Simple time-based circuit breaker.
    - opens after max_failures
    - stays open for reset_timeout seconds
    - automatic half-open after timeout (next call allowed)
    """

    def __init__(self, max_failures: int = 3, reset_timeout: int = 60):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.opened_at: Optional[float] = None

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.max_failures and self.opened_at is None:
            self.opened_at = time.time()

    def is_open(self) -> bool:
        if self.opened_at is None:
            return False
        if (time.time() - self.opened_at) > self.reset_timeout:
            # auto-reset (half-open)
            self.failures = 0
            self.opened_at = None
            return False
        return True


class Router:
    """Router that selects providers deterministically and handles retries, fallback and circuit breakers."""

    def __init__(self,
                 providers: List[Provider],
                 preventive_threshold: float = 0.1,
                 backoff: Callable[[int], float] = lambda n: 0.1 * (2 ** n),
                 max_retries: int = 2,
                 cb_factory: Callable[[], CircuitBreaker] = lambda: CircuitBreaker()):
        if not providers:
            raise ValueError("providers list must not be empty")
        self.providers = providers
        self.threshold = preventive_threshold
        self.backoff = backoff
        self.max_retries = max_retries
        # per-provider circuit breakers
        self.cbs: Dict[str, CircuitBreaker] = {p.id: cb_factory() for p in providers}

    def _eligible_providers(self) -> List[Provider]:
        # deterministic order by id
        ordered = sorted(self.providers, key=lambda p: p.id)
        # filter out open circuit breakers
        return [p for p in ordered if not self.cbs[p.id].is_open()]

    def _below_threshold(self, q: QuotaState) -> bool:
        if q.limit and q.remaining is not None:
            try:
                return (q.remaining / q.limit) <= self.threshold
            except Exception:
                return True
        # unknown quotas are considered unknown (not below threshold)
        return False

    def select(self) -> Provider:
        """Select first provider that is not below threshold; if all below, choose first eligible; unknown quotas are treated as acceptable and selected deterministically."""
        eligible = self._eligible_providers()
        if not eligible:
            # all circuited; fall back to all providers (even if open) to allow attempts
            eligible = sorted(self.providers, key=lambda p: p.id)
        # prefer providers with known good quota
        for p in eligible:
            q = p.get_quota()
            if not self._below_threshold(q):
                return p
        # fallback deterministic
        return eligible[0]

    def call(self, *args, **kwargs) -> Any:
        """Attempt call using selection + retries + fallback.
        On failure, mark circuit breaker failure and try next provider.
        """
        tried = set()
        last_exc = None
        providers_list = self._eligible_providers() or sorted(self.providers, key=lambda p: p.id)
        for p in providers_list:
            if p.id in tried:
                continue
            cb = self.cbs[p.id]
            if cb.is_open():
                tried.add(p.id)
                continue
            # try with retries
            for attempt in range(0, self.max_retries + 1):
                try:
                    result = p.call(*args, **kwargs)
                    cb.record_success()
                    return result
                except Exception as exc:
                    last_exc = exc
                    cb.record_failure()
                    if attempt < self.max_retries:
                        delay = self.backoff(attempt)
                        time.sleep(delay)
                    else:
                        # give up on this provider, try next
                        break
            tried.add(p.id)
        # all providers exhausted
        if last_exc:
            raise last_exc
        raise RuntimeError("No providers available")
