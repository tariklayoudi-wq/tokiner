from __future__ import annotations
from typing import Optional, Protocol, List, Callable, Any, Dict
from pydantic import BaseModel
from datetime import datetime
import math
import time


class QuotaState(BaseModel):
    remaining: Optional[int] = None
    limit: Optional[int] = None
    reset: Optional[datetime] = None


class Provider(Protocol):
    id: str

    def get_quota(self) -> QuotaState: ...

    def call(self, *args, **kwargs) -> Any: ...


from enum import Enum


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker with explicit states and injectable clock for deterministic tests.

    States:
    - CLOSED: normal operation
    - OPEN: calls are blocked until reset_timeout elapses
    - HALF_OPEN: one or more calls are allowed to probe recovery

    Behavior:
    - After max_failures in CLOSED -> OPEN (opened_at recorded)
    - When OPEN and (clock() - opened_at) >= reset_timeout -> HALF_OPEN
    - In HALF_OPEN: a failure immediately re-opens; a success closes and resets counters

    Clock injection: pass clock=callable() (defaults to time.time)
    """

    def __init__(self, max_failures: int = 3, reset_timeout: int = 60, clock: Callable[[], float] = time.time):
        self.max_failures = max_failures
        self.reset_timeout = reset_timeout
        self.clock = clock
        self.failures = 0
        self.state: CircuitState = CircuitState.CLOSED
        self.opened_at: Optional[float] = None

    def _to_open(self) -> None:
        self.state = CircuitState.OPEN
        self.opened_at = self.clock()
        self.failures = 0

    def record_success(self) -> None:
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.opened_at = None

    def record_failure(self) -> None:
        # In HALF_OPEN, any failure re-opens immediately
        if self.state == CircuitState.HALF_OPEN:
            self._to_open()
            return
        self.failures += 1
        if self.state == CircuitState.CLOSED and self.failures >= self.max_failures:
            self._to_open()

    def is_open(self) -> bool:
        if self.state == CircuitState.OPEN:
            # check if timeout expired -> move to HALF_OPEN
            if self.opened_at is not None and (self.clock() - self.opened_at) >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.opened_at = None
                return False
            return True
        # CLOSED or HALF_OPEN are not open (HALF_OPEN allows probing)
        return False


class Router:
    """Router that selects providers deterministically and handles retries, fallback and circuit breakers.

    Ordering rules (shared by select() and call()):
    - deterministic by provider.id
    - exclude providers whose circuit is OPEN
    - prefer providers NOT below preventive_threshold first
    - providers below threshold only considered as last resort
    """

    def __init__(self,
                 providers: List[Provider],
                 preventive_threshold: float = 0.1,
                 backoff: Callable[[int], float] = lambda n: 0.1 * (2 ** n),
                 max_retries: int = 2,
                 cb_factory: Callable[[], CircuitBreaker] = lambda: CircuitBreaker(),
                 priority_order: Optional[List[str]] = None):
        if not providers:
            raise ValueError("providers list must not be empty")
        # validate unique ids deterministically
        ids = [p.id for p in providers]
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        if dupes:
            # deterministic single-message error
            raise ValueError(f"duplicate provider id: '{dupes[0]}'")

        try:
            threshold = float(preventive_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("preventive_threshold must be a finite value between 0 and 1") from exc
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("preventive_threshold must be a finite value between 0 and 1")

        self.providers = providers
        self.threshold = threshold
        self.backoff = backoff
        self.max_retries = max_retries
        self.priority_order = priority_order or ["nvidia", "openvino", "copilot"]
        # per-provider circuit breakers (invoke factory for each provider)
        self.cbs: Dict[str, CircuitBreaker] = {p.id: cb_factory() for p in providers}

    def _candidate_order(self) -> List[Provider]:
        # deterministic explicit provider priority, then by id
        ordered = sorted(self.providers, key=lambda p: (self._priority_rank(p), p.id))
        # exclude providers with OPEN circuit
        non_open = [p for p in ordered if not self.cbs[p.id].is_open()]
        # split by threshold: first those NOT below threshold, then those below
        not_below = []
        below = []
        for p in non_open:
            try:
                q = p.get_quota()
            except Exception:
                q = QuotaState()
            if not self._below_threshold(q):
                not_below.append(p)
            else:
                below.append(p)
        return not_below + below

    def _priority_rank(self, provider: Provider) -> int:
        provider_id = provider.id.lower()
        for rank, prefix in enumerate(self.priority_order):
            if provider_id == prefix or provider_id.startswith(f"{prefix}-") or provider_id.startswith(f"{prefix}:"):
                return rank
        return len(self.priority_order)

    def _below_threshold(self, q: QuotaState) -> bool:
        if q.limit and q.remaining is not None:
            try:
                return (q.remaining / q.limit) <= self.threshold
            except Exception:
                return True
        # unknown quotas are considered unknown (not below threshold)
        return False

    def select(self) -> Provider:
        """Select first provider according to candidate ordering. If no candidates (all OPEN), raise RuntimeError with deterministic message."""
        candidates = self._candidate_order()
        if not candidates:
            raise RuntimeError("All providers are in OPEN state")
        return candidates[0]

    def call(self, *args, **kwargs) -> Any:
        """Attempt call using the shared candidate ordering, retries, and per-provider circuit rules.

        - HALF_OPEN: allow a single probe (no retries); success closes circuit, failure re-opens and move to next provider
        - CLOSED: allow up to max_retries+1 attempts, but if record_failure opened the circuit, stop retrying this provider
        - HTTP 401/403/404/429/500 and timeout-like exceptions fallback immediately without retrying the same provider
        - OPEN providers are excluded from candidates
        """
        last_exc = None
        candidates = self._candidate_order()
        if not candidates:
            raise RuntimeError("All providers are in OPEN state")

        for p in candidates:
            cb = self.cbs[p.id]
            # Determine allowed attempts: 1 for HALF_OPEN, else max_retries+1 for CLOSED
            if cb.state == CircuitState.HALF_OPEN:
                attempts_allowed = 1
            else:
                attempts_allowed = self.max_retries + 1

            for attempt in range(attempts_allowed):
                try:
                    result = p.call(*args, **kwargs)
                    cb.record_success()
                    return result
                except Exception as exc:
                    last_exc = exc
                    cb.record_failure()
                    if _should_fallback_without_retry(exc):
                        break
                    # if circuit became OPEN, stop retrying this provider and proceed
                    if cb.is_open() or cb.state == CircuitState.OPEN:
                        break
                    # otherwise, if we will retry this provider, sleep only then
                    if attempt < (attempts_allowed - 1):
                        delay = self.backoff(attempt)
                        if delay:
                            time.sleep(delay)
                    # else will proceed to next provider
            # move to next candidate
        if last_exc:
            raise RuntimeError(str(last_exc)) from last_exc
        raise RuntimeError("No providers available")


def _should_fallback_without_retry(exc: Exception) -> bool:
    status_code = _status_code_from_exception(exc)
    if status_code in {401, 403, 404, 429, 500}:
        return True
    name = exc.__class__.__name__.lower()
    return "timeout" in name or "timedout" in name


def _status_code_from_exception(exc: Exception) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None
