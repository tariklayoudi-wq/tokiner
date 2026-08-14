# free_cred/core.py - Router full review artifact

This artifact repeats the current `Router` implementation from `free_cred/core.py`
so the Edge review can inspect it without attachment-preview truncation.

```python
from typing import Any, Callable, Dict, List
import math
import time

from free_cred.core import CircuitBreaker, CircuitState, Provider, QuotaState


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
                 cb_factory: Callable[[], CircuitBreaker] = lambda: CircuitBreaker()):
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
        # per-provider circuit breakers (invoke factory for each provider)
        self.cbs: Dict[str, CircuitBreaker] = {p.id: cb_factory() for p in providers}

    def _candidate_order(self) -> List[Provider]:
        # deterministic order by id
        ordered = sorted(self.providers, key=lambda p: p.id)
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
```
