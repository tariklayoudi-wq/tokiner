# free_cred/core.py - CircuitBreaker full review artifact

This artifact repeats the current `CircuitBreaker` implementation from `free_cred/core.py`
so the Edge review can inspect it without attachment-preview truncation.

```python
from enum import Enum
from typing import Callable, Optional
import time


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
```
