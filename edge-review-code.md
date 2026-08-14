# Code review package — free cred

Repository: `tariklayoudi-wq/tokiner`  
Branch: `dev`  
Scope: local MVP only. The implementation uses simulated providers only; it must not make network calls, use tokens, commit, push, or deploy.

## Review request

Review the code below. Do not infer anything not shown here. Return:

1. `APPROVATO`, `RICHIEDE CORREZIONI`, or `BLOCCATO`;
2. findings ordered by severity, with file and function;
3. missing tests, security risks, and technical debt;
4. a short, copy-ready GitHub Copilot Desktop brief for the smallest corrective increment, including acceptance criteria and test command. Do not propose real provider integration in this cycle.

## Test evidence

`python -m pytest -q` reported `11 passed`.

Covered cases: health; successful route; fallback; 503; absence of token in a response; clock-controlled circuit recovery; selection; preventive rotation; deterministic unknown quota; retry/fallback; circuit opening.

## `free_cred/core.py`

```python
from __future__ import annotations
from typing import Optional, Protocol, List, Callable, Any, Dict
from pydantic import BaseModel
from datetime import datetime
import time
from enum import Enum


class QuotaState(BaseModel):
    remaining: Optional[int] = None
    limit: Optional[int] = None
    reset: Optional[datetime] = None


class Provider(Protocol):
    id: str

    def get_quota(self) -> QuotaState: ...

    def call(self, *args, **kwargs) -> Any: ...


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
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
        if self.state == CircuitState.HALF_OPEN:
            self._to_open()
            return
        self.failures += 1
        if self.state == CircuitState.CLOSED and self.failures >= self.max_failures:
            self._to_open()

    def is_open(self) -> bool:
        if self.state == CircuitState.OPEN:
            if self.opened_at is not None and (self.clock() - self.opened_at) >= self.reset_timeout:
                self.state = CircuitState.HALF_OPEN
                self.opened_at = None
                return False
            return True
        return False


class Router:
    def __init__(self, providers: List[Provider], preventive_threshold: float = 0.1,
                 backoff: Callable[[int], float] = lambda n: 0.1 * (2 ** n),
                 max_retries: int = 2, cb_factory: Callable[[], CircuitBreaker] = lambda: CircuitBreaker()):
        if not providers:
            raise ValueError("providers list must not be empty")
        self.providers = providers
        self.threshold = preventive_threshold
        self.backoff = backoff
        self.max_retries = max_retries
        self.cbs: Dict[str, CircuitBreaker] = {p.id: cb_factory() for p in providers}

    def _eligible_providers(self) -> List[Provider]:
        return [p for p in sorted(self.providers, key=lambda p: p.id) if not self.cbs[p.id].is_open()]

    def _below_threshold(self, q: QuotaState) -> bool:
        if q.limit and q.remaining is not None:
            try:
                return (q.remaining / q.limit) <= self.threshold
            except Exception:
                return True
        return False

    def select(self) -> Provider:
        eligible = self._eligible_providers()
        if not eligible:
            eligible = sorted(self.providers, key=lambda p: p.id)
        for p in eligible:
            if not self._below_threshold(p.get_quota()):
                return p
        return eligible[0]

    def call(self, *args, **kwargs) -> Any:
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
            for attempt in range(0, self.max_retries + 1):
                try:
                    result = p.call(*args, **kwargs)
                    cb.record_success()
                    return result
                except Exception as exc:
                    last_exc = exc
                    cb.record_failure()
                    if attempt < self.max_retries:
                        time.sleep(self.backoff(attempt))
                    else:
                        break
            tried.add(p.id)
        if last_exc:
            raise RuntimeError(str(last_exc))
        raise RuntimeError("No providers available")
```

## `free_cred/api.py`

```python
import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .schemas import RouteRequest, RouteResponse, ErrorResponse
from .providers import providers_from_env
from .core import Router


def create_app(providers=None, preventive_threshold: float = 0.1, max_retries: int = 2, cb_factory=None) -> FastAPI:
    if providers is None:
        providers = providers_from_env()
    try:
        env_thr = float(os.environ.get("THRESHOLD", str(preventive_threshold)))
    except Exception:
        env_thr = preventive_threshold
    if cb_factory is None:
        cb_factory = lambda: None
    r = Router(
        providers,
        preventive_threshold=env_thr,
        max_retries=max_retries,
        cb_factory=lambda: __import__("free_cred.core", fromlist=["CircuitBreaker"]).CircuitBreaker(),
    )
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/route", response_model=RouteResponse, responses={503: {"model": ErrorResponse}})
    def route(req: RouteRequest):
        try:
            res = r.call(req.dict())
            provider_id = res.get("provider") if isinstance(res, dict) and "provider" in res else getattr(res, "provider", "unknown")
            return {"provider_id": provider_id, "result": res}
        except RuntimeError as e:
            return JSONResponse(status_code=503, content={"error": {"message": str(e)}})

    return app
```

## `free_cred/providers.py`

```python
from typing import List, Optional
import os
from .core import Provider, QuotaState


class MockProvider:
    def __init__(self, id: str, remaining: Optional[int] = None, limit: Optional[int] = None, fail: bool = False):
        self.id = id
        self._quota = QuotaState(remaining=remaining, limit=limit, reset=None)
        self.fail = fail
        self.calls = 0

    def get_quota(self) -> QuotaState:
        return self._quota

    def call(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"provider {self.id} simulated failure")
        return {"provider": self.id, "payload": args or kwargs}


def providers_from_env() -> List[Provider]:
    raw = os.environ.get("PROVIDERS", "a,b")
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    return [MockProvider(pid, remaining=None if i % 2 == 0 else 80, limit=None if i % 2 == 0 else 100) for i, pid in enumerate(ids)]
```

## `free_cred/schemas.py`

```python
from pydantic import BaseModel
from typing import Optional, Dict, Any


class RouteRequest(BaseModel):
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


class RouteResponse(BaseModel):
    provider_id: str
    result: Any


class ErrorResponse(BaseModel):
    error: Dict[str, Any]
```
