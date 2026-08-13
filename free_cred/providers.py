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
    """Create mock providers from PROVIDERS env var (comma-separated ids).
    Example: PROVIDERS=a,b
    """
    raw = os.environ.get("PROVIDERS", "a,b")
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[Provider] = []
    for i, pid in enumerate(ids):
        # stagger quotas for demo
        remaining = None if i % 2 == 0 else 80
        limit = None if remaining is None else 100
        out.append(MockProvider(pid, remaining=remaining, limit=limit))
    return out
