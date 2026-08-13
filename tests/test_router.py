import pytest
from datetime import datetime, timedelta
from free_cred.core import QuotaState, Router, CircuitBreaker, Provider


class FakeProvider:
    def __init__(self, id, remaining=None, limit=None, fail=False, fail_times=0):
        self.id = id
        self._quota = QuotaState(remaining=remaining, limit=limit, reset=None)
        self.fail = fail
        self.calls = 0
        self.fail_times = fail_times

    def get_quota(self):
        return self._quota

    def call(self, *args, **kwargs):
        self.calls += 1
        if self.fail and self.calls <= self.fail_times:
            raise RuntimeError(f"provider {self.id} simulated failure")
        return f"ok:{self.id}"


def test_selection_prefers_higher_remaining():
    a = FakeProvider('a', remaining=80, limit=100)
    b = FakeProvider('b', remaining=10, limit=100)
    r = Router([a, b], preventive_threshold=0.2)
    sel = r.select()
    assert sel.id == 'a'


def test_preventive_rotation_when_below_threshold():
    a = FakeProvider('a', remaining=25, limit=100)
    b = FakeProvider('b', remaining=90, limit=100)
    r = Router([a, b], preventive_threshold=0.3)
    sel = r.select()
    assert sel.id == 'b'


def test_unknown_quota_selected_deterministically():
    a = FakeProvider('a', remaining=None, limit=None)
    b = FakeProvider('b', remaining=10, limit=100)
    r = Router([b, a], preventive_threshold=0.1)
    # deterministic by id: a then b -> a should be selected because unknown treated as acceptable
    sel = r.select()
    assert sel.id == 'a'


def test_failure_and_fallback_with_retries():
    a = FakeProvider('a', remaining=80, limit=100, fail=True, fail_times=2)
    b = FakeProvider('b', remaining=80, limit=100)
    # backoff small for test
    r = Router([a, b], backoff=lambda n: 0.01, max_retries=1)
    # a will fail on first call attempts (fail_times=2) and retries=1 -> both attempts fail -> fallback to b
    res = r.call()
    assert res == 'ok:b'


def test_circuit_breaker_opens_and_skips_provider():
    a = FakeProvider('a', remaining=80, limit=100, fail=True, fail_times=5)
    b = FakeProvider('b', remaining=80, limit=100)
    # small circuit breaker for test
    cb_factory = lambda: CircuitBreaker(max_failures=2, reset_timeout=1)
    r = Router([a, b], backoff=lambda n: 0.01, max_retries=0, cb_factory=cb_factory)
    # first call: a fails -> record failures and then b succeeds
    res1 = r.call()
    assert res1 == 'ok:b'
    # cause a to hit circuit open by calling until it opens
    with pytest.raises(RuntimeError):
        # attempt to call when both providers fail by artificially setting b to fail
        a.fail = True
        a.fail_times = 10
        b.fail = True
        r.call()
