import pytest
from free_cred.core import Router, CircuitBreaker, CircuitState
from free_cred.core import QuotaState
from free_cred.api import create_app


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
        if self.fail and (self.fail_times == 0 or self.calls <= self.fail_times):
            raise RuntimeError(f"provider {self.id} simulated failure")
        return f"ok:{self.id}"


def test_prefers_higher_quota_and_skips_low():
    a = FakeProvider("a", remaining=1, limit=100)
    b = FakeProvider("b", remaining=80, limit=100)
    r = Router([a, b], preventive_threshold=0.1, backoff=lambda n: 0, max_retries=1)
    res = r.call()
    assert res == "ok:b"
    assert a.calls == 0


def test_fallback_when_all_below_threshold():
    a = FakeProvider("a", remaining=5, limit=100)
    b = FakeProvider("b", remaining=1, limit=100)
    r = Router([a, b], preventive_threshold=0.2, backoff=lambda n: 0, max_retries=0)
    res = r.call()
    assert res == "ok:a"  # deterministic fallback to first by id


def test_all_below_threshold_both_called_once():
    # both below threshold; first fails, second succeeds; both should be called once
    p1 = FakeProvider("a", remaining=1, limit=100, fail=True)
    p2 = FakeProvider("b", remaining=1, limit=100)
    r = Router([p1, p2], preventive_threshold=0.2, backoff=lambda n: 0, max_retries=0)
    res = r.call()
    assert res == "ok:b"
    assert p1.calls == 1
    assert p2.calls == 1


def test_unknown_quota_selected():
    a = FakeProvider("a", remaining=None, limit=None)
    b = FakeProvider("b", remaining=1, limit=100)
    r = Router([a, b], preventive_threshold=0.1, backoff=lambda n: 0, max_retries=0)
    res = r.call()
    assert res == "ok:a"


def test_retry_stops_when_circuit_opens():
    p1 = FakeProvider("a", fail=True)
    p2 = FakeProvider("b")
    def cb_factory():
        return CircuitBreaker(max_failures=1, reset_timeout=100)
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=3, cb_factory=cb_factory)
    res = r.call()
    assert res == "ok:b"
    assert p1.calls == 1  # no retries after circuit opened


def test_create_app_cb_factory_invoked():
    calls = {"n": 0}
    def spy_factory():
        calls["n"] += 1
        return CircuitBreaker()
    a = FakeProvider("a")
    b = FakeProvider("b")
    # create_app should pass cb_factory through to Router which will invoke it per provider
    app = create_app(providers=[a, b], cb_factory=spy_factory)
    assert calls["n"] == 2


def test_half_open_probe_success_and_failure():
    t = {"now": 0.0}
    def clock():
        return t["now"]

    def cb_factory():
        return CircuitBreaker(max_failures=1, reset_timeout=10, clock=clock)

    p1 = FakeProvider("a", fail=True, fail_times=1)
    p2 = FakeProvider("b")
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=0, cb_factory=cb_factory)

    # first call: p1 fails once and opens; p2 used
    res1 = r.call()
    assert res1 == "ok:b"
    assert p1.calls == 1

    # advance time to HALF_OPEN
    t["now"] += 11

    # probe success: make p1 healthy
    p1.fail = False
    res2 = r.call()
    assert res2 == "ok:a"
    assert p1.calls == 2
    assert r.cbs[p1.id].state == CircuitState.CLOSED

    # re-open by making it fail again
    p1.fail = True
    p1.fail_times = 10
    # cause failure to open again
    res3 = r.call()
    assert res3 == "ok:b"
    assert r.cbs[p1.id].state == CircuitState.OPEN

    # advance to HALF_OPEN, probe fails and should re-open and continue
    t["now"] += 11
    p1.fail = True
    p1.fail_times = 10
    res4 = r.call()
    assert res4 == "ok:b"
    assert r.cbs[p1.id].state == CircuitState.OPEN


def test_half_open_probe_no_retry_with_max_retries_3():
    t = {"now": 0.0}
    def clock():
        return t["now"]

    def cb_factory():
        return CircuitBreaker(max_failures=1, reset_timeout=10, clock=clock)

    p1 = FakeProvider("a", fail=True, fail_times=1)
    p2 = FakeProvider("b")
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=3, cb_factory=cb_factory)

    # first call opens p1
    res1 = r.call()
    assert res1 == "ok:b"
    assert p1.calls == 1

    # advance to HALF_OPEN
    t["now"] += 11

    # set p1 to fail (probe should attempt only once despite max_retries=3)
    p1.fail = True
    p1.fail_times = 10
    res2 = r.call()
    assert res2 == "ok:b"
    assert p1.calls == 2  # only one probe call


def test_open_skip_and_fallback():
    p1 = FakeProvider("a")
    p2 = FakeProvider("b")
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=0)
    # manually open p1
    r.cbs[p1.id].state = CircuitState.OPEN
    res = r.call()
    assert res == "ok:b"


def test_all_open_raises_message_and_no_calls():
    p1 = FakeProvider("a")
    p2 = FakeProvider("b")
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=0)
    r.cbs[p1.id].state = CircuitState.OPEN
    r.cbs[p2.id].state = CircuitState.OPEN
    with pytest.raises(RuntimeError) as ei:
        r.call()
    assert str(ei.value) == "All providers are in OPEN state"
    assert p1.calls == 0 and p2.calls == 0


def test_all_open_via_failures_and_no_calls_before_timeout():
    t = {"now": 0.0}
    def clock():
        return t["now"]

    def cb_factory():
        return CircuitBreaker(max_failures=1, reset_timeout=10, clock=clock)

    p1 = FakeProvider("a", fail=True)
    p2 = FakeProvider("b", fail=True)
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=0, cb_factory=cb_factory)

    # first call will attempt providers and open their circuits
    with pytest.raises(RuntimeError):
        r.call()
    calls_after_open = (p1.calls, p2.calls)

    # immediate second call before timeout: should raise All providers OPEN and not call providers again
    with pytest.raises(RuntimeError) as ei:
        r.call()
    assert str(ei.value) == "All providers are in OPEN state"
    assert (p1.calls, p2.calls) == calls_after_open


def test_cb_factory_invoked_for_each_provider():
    calls = {"n": 0}
    def spy_factory():
        calls["n"] += 1
        return CircuitBreaker()
    p1 = FakeProvider("a")
    p2 = FakeProvider("b")
    r = Router([p1, p2], cb_factory=spy_factory)
    assert calls["n"] == 2


def test_duplicate_id_rejected():
    p1 = FakeProvider("x")
    p2 = FakeProvider("x")
    with pytest.raises(ValueError) as ei:
        Router([p1, p2])
    assert "duplicate provider id" in str(ei.value)


def test_runtime_error_preserves_cause():
    p1 = FakeProvider("a", fail=True)
    p2 = FakeProvider("b", fail=True)
    r = Router([p1, p2], backoff=lambda n: 0, max_retries=0)
    with pytest.raises(RuntimeError) as ei:
        r.call()
    # ensure the raised RuntimeError preserves the original cause
    assert ei.value.__cause__ is not None
    assert "provider" in str(ei.value.__cause__)


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_threshold_must_be_finite_and_in_range(threshold):
    with pytest.raises(ValueError, match="preventive_threshold"):
        Router([FakeProvider("a")], preventive_threshold=threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_threshold_accepts_inclusive_boundaries(threshold):
    router = Router([FakeProvider("a")], preventive_threshold=threshold)
    assert router.threshold == threshold
