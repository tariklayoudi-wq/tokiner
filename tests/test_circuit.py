from free_cred.core import CircuitBreaker, CircuitState


def test_circuit_breaker_clock_controlled():
    t = {"now": 0.0}

    def clock():
        return t["now"]

    cb = CircuitBreaker(max_failures=2, reset_timeout=10, clock=clock)
    assert not cb.is_open()
    assert cb.state == CircuitState.CLOSED

    # cause failures to open the circuit
    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.is_open()

    # advance time to force transition to HALF_OPEN
    t["now"] += 11
    assert not cb.is_open()
    assert cb.state == CircuitState.HALF_OPEN

    # failure in HALF_OPEN re-opens
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    # advance again and then success closes
    t["now"] += 11
    assert not cb.is_open()
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
