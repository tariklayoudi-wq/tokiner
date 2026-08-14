# Review artifact: final smoke harness correction

## Prior Edge decision

Decisione precedente: `RICHIEDE CORREZIONI`.

Remaining issue:

- `main()` caught exceptions but printed `str(exc)`, which could include
  uncontrolled SDK/provider details.

## Change made

### `free_cred/smoke.py`

Changed CLI error output from:

```python
parser.exit(status=2, message=f"smoke failed: {exc.__class__.__name__}: {exc}\n")
```

to:

```python
parser.exit(status=2, message=f"smoke failed: {exc.__class__.__name__}\n")
```

The CLI now reports the exception class only and does not print the exception
message.

### `tests/test_smoke.py`

Added:

```python
def test_smoke_cli_error_does_not_include_exception_message_secret(monkeypatch, capsys):
    sentinel = "SENTINEL_SECRET_VALUE"

    def factory(name, *, max_tokens=None, quota_source=None):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(smoke, "openai_compatible_providers_from_preset", factory)
    monkeypatch.setattr(sys, "argv", ["free_cred.smoke", "groq"])

    with pytest.raises(SystemExit) as exc:
        smoke.main()
    captured = capsys.readouterr()

    assert exc.value.code == 2
    assert "RuntimeError" in captured.err
    assert sentinel not in captured.out
    assert sentinel not in captured.err
```

## Previously corrected in the same cycle

- Removed inconsistent public `environ=` smoke parameter.
- Renamed "redacted summary" wording to "minimal smoke summary".
- Added anti-leak tests for summary and CLI stdout/stderr success path.
- Added smoke `max_tokens` validation: 1 through 64 inclusive.
- Documented fallback across credentials.
- Documented process-environment lifetime for loaded keys.

## Local test evidence

Command:

```powershell
python -m pytest -q
```

Result:

```text
70 passed, 1 skipped, 1 warning
```

Collected tests:

```text
tests/test_api.py: 5
tests/test_circuit.py: 1
tests/test_credentials.py: 10
tests/test_nvidia_provider.py: 4
tests/test_openai_compatible_providers.py: 17
tests/test_router.py: 5
tests/test_router_extra.py: 21
tests/test_smoke.py: 9
```

Command:

```powershell
git diff --check
```

Result:

```text
no whitespace errors; only Windows LF/CRLF warnings for README.md,
free_cred/providers.py, and pyproject.toml
```

## Requested Edge review decision

Please decide whether this final correction is sufficient to approve the local
real-provider smoke harness for one controlled smoke test.

Decision requested:

- APPROVATO
- RICHIEDE CORREZIONI
- BLOCCATO
