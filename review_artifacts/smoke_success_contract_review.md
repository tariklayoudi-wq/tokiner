# Review request - smoke success contract

Project: free cred / Hermes Agent credential rotation
Branch: dev
Date: 2026-08-14

## Context

The previous real NVIDIA/Nemotron smoke succeeded at the transport/router level,
but the model did not reply with exactly `ok`. Edge reviewer already approved it
as a connectivity/router smoke. This patch makes that behavior explicit:
connectivity smoke succeeds when the provider returns non-empty content, while
strict exact-output checking remains available only when requested.

No key files were read into this artifact. Do not ask for secrets.

## Changed files

- `free_cred/smoke.py`
- `tests/test_smoke.py`

## Implementation summary

- `run_provider_smoke(...)` now returns `ok=True` when provider content is
  non-empty after trimming whitespace.
- Added optional `expect_exact` argument for strict demo/assertion use cases.
- CLI gained `--expect-exact`.
- CLI now prints `ok=true|false`.
- CLI returns exit code `1` only when the explicit or default smoke success
  contract fails; provider/config/runtime exceptions still exit `2` with only
  the exception class name.
- Existing anti-secret behavior is preserved: key file contents are not printed,
  and exception messages are not printed.

## Test evidence

Targeted:

```text
python -m pytest tests/test_smoke.py -q
11 passed
```

Full suite:

```text
python -m pytest -q
72 passed, 1 skipped, 1 warning
```

Known warning:

```text
StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
```

Whitespace:

```text
git diff --check
no whitespace errors
warning LF/CRLF Windows on README.md, free_cred/providers.py, pyproject.toml
```

## Reviewer request

Please review:

- Is the smoke success contract appropriate for real free-tier/provider checks?
- Does `--expect-exact` cover stricter demos without making connectivity smoke brittle?
- Are exit codes `0` success, `1` assertion/content failure, `2` runtime/config failure reasonable?
- Any security regressions around key file loading, stdout/stderr, or exception handling?

Verdict requested: APPROVATO / RICHIEDE CORREZIONI / BLOCCATO.
