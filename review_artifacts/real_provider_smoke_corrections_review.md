# Review artifact: smoke harness corrections after Edge review

## Prior Edge decision

Decisione precedente: `RICHIEDE CORREZIONI`.

Requested fixes:

- remove or truly support the inconsistent `environ=` contract;
- rename misleading "redacted summary" wording;
- add explicit anti-leak tests for result and CLI stdout/stderr;
- validate unsafe `max_tokens`;
- document that router fallback can try multiple credentials;
- document that loaded keys remain in the process environment until exit.

## Changes made

### `free_cred/smoke.py`

- Removed public `environ=` from `build_router_for_provider_smoke()` and
  `run_provider_smoke()`.
- `load_api_keys_file()` now writes only to the current process environment for
  this smoke path, matching how real providers read credentials.
- Renamed docstring from "redacted smoke summary" to "minimal smoke summary".
- Added local smoke cap:
  - `max_tokens >= 1`
  - `max_tokens <= 64`
- Added CLI exception handling with a concise `smoke failed: ...` message.
- Kept smoke path through `Router.call()`.

Current implementation:

```python
from __future__ import annotations

import argparse
from typing import Any, Callable, Optional

from .core import Provider, QuotaState, Router
from .credentials import load_api_keys_file
from .providers import openai_compatible_providers_from_preset


ProviderFactory = Callable[..., list[Provider]]
MAX_SMOKE_TOKENS = 64


def _validate_smoke_max_tokens(max_tokens: int) -> int:
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if max_tokens > MAX_SMOKE_TOKENS:
        raise ValueError(f"max_tokens must be at most {MAX_SMOKE_TOKENS} for smoke tests")
    return max_tokens


def build_router_for_provider_smoke(
    provider_name: str,
    *,
    keys_file: Optional[str] = None,
    max_tokens: int = 8,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
    provider_factory: Optional[ProviderFactory] = None,
) -> Router:
    """Build a no-retry router for a local, opt-in real-provider smoke test."""

    max_tokens = _validate_smoke_max_tokens(max_tokens)
    if keys_file:
        load_api_keys_file(provider_name, keys_file, overwrite=True)
    factory = provider_factory or openai_compatible_providers_from_preset
    providers = factory(
        provider_name,
        max_tokens=max_tokens,
        quota_source=quota_source,
    )
    return Router(providers, backoff=lambda attempt: 0, max_retries=0)


def run_provider_smoke(
    provider_name: str,
    *,
    keys_file: Optional[str] = None,
    prompt: str = "Reply with one word: ok",
    max_tokens: int = 8,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
    provider_factory: Optional[ProviderFactory] = None,
) -> dict[str, Any]:
    """Call one provider through the router and return a minimal smoke summary."""

    router = build_router_for_provider_smoke(
        provider_name,
        keys_file=keys_file,
        max_tokens=max_tokens,
        quota_source=quota_source,
        provider_factory=provider_factory,
    )
    result = router.call({"prompt": prompt}, max_tokens=max_tokens)
    return {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "content": result.get("content"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local opt-in smoke test against one OpenAI-compatible provider."
    )
    parser.add_argument("provider", help="Provider preset, for example nvidia, groq, openrouter, gemini.")
    parser.add_argument("--keys-file", help="Local file outside the repo with one API key per line.")
    parser.add_argument("--prompt", default="Reply with one word: ok")
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()

    try:
        result = run_provider_smoke(
            args.provider,
            keys_file=args.keys_file,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
    except Exception as exc:
        parser.exit(status=2, message=f"smoke failed: {exc.__class__.__name__}: {exc}\n")
    print(f"provider={result['provider']}")
    print(f"model={result['model']}")
    print(f"content={result['content']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tests/test_smoke.py`

Added/updated tests:

- key-file load uses real process env for this smoke path and restores it after
  the test;
- summary does not include a sentinel fake secret;
- CLI stdout/stderr does not include the sentinel fake secret;
- `max_tokens` rejects `0`, `-1`, and `65`;
- quota rotation test remains in place.

Relevant added anti-leak assertions:

```python
assert sentinel not in repr(result)
assert sentinel not in captured.out
assert sentinel not in captured.err
```

### `README.md`

Added:

- local safety cap: `--max-tokens` rejects values above 64;
- no retry on the same credential, but router fallback can try another
  configured credential after failure;
- loaded keys remain in the Python process environment until the command exits.

## Acceptance criteria checked locally

- Standard suite remains offline.
- Smoke command remains manual/explicit.
- Smoke path still goes through `Router.call()`.
- `environ=` no longer suggests an unsupported end-to-end behavior.
- Sentinel fake secret is absent from smoke result and CLI stdout/stderr.
- Unsafe `max_tokens` values are rejected locally.
- README documents fallback and process-env lifetime.

## Local test evidence

Command:

```powershell
python -m pytest -q
```

Result:

```text
69 passed, 1 skipped, 1 warning
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
tests/test_smoke.py: 8
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

Please review whether the requested corrections are sufficient for running the
next single real-provider smoke test locally with one test key file.

Decision requested:

- APPROVATO
- RICHIEDE CORREZIONI
- BLOCCATO
