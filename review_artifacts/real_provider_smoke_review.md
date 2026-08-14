# Review artifact: local real-provider smoke harness

## Scope

This cycle adds a local, explicit smoke-test harness for real OpenAI-compatible
providers. It is meant for manually validating keys prepared in local files,
one key per line, without changing the standard offline test suite.

No push, merge, deploy, provider signup, credential creation, or persistent
credential change is included.

## Files changed

- `free_cred/smoke.py`
- `tests/test_smoke.py`
- `README.md`

## Implementation

### `free_cred/smoke.py`

```python
from __future__ import annotations

import argparse
from typing import Any, Callable, MutableMapping, Optional

from .core import Provider, QuotaState, Router
from .credentials import load_api_keys_file
from .providers import openai_compatible_providers_from_preset


ProviderFactory = Callable[..., list[Provider]]


def build_router_for_provider_smoke(
    provider_name: str,
    *,
    keys_file: Optional[str] = None,
    environ: Optional[MutableMapping[str, str]] = None,
    max_tokens: int = 8,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
    provider_factory: ProviderFactory = openai_compatible_providers_from_preset,
) -> Router:
    """Build a no-retry router for a local, opt-in real-provider smoke test."""

    if keys_file:
        load_api_keys_file(provider_name, keys_file, environ=environ, overwrite=True)
    providers = provider_factory(
        provider_name,
        max_tokens=max_tokens,
        quota_source=quota_source,
    )
    return Router(providers, backoff=lambda attempt: 0, max_retries=0)


def run_provider_smoke(
    provider_name: str,
    *,
    keys_file: Optional[str] = None,
    environ: Optional[MutableMapping[str, str]] = None,
    prompt: str = "Reply with one word: ok",
    max_tokens: int = 8,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
    provider_factory: ProviderFactory = openai_compatible_providers_from_preset,
) -> dict[str, Any]:
    """Call one provider through the router and return a redacted smoke summary."""

    router = build_router_for_provider_smoke(
        provider_name,
        keys_file=keys_file,
        environ=environ,
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

    result = run_provider_smoke(
        args.provider,
        keys_file=args.keys_file,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    print(f"provider={result['provider']}")
    print(f"model={result['model']}")
    print(f"content={result['content']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

### `tests/test_smoke.py`

```python
from free_cred.core import QuotaState
from free_cred.smoke import build_router_for_provider_smoke, run_provider_smoke


class FakeProvider:
    def __init__(self, id: str, remaining: int):
        self.id = id
        self.model = f"{id}/model"
        self.remaining = remaining
        self.calls = 0

    def get_quota(self):
        return QuotaState(remaining=self.remaining, limit=100)

    def call(self, payload=None, **kwargs):
        self.calls += 1
        return {
            "provider": self.id,
            "model": self.model,
            "content": "ok",
            "payload": payload,
            "kwargs": kwargs,
        }


def test_build_router_for_provider_smoke_loads_key_file(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("test-key-1\ntest-key-2\n", encoding="utf-8")
    env = {}

    captured = {}

    def factory(name, *, max_tokens=None, quota_source=None):
        captured["name"] = name
        captured["max_tokens"] = max_tokens
        captured["quota_source"] = quota_source
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    router = build_router_for_provider_smoke(
        "groq",
        keys_file=str(key_file),
        environ=env,
        max_tokens=5,
        provider_factory=factory,
    )

    assert captured == {"name": "groq", "max_tokens": 5, "quota_source": None}
    assert env["GROQ_API_KEY_ENVS"] == "GROQ_API_KEY_1,GROQ_API_KEY_2"
    assert set(env) == {"GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_ENVS"}
    assert router.select().id == "groq:groq_api_key_1"


def test_run_provider_smoke_returns_redacted_summary():
    def factory(name, *, max_tokens=None, quota_source=None):
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    result = run_provider_smoke(
        "groq",
        prompt="small ping",
        max_tokens=3,
        provider_factory=factory,
    )

    assert result == {
        "provider": "groq:groq_api_key_1",
        "model": "groq:groq_api_key_1/model",
        "content": "ok",
    }


def test_provider_smoke_router_still_uses_quota_rotation():
    low = FakeProvider("groq:groq_api_key_1", remaining=5)
    healthy = FakeProvider("groq:groq_api_key_2", remaining=80)

    def factory(name, *, max_tokens=None, quota_source=None):
        return [low, healthy]

    result = run_provider_smoke("groq", provider_factory=factory)

    assert result["provider"] == "groq:groq_api_key_2"
    assert low.calls == 0
    assert healthy.calls == 1
```

## README addition

The README now documents:

- `python -m free_cred.smoke nvidia --keys-file C:\Users\localad\nvm.txt --max-tokens 8`
- supported presets: `nvidia`, `groq`, `cerebras`, `together`, `openrouter`, `gemini`;
- keys stay outside the repository;
- keys are loaded only into the current Python process;
- output includes provider/model/content only, not key values;
- small `--max-tokens` values are recommended for free-tier checks.

## Acceptance criteria checked locally

- The standard suite still avoids network and real token usage.
- Real provider smoke is explicit/manual via `python -m free_cred.smoke ...`.
- Key files are loaded through the hardened `load_api_keys_file`.
- The router remains the single path for real-provider selection.
- The smoke path still honors quota-aware rotation.
- The smoke summary does not return or print secret values.

## Local test evidence

Command:

```powershell
python -m pytest -q
```

Result:

```text
65 passed, 1 skipped, 1 warning
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
tests/test_smoke.py: 3
```

Command:

```powershell
git diff --check
```

Result:

```text
no whitespace errors; only existing Windows LF/CRLF warnings for README.md,
free_cred/providers.py, and pyproject.toml
```

## Requested Edge review decision

Please review this cycle and decide one of:

- APPROVATO
- RICHIEDE CORREZIONI
- BLOCCATO

Focus review on:

- whether the smoke command can accidentally leak or persist secrets;
- whether the explicit opt-in boundary is clear enough;
- whether the smoke path correctly exercises the router instead of bypassing it;
- whether the fake tests cover the important behavior without calling real providers;
- any risk before running more real free-tier provider checks.
