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
    expect_exact: Optional[str] = None,
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
    content = result.get("content")
    content_text = "" if content is None else str(content)
    summary = {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "content": content,
        "ok": bool(content_text.strip()),
    }
    if expect_exact is not None:
        summary["ok"] = content_text.strip() == expect_exact
        summary["expected"] = expect_exact
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a local opt-in smoke test against one OpenAI-compatible provider."
    )
    parser.add_argument("provider", help="Provider preset, for example nvidia, groq, openrouter, gemini.")
    parser.add_argument("--keys-file", help="Local file outside the repo with one API key per line.")
    parser.add_argument("--prompt", default="Reply with one word: ok")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--expect-exact",
        help="Optional strict assertion for demos that need an exact response after trimming whitespace.",
    )
    args = parser.parse_args()

    try:
        result = run_provider_smoke(
            args.provider,
            keys_file=args.keys_file,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            expect_exact=args.expect_exact,
        )
    except Exception as exc:
        parser.exit(status=2, message=f"smoke failed: {exc.__class__.__name__}\n")
    print(f"provider={result['provider']}")
    print(f"model={result['model']}")
    print(f"ok={str(result['ok']).lower()}")
    if "expected" in result:
        print("check=exact")
    else:
        print("check=non_empty_content")
    print(f"content={result['content']}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
