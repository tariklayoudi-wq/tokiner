# Credential rotation runtime review artifact

This artifact contains the current runtime quota-source mechanism added for
credential rotation before free-tier exhaustion. It is intentionally small so
Edge Copilot can inspect it without stale attachment caching.

## Runtime quota source

```python
class InMemoryQuotaSource:
    """Runtime-updatable quota source keyed by credential env var name."""

    def __init__(self, quotas: Optional[Dict[str, QuotaState]] = None):
        self._quotas = dict(quotas or {})

    def set_quota(self, api_key_env: str, remaining: Optional[int], limit: Optional[int]) -> None:
        self._quotas[api_key_env] = QuotaState(remaining=remaining, limit=limit)

    def get_quota(self, api_key_env: str) -> QuotaState:
        return self._quotas.get(api_key_env, QuotaState())
```

## Provider injection point

```python
class OpenAICompatibleProvider:
    def __init__(
        self,
        id: str,
        model: str,
        api_key_env: str,
        base_url: str = "https://api.openai.com/v1",
        client: Optional[Any] = None,
        quota_source: Optional[Callable[[str], QuotaState]] = None,
        max_tokens: int = 64,
    ):
        self.id = id
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self._api_key = os.environ.get(api_key_env)
        self._client = client
        self._quota_source = quota_source

    def get_quota(self) -> QuotaState:
        if self._quota_source is not None:
            return self._quota_source(self.api_key_env)
        return QuotaState(
            remaining=_optional_int_env(f"{self.api_key_env}_REMAINING"),
            limit=_optional_int_env(f"{self.api_key_env}_LIMIT"),
        )
```

## Preset propagation

```python
def openai_compatible_provider_from_preset(
    name: str,
    *,
    client: Optional[Any] = None,
    max_tokens: Optional[int] = None,
    api_key_env: Optional[str] = None,
    id: Optional[str] = None,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
) -> OpenAICompatibleProvider:
    ...
    return OpenAICompatibleProvider(
        id=id or preset.id,
        model=os.environ.get(preset.model_env, preset.default_model),
        api_key_env=api_key_env or preset.api_key_env,
        base_url=preset.base_url,
        client=client,
        quota_source=quota_source,
        max_tokens=provider_max_tokens,
    )


def openai_compatible_providers_from_preset(
    name: str,
    *,
    max_tokens: Optional[int] = None,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
) -> List[OpenAICompatibleProvider]:
    ...
    return [
        openai_compatible_provider_from_preset(
            name,
            max_tokens=max_tokens,
            api_key_env=api_key_env,
            id=_credential_provider_id(preset.id, api_key_env, len(credential_envs)),
            quota_source=quota_source,
        )
        for api_key_env in credential_envs
    ]
```

## Runtime rotation test

```python
def test_router_rotation_uses_runtime_quota_updates_after_start(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_ENVS", "GROQ_API_KEY_1,GROQ_API_KEY_2")
    source = InMemoryQuotaSource()
    source.set_quota("GROQ_API_KEY_1", remaining=90, limit=100)
    source.set_quota("GROQ_API_KEY_2", remaining=80, limit=100)
    providers = openai_compatible_providers_from_preset("groq", quota_source=source.get_quota)
    router = Router(providers, preventive_threshold=0.1)
    assert router.select().id == "groq:groq_api_key_1"

    source.set_quota("GROQ_API_KEY_1", remaining=5, limit=100)

    assert router.select().id == "groq:groq_api_key_2"
```
