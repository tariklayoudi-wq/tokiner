import pytest

from free_cred.providers import (
    OPENAI_COMPATIBLE_PRESETS,
    InMemoryQuotaSource,
    OpenAICompatibleProvider,
    openai_compatible_provider_from_preset,
    openai_compatible_providers_from_preset,
    providers_from_env,
)
from free_cred.core import Router


class FakeMessage:
    content = "ok"


class FakeChoice:
    message = FakeMessage()


class FakeCompletion:
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeCompletion()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self):
        self.chat = FakeChat()


@pytest.mark.parametrize("name", ["groq", "cerebras", "together", "openrouter", "gemini"])
def test_openai_compatible_presets_build_generic_provider(name, monkeypatch):
    preset = OPENAI_COMPATIBLE_PRESETS[name]
    monkeypatch.setenv(preset.model_env, f"custom/{name}")

    provider = openai_compatible_provider_from_preset(name, client=FakeClient(), max_tokens=7)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.id == preset.id
    assert provider.base_url == preset.base_url
    assert provider.api_key_env == preset.api_key_env
    assert provider.model == f"custom/{name}"
    assert provider.max_tokens == 7


def test_openai_compatible_provider_calls_chat_completions():
    client = FakeClient()
    provider = OpenAICompatibleProvider(
        id="example",
        model="example/free",
        api_key_env="EXAMPLE_API_KEY",
        base_url="https://example.test/v1",
        client=client,
        max_tokens=9,
    )

    result = provider.call({"prompt": "Say ok"})

    assert result == {"provider": "example", "model": "example/free", "content": "ok"}
    sent = client.chat.completions.calls[0]
    assert sent["model"] == "example/free"
    assert sent["messages"] == [{"role": "user", "content": "Say ok"}]
    assert sent["temperature"] == 0.2
    assert sent["top_p"] == 0.95
    assert sent["max_tokens"] == 9
    assert sent["stream"] is False


def test_openai_compatible_provider_passes_agentic_messages_and_tools():
    client = FakeClient()
    provider = OpenAICompatibleProvider(
        id="example",
        model="example/free",
        api_key_env="EXAMPLE_API_KEY",
        client=client,
    )
    messages = [{"role": "user", "content": "Use a tool if needed"}]
    tools = [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]

    provider.call(
        {"messages": messages},
        tools=tools,
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    sent = client.chat.completions.calls[0]
    assert sent["messages"] == messages
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"
    assert sent["parallel_tool_calls"] is False


def test_openai_compatible_provider_requires_provider_specific_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = openai_compatible_provider_from_preset("groq")

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        provider.call({"prompt": "Say ok"})


def test_openai_compatible_provider_reads_normalized_quota_from_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_1_REMAINING", "5")
    monkeypatch.setenv("GROQ_API_KEY_1_LIMIT", "100")
    provider = OpenAICompatibleProvider(
        id="groq:GROQ_API_KEY_1",
        model="llama",
        api_key_env="GROQ_API_KEY_1",
        client=FakeClient(),
    )

    quota = provider.get_quota()

    assert quota.remaining == 5
    assert quota.limit == 100


def test_openai_compatible_provider_ignores_malformed_quota_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_1_REMAINING", "not-a-number")
    monkeypatch.setenv("GROQ_API_KEY_1_LIMIT", "-1")
    provider = OpenAICompatibleProvider(
        id="groq:GROQ_API_KEY_1",
        model="llama",
        api_key_env="GROQ_API_KEY_1",
        client=FakeClient(),
    )

    quota = provider.get_quota()

    assert quota.remaining is None
    assert quota.limit == 0


def test_openai_compatible_provider_can_use_runtime_quota_source():
    source = InMemoryQuotaSource()
    source.set_quota("GROQ_API_KEY_1", remaining=42, limit=100)
    provider = OpenAICompatibleProvider(
        id="groq:groq_api_key_1",
        model="llama",
        api_key_env="GROQ_API_KEY_1",
        client=FakeClient(),
        quota_source=source.get_quota,
    )

    quota = provider.get_quota()

    assert quota.remaining == 42
    assert quota.limit == 100


def test_provider_preset_expands_multiple_credential_envs(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_ENVS", "GROQ_API_KEY_1, GROQ_API_KEY_2")

    providers = openai_compatible_providers_from_preset("groq")

    assert [provider.id for provider in providers] == [
        "groq:groq_api_key_1",
        "groq:groq_api_key_2",
    ]
    assert [provider.api_key_env for provider in providers] == [
        "GROQ_API_KEY_1",
        "GROQ_API_KEY_2",
    ]


def test_router_rotates_between_credentials_before_free_tier_is_exhausted(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_ENVS", "GROQ_API_KEY_1,GROQ_API_KEY_2")
    monkeypatch.setenv("GROQ_API_KEY_1_REMAINING", "5")
    monkeypatch.setenv("GROQ_API_KEY_1_LIMIT", "100")
    monkeypatch.setenv("GROQ_API_KEY_2_REMAINING", "90")
    monkeypatch.setenv("GROQ_API_KEY_2_LIMIT", "100")
    providers = openai_compatible_providers_from_preset("groq")

    router = Router(providers, preventive_threshold=0.1)

    assert router.select().id == "groq:groq_api_key_2"


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


def test_router_uses_low_credentials_as_last_resort(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY_ENVS", "GROQ_API_KEY_1,GROQ_API_KEY_2")
    monkeypatch.setenv("GROQ_API_KEY_1_REMAINING", "5")
    monkeypatch.setenv("GROQ_API_KEY_1_LIMIT", "100")
    monkeypatch.setenv("GROQ_API_KEY_2_REMAINING", "4")
    monkeypatch.setenv("GROQ_API_KEY_2_LIMIT", "100")
    providers = openai_compatible_providers_from_preset("groq")

    router = Router(providers, preventive_threshold=0.1)

    assert router.select().id == "groq:groq_api_key_1"


def test_providers_from_env_supports_multiple_real_provider_presets(monkeypatch):
    monkeypatch.setenv("PROVIDERS", "groq, openrouter, local-mock")

    providers = providers_from_env()

    assert [provider.id for provider in providers] == ["groq", "openrouter", "local-mock"]


def test_providers_from_env_expands_credentials_for_real_provider(monkeypatch):
    monkeypatch.setenv("PROVIDERS", "groq")
    monkeypatch.setenv("GROQ_API_KEY_ENVS", "GROQ_API_KEY_1,GROQ_API_KEY_2")

    providers = providers_from_env()

    assert [provider.id for provider in providers] == [
        "groq:groq_api_key_1",
        "groq:groq_api_key_2",
    ]


def test_unknown_openai_compatible_preset_is_rejected():
    with pytest.raises(ValueError, match="unsupported OpenAI-compatible provider"):
        openai_compatible_provider_from_preset("unknown")
