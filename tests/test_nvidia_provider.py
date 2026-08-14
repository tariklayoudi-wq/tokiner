import os

import pytest

from free_cred.core import Router
from free_cred.providers import NvidiaNemotronProvider, providers_from_env


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


def test_nvidia_provider_uses_openai_compatible_chat_client():
    client = FakeClient()
    provider = NvidiaNemotronProvider(client=client, max_tokens=8)

    result = provider.call({"prompt": "Say ok"})

    assert result == {
        "provider": "nvidia-nemotron",
        "model": "nvidia/nemotron-3-ultra-550b-a55b",
        "content": "ok",
    }
    sent = client.chat.completions.calls[0]
    assert sent["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert sent["messages"] == [{"role": "user", "content": "Say ok"}]
    assert sent["max_tokens"] == 8
    assert sent["stream"] is False


def test_nvidia_provider_requires_key_without_injected_client(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    provider = NvidiaNemotronProvider()

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        provider.call({"prompt": "Say ok"})


def test_providers_from_env_can_create_nvidia_provider(monkeypatch):
    monkeypatch.setenv("PROVIDERS", "nvidia")
    monkeypatch.setenv("NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b")

    providers = providers_from_env()

    assert len(providers) == 1
    assert providers[0].id == "nvidia-nemotron"


@pytest.mark.integration
def test_router_smoke_with_real_nvidia_nemotron():
    if os.environ.get("RUN_NVIDIA_INTEGRATION") != "1":
        pytest.skip("set RUN_NVIDIA_INTEGRATION=1 to call NVIDIA")
    if not os.environ.get("NVIDIA_API_KEY"):
        pytest.skip("NVIDIA_API_KEY not set")
    provider = NvidiaNemotronProvider(max_tokens=8)
    router = Router([provider], backoff=lambda n: 0, max_retries=0)

    result = router.call({"prompt": "Reply with one word: ok"})

    assert result["provider"] == "nvidia-nemotron"
    assert result["model"] == "nvidia/nemotron-3-ultra-550b-a55b"
    assert isinstance(result["content"], str)
    assert result["content"].strip()
