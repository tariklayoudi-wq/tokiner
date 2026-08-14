from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import os
from .core import Provider, QuotaState


class MockProvider:
    def __init__(self, id: str, remaining: Optional[int] = None, limit: Optional[int] = None, fail: bool = False):
        self.id = id
        self._quota = QuotaState(remaining=remaining, limit=limit, reset=None)
        self.fail = fail
        self.calls = 0

    def get_quota(self) -> QuotaState:
        return self._quota

    def call(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError(f"provider {self.id} simulated failure")
        return {"provider": self.id, "payload": args or kwargs}


@dataclass(frozen=True)
class OpenAICompatiblePreset:
    id: str
    base_url: str
    api_key_env: str
    model_env: str
    default_model: str


OPENAI_COMPATIBLE_PRESETS: Dict[str, OpenAICompatiblePreset] = {
    "nvidia": OpenAICompatiblePreset(
        id="nvidia-nemotron",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_env="NVIDIA_MODEL",
        default_model="nvidia/nemotron-3-ultra-550b-a55b",
    ),
    "nemotron": OpenAICompatiblePreset(
        id="nvidia-nemotron",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_env="NVIDIA_MODEL",
        default_model="nvidia/nemotron-3-ultra-550b-a55b",
    ),
    "nvidia-nemotron": OpenAICompatiblePreset(
        id="nvidia-nemotron",
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_env="NVIDIA_MODEL",
        default_model="nvidia/nemotron-3-ultra-550b-a55b",
    ),
    "groq": OpenAICompatiblePreset(
        id="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model_env="GROQ_MODEL",
        default_model="llama-3.1-8b-instant",
    ),
    "cerebras": OpenAICompatiblePreset(
        id="cerebras",
        base_url="https://api.cerebras.ai/v1",
        api_key_env="CEREBRAS_API_KEY",
        model_env="CEREBRAS_MODEL",
        default_model="llama3.1-8b",
    ),
    "together": OpenAICompatiblePreset(
        id="together",
        base_url="https://api.together.ai/v1",
        api_key_env="TOGETHER_API_KEY",
        model_env="TOGETHER_MODEL",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    ),
    "openrouter": OpenAICompatiblePreset(
        id="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key_env="OPENROUTER_API_KEY",
        model_env="OPENROUTER_MODEL",
        default_model="openrouter/auto",
    ),
    "gemini": OpenAICompatiblePreset(
        id="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        default_model="gemini-2.5-flash",
    ),
}


class InMemoryQuotaSource:
    """Runtime-updatable quota source keyed by credential env var name."""

    def __init__(self, quotas: Optional[Dict[str, QuotaState]] = None):
        self._quotas = dict(quotas or {})

    def set_quota(self, api_key_env: str, remaining: Optional[int], limit: Optional[int]) -> None:
        self._quotas[api_key_env] = QuotaState(remaining=remaining, limit=limit)

    def get_quota(self, api_key_env: str) -> QuotaState:
        return self._quotas.get(api_key_env, QuotaState())


class OpenAICompatibleProvider:
    """Generic chat-completions provider for OpenAI-compatible free/free-tier APIs.

    The API key is read only from the environment by default. Tests can inject a
    fake client so unit coverage never needs a network call or a secret.
    """

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
        # Most OpenAI-compatible APIs do not expose normalized remaining/limit/reset
        # values here. A local quota probe can publish non-secret normalized values
        # beside each credential env var, for example GROQ_API_KEY_1_REMAINING.
        return QuotaState(
            remaining=_optional_int_env(f"{self.api_key_env}_REMAINING"),
            limit=_optional_int_env(f"{self.api_key_env}_LIMIT"),
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise RuntimeError(f"{self.api_key_env} is required for {self.__class__.__name__}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(f"openai package is required for {self.__class__.__name__}") from exc
        self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def call(self, payload=None, **kwargs):
        prompt = ""
        if isinstance(payload, dict):
            prompt = str(payload.get("prompt") or "")
        elif payload is not None:
            prompt = str(payload)
        if not prompt:
            prompt = "Respond with exactly: ok"

        completion = self._get_client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.2),
            top_p=kwargs.get("top_p", 0.95),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            extra_body=kwargs.get("extra_body"),
            stream=False,
        )
        message = completion.choices[0].message
        return {
            "provider": self.id,
            "model": self.model,
            "content": message.content,
        }


class NvidiaNemotronProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        id: str = "nvidia-nemotron",
        model: str = "nvidia/nemotron-3-ultra-550b-a55b",
        api_key_env: str = "NVIDIA_API_KEY",
        base_url: str = "https://integrate.api.nvidia.com/v1",
        client: Optional[Any] = None,
        quota_source: Optional[Callable[[str], QuotaState]] = None,
        max_tokens: int = 64,
    ):
        super().__init__(
            id=id,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            client=client,
            quota_source=quota_source,
            max_tokens=max_tokens,
        )


def _optional_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return max(parsed, 0)


def _credential_envs_for_preset(preset: OpenAICompatiblePreset) -> List[str]:
    raw = os.environ.get(f"{preset.api_key_env}_ENVS")
    if not raw:
        return [preset.api_key_env]
    envs = [item.strip() for item in raw.split(",") if item.strip()]
    return envs or [preset.api_key_env]


def _credential_provider_id(base_id: str, api_key_env: str, total: int) -> str:
    if total == 1:
        return base_id
    return f"{base_id}:{api_key_env.lower()}"


def openai_compatible_provider_from_preset(
    name: str,
    *,
    client: Optional[Any] = None,
    max_tokens: Optional[int] = None,
    api_key_env: Optional[str] = None,
    id: Optional[str] = None,
    quota_source: Optional[Callable[[str], QuotaState]] = None,
) -> OpenAICompatibleProvider:
    key = name.strip().lower()
    try:
        preset = OPENAI_COMPATIBLE_PRESETS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(OPENAI_COMPATIBLE_PRESETS))
        raise ValueError(f"unsupported OpenAI-compatible provider '{name}'. Supported: {supported}") from exc

    provider_max_tokens = max_tokens
    if provider_max_tokens is None:
        env_name = f"{preset.id.upper().replace('-', '_')}_MAX_TOKENS"
        provider_max_tokens = int(os.environ.get(env_name, "64"))

    if preset.id == "nvidia-nemotron":
        return NvidiaNemotronProvider(
            id=id or preset.id,
            model=os.environ.get(preset.model_env, preset.default_model),
            api_key_env=api_key_env or preset.api_key_env,
            client=client,
            quota_source=quota_source,
            max_tokens=provider_max_tokens,
        )

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
    key = name.strip().lower()
    try:
        preset = OPENAI_COMPATIBLE_PRESETS[key]
    except KeyError as exc:
        supported = ", ".join(sorted(OPENAI_COMPATIBLE_PRESETS))
        raise ValueError(f"unsupported OpenAI-compatible provider '{name}'. Supported: {supported}") from exc

    credential_envs = _credential_envs_for_preset(preset)
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


def providers_from_env() -> List[Provider]:
    """Create mock providers from PROVIDERS env var (comma-separated ids).
    Example: PROVIDERS=a,b or PROVIDERS=nvidia,groq,openrouter
    """
    raw = os.environ.get("PROVIDERS", "a,b")
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    out: List[Provider] = []
    for i, pid in enumerate(ids):
        if pid.lower() in OPENAI_COMPATIBLE_PRESETS:
            out.extend(openai_compatible_providers_from_preset(pid))
            continue
        # stagger quotas for demo
        remaining = None if i % 2 == 0 else 80
        limit = None if remaining is None else 100
        out.append(MockProvider(pid, remaining=remaining, limit=limit))
    return out
