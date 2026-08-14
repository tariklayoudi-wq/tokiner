import os
import sys

import pytest

import free_cred.smoke as smoke
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
    env_names = ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_ENVS"]
    old_env = {name: os.environ.get(name) for name in env_names}

    captured = {}

    def factory(name, *, max_tokens=None, quota_source=None):
        captured["name"] = name
        captured["max_tokens"] = max_tokens
        captured["quota_source"] = quota_source
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    try:
        router = build_router_for_provider_smoke(
            "groq",
            keys_file=str(key_file),
            max_tokens=5,
            provider_factory=factory,
        )

        assert captured == {"name": "groq", "max_tokens": 5, "quota_source": None}
        assert os.environ["GROQ_API_KEY_ENVS"] == "GROQ_API_KEY_1,GROQ_API_KEY_2"
        assert router.select().id == "groq:groq_api_key_1"
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


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
        "ok": True,
    }


def test_run_provider_smoke_exact_expectation_is_opt_in():
    def factory(name, *, max_tokens=None, quota_source=None):
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    result = run_provider_smoke(
        "groq",
        prompt="small ping",
        max_tokens=3,
        expect_exact="different",
        provider_factory=factory,
    )

    assert result == {
        "provider": "groq:groq_api_key_1",
        "model": "groq:groq_api_key_1/model",
        "content": "ok",
        "ok": False,
        "expected": "different",
    }


def test_run_provider_smoke_summary_does_not_include_loaded_secret(tmp_path):
    sentinel = "SENTINEL_SECRET_VALUE"
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text(f"{sentinel}\n", encoding="utf-8")
    env_names = ["GROQ_API_KEY_1", "GROQ_API_KEY_ENVS"]
    old_env = {name: os.environ.get(name) for name in env_names}

    def factory(name, *, max_tokens=None, quota_source=None):
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    try:
        result = run_provider_smoke("groq", keys_file=str(key_file), provider_factory=factory)

        assert sentinel not in repr(result)
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_smoke_cli_stdout_does_not_include_loaded_secret(monkeypatch, tmp_path, capsys):
    sentinel = "SENTINEL_SECRET_VALUE"
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text(f"{sentinel}\n", encoding="utf-8")
    env_names = ["GROQ_API_KEY_1", "GROQ_API_KEY_ENVS"]
    old_env = {name: os.environ.get(name) for name in env_names}

    def factory(name, *, max_tokens=None, quota_source=None):
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    monkeypatch.setattr(smoke, "openai_compatible_providers_from_preset", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        ["free_cred.smoke", "groq", "--keys-file", str(key_file), "--max-tokens", "3"],
    )
    try:
        assert smoke.main() == 0
        captured = capsys.readouterr()

        assert sentinel not in captured.out
        assert sentinel not in captured.err
        assert "provider=groq:groq_api_key_1" in captured.out
        assert "ok=true" in captured.out
        assert "check=non_empty_content" in captured.out
    finally:
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_smoke_cli_returns_nonzero_for_failed_exact_expectation(monkeypatch, capsys):
    def factory(name, *, max_tokens=None, quota_source=None):
        return [FakeProvider("groq:groq_api_key_1", remaining=90)]

    monkeypatch.setattr(smoke, "openai_compatible_providers_from_preset", factory)
    monkeypatch.setattr(
        sys,
        "argv",
        ["free_cred.smoke", "groq", "--expect-exact", "different"],
    )

    assert smoke.main() == 1
    captured = capsys.readouterr()

    assert "ok=false" in captured.out
    assert "check=exact" in captured.out


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


def test_provider_smoke_router_still_uses_quota_rotation():
    low = FakeProvider("groq:groq_api_key_1", remaining=5)
    healthy = FakeProvider("groq:groq_api_key_2", remaining=80)

    def factory(name, *, max_tokens=None, quota_source=None):
        return [low, healthy]

    result = run_provider_smoke("groq", provider_factory=factory)

    assert result["provider"] == "groq:groq_api_key_2"
    assert low.calls == 0
    assert healthy.calls == 1


@pytest.mark.parametrize("max_tokens", [0, -1, 65])
def test_provider_smoke_rejects_unsafe_max_tokens(max_tokens):
    with pytest.raises(ValueError):
        build_router_for_provider_smoke(
            "groq",
            max_tokens=max_tokens,
            provider_factory=lambda name, **kwargs: [FakeProvider("groq", remaining=90)],
        )
