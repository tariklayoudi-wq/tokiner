import pytest
import os

from free_cred.credentials import load_api_keys_file
from free_cred.providers import providers_from_env


def test_load_api_keys_file_populates_numbered_env_without_returning_values(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("\n# comment\nkey-one\n\nkey-two\n", encoding="utf-8")
    env = {}

    env_names = load_api_keys_file("groq", key_file, environ=env)

    assert env_names == ["GROQ_API_KEY_1", "GROQ_API_KEY_2"]
    assert env["GROQ_API_KEY_ENVS"] == "GROQ_API_KEY_1,GROQ_API_KEY_2"
    assert env["GROQ_API_KEY_1"] == "key-one"
    assert env["GROQ_API_KEY_2"] == "key-two"
    assert "key-one" not in repr(env_names)
    assert "key-two" not in repr(env_names)


def test_load_api_keys_file_keeps_inline_hash_as_secret_content(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("key-one # not a comment\n", encoding="utf-8")
    env = {}

    env_names = load_api_keys_file("groq", key_file, environ=env)

    assert env_names == ["GROQ_API_KEY_1"]
    assert env["GROQ_API_KEY_1"] == "key-one # not a comment"


def test_load_api_keys_file_rejects_empty_file(tmp_path):
    key_file = tmp_path / "empty.txt"
    key_file.write_text("\n# comment\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no API keys found"):
        load_api_keys_file("groq", key_file, environ={})


def test_load_api_keys_file_rejects_missing_file_without_path_in_message(tmp_path):
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(ValueError, match="could not read API keys file") as exc:
        load_api_keys_file("groq", missing_file, environ={})

    assert str(missing_file) not in str(exc.value)


def test_load_api_keys_file_rejects_unknown_provider(tmp_path):
    key_file = tmp_path / "keys.txt"
    key_file.write_text("secret\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported OpenAI-compatible provider"):
        load_api_keys_file("unknown", key_file, environ={})


def test_load_api_keys_file_refuses_existing_secret_overwrite_by_default(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("new-key\n", encoding="utf-8")
    env = {"GROQ_API_KEY_1": "existing-key"}

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        load_api_keys_file("groq", key_file, environ=env)

    assert env["GROQ_API_KEY_1"] == "existing-key"


def test_load_api_keys_file_can_overwrite_when_explicit(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("new-key\n", encoding="utf-8")
    env = {"GROQ_API_KEY_1": "existing-key"}

    env_names = load_api_keys_file("groq", key_file, environ=env, overwrite=True)

    assert env_names == ["GROQ_API_KEY_1"]
    assert env["GROQ_API_KEY_1"] == "new-key"


def test_load_api_keys_file_cleans_obsolete_numbered_vars_on_shorter_reload(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("new-key\n", encoding="utf-8")
    env = {
        "GROQ_API_KEY_1": "old-key-1",
        "GROQ_API_KEY_2": "old-key-2",
        "GROQ_API_KEY_3": "old-key-3",
        "GROQ_API_KEY_ENVS": "GROQ_API_KEY_1,GROQ_API_KEY_2,GROQ_API_KEY_3",
    }

    env_names = load_api_keys_file("groq", key_file, environ=env, overwrite=True)

    assert env_names == ["GROQ_API_KEY_1"]
    assert env["GROQ_API_KEY_1"] == "new-key"
    assert "GROQ_API_KEY_2" not in env
    assert "GROQ_API_KEY_3" not in env
    assert env["GROQ_API_KEY_ENVS"] == "GROQ_API_KEY_1"


def test_load_api_keys_file_rejects_duplicate_values_without_leaking_them(tmp_path):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("duplicate-key\nduplicate-key\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate API keys") as exc:
        load_api_keys_file("groq", key_file, environ={})

    assert "duplicate-key" not in str(exc.value)


def test_load_api_keys_file_integrates_with_providers_from_env(tmp_path, monkeypatch):
    key_file = tmp_path / "groq_keys.txt"
    key_file.write_text("key-one\nkey-two\n", encoding="utf-8")

    monkeypatch.setenv("PROVIDERS", "groq")
    try:
        env_names = load_api_keys_file("groq", key_file, overwrite=True)
        providers = providers_from_env()
    finally:
        for name in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY_ENVS"]:
            os.environ.pop(name, None)

    assert env_names == ["GROQ_API_KEY_1", "GROQ_API_KEY_2"]
    assert [provider.id for provider in providers] == [
        "groq:groq_api_key_1",
        "groq:groq_api_key_2",
    ]
