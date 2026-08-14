# Key file loader review artifact

This artifact contains the current hardened key-file loader and the central
tests. It intentionally omits real keys.

## Loader

```python
from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping, Optional

from .providers import OPENAI_COMPATIBLE_PRESETS


def load_api_keys_file(
    provider_name: str,
    path: str | Path,
    *,
    environ: Optional[MutableMapping[str, str]] = None,
    overwrite: bool = False,
    cleanup_obsolete: bool = True,
) -> list[str]:
    """Load one API key per line into provider-specific environment variables.

    Returns only the env var names that were populated. It never returns or logs
    key values. Existing credential env vars are preserved by default; pass
    overwrite=True for an intentional local reload.
    """

    env = environ if environ is not None else os.environ
    provider_key = provider_name.strip().lower()
    try:
        preset = OPENAI_COMPATIBLE_PRESETS[provider_key]
    except KeyError as exc:
        supported = ", ".join(sorted(OPENAI_COMPATIBLE_PRESETS))
        raise ValueError(f"unsupported OpenAI-compatible provider '{provider_name}'. Supported: {supported}") from exc

    try:
        raw_lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError("could not read API keys file") from exc

    keys = [
        line.strip()
        for line in raw_lines
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not keys:
        raise ValueError("no API keys found")
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate API keys are not allowed")

    base_env = preset.api_key_env
    env_names = [f"{base_env}_{index}" for index in range(1, len(keys) + 1)]
    collisions = [env_name for env_name in env_names if env_name in env and env[env_name] != keys[env_names.index(env_name)]]
    if collisions and not overwrite:
        raise RuntimeError(f"refusing to overwrite existing credential env vars: {', '.join(collisions)}")

    previous_envs = [item.strip() for item in env.get(f"{base_env}_ENVS", "").split(",") if item.strip()]
    if previous_envs and previous_envs != env_names and not overwrite:
        raise RuntimeError(f"refusing to replace existing credential env list: {base_env}_ENVS")

    for env_name, key in zip(env_names, keys):
        env[env_name] = key
    if cleanup_obsolete:
        for old_env_name in previous_envs:
            if old_env_name not in env_names:
                env.pop(old_env_name, None)
    env[f"{base_env}_ENVS"] = ",".join(env_names)
    return env_names
```

## Tests covered

```python
def test_load_api_keys_file_populates_numbered_env_without_returning_values(...)
def test_load_api_keys_file_keeps_inline_hash_as_secret_content(...)
def test_load_api_keys_file_rejects_empty_file(...)
def test_load_api_keys_file_rejects_missing_file_without_path_in_message(...)
def test_load_api_keys_file_rejects_unknown_provider(...)
def test_load_api_keys_file_refuses_existing_secret_overwrite_by_default(...)
def test_load_api_keys_file_can_overwrite_when_explicit(...)
def test_load_api_keys_file_cleans_obsolete_numbered_vars_on_shorter_reload(...)
def test_load_api_keys_file_rejects_duplicate_values_without_leaking_them(...)
def test_load_api_keys_file_integrates_with_providers_from_env(...)
```

## Local verification

`python -m pytest --tb=short`: 62 passed, 1 skipped, 1 warning.
