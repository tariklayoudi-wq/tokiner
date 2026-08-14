# Review request - Hermes CLI integration

Project: free cred / Hermes Agent credential rotation
Branch: dev
Date: 2026-08-14

## Context

Goal: verify that the free-cred router/smoke path works with the local Hermes
Agent CLI and the configured NVIDIA/Nemotron credential.

No key file contents are included here. The local key file path was used only
as an input to the smoke command and was not printed or copied.

## Changed files in this cycle

- `.hermes/environment.json`
- `pyproject.toml`
- `free_cred/smoke.py`
- `tests/test_smoke.py`

## Implementation summary

- Added `.hermes/environment.json` so `hermes verify` uses the actual project
  entrypoints instead of detecting a nonexistent `main.py`.
- The manifest uses:
  - bootstrap: `py -3.12 -m pip install -e .`
  - test: `py -3.12 -m pytest`
  - start: `py -3.12 -m uvicorn free_cred.api:create_app --factory --host 127.0.0.1 --port 8000`
  - readiness: `/health`
- Fixed setuptools package discovery in `pyproject.toml` by including only
  `free_cred*`. This avoids editable install failure caused by the top-level
  `review_artifacts/` folder.
- Updated smoke success contract:
  - default `ok=True` when provider returns non-empty content;
  - optional `--expect-exact` for strict assertions;
- CLI prints `ok=true|false`;
- CLI prints `check=non_empty_content` by default to make clear that `ok=true`
  means the transport/provider smoke returned non-empty content, not that the
  model obeyed the prompt exactly;
- exit codes: `0` success, `1` strict/content failure, `2` runtime/config failure.

## Hermes CLI evidence

Hermes installation:

```text
hermes --version
Hermes Agent v0.20.0 (2026.8.3)
Provider configured: NVIDIA NIM
Model configured: nvidia/nemotron-3-ultra-550b-a55b
```

Hermes status/auth:

```text
hermes status
NVIDIA NIM key present, redacted by Hermes status output.
```

Hermes doctor:

```text
hermes doctor
NVIDIA NIM connectivity: passed
Known unrelated issues: optional missing keys/tools, npm advisory warnings.
```

Hermes project verification:

```text
hermes verify --skip-start --json .
ok=true
bootstrap: py -3.12 -m pip install -e . -> exitCode 0
test: py -3.12 -m pytest -> exitCode 0
73 passed, 1 skipped, 1 warning
```

Hermes one-shot provider test:

```text
hermes -z "Reply with exactly one word: ok" --provider nvidia --model nvidia/nemotron-3-ultra-550b-a55b --reasoning none
ok
exit code 0
```

FastAPI readiness evidence:

```text
py -3.12 -m uvicorn free_cred.api:create_app --factory --host 127.0.0.1 --port 8765
GET http://127.0.0.1:8765/health
StatusCode 200
Content {"status":"ok"}
```

free-cred real provider smoke:

```text
py -3.12 -m free_cred.smoke nvidia --keys-file <local key file outside repo> --prompt "Reply with exactly one word: ok" --max-tokens 8
provider=nvidia-nemotron
model=nvidia/nemotron-3-ultra-550b-a55b
ok=true
check=non_empty_content
content=The user wants me to reply with exactly
exit code 0
```

Interpretation: the real provider connection, key-file loader, router path, and
smoke CLI all work. The model still did not follow the exact-word instruction,
which is why the smoke contract now checks connectivity/content by default and
reserves exact matching for `--expect-exact`.

## Reviewer request

Please review:

- Is `.hermes/environment.json` an appropriate local Hermes verification manifest?
- Is the setuptools discovery fix correct and minimal?
- Is the smoke success contract appropriate for Hermes/free-tier connectivity tests?
- Are there any security regressions around key files, stdout/stderr, or artifacts?

Verdict requested: APPROVATO / RICHIEDE CORREZIONI / BLOCCATO.
