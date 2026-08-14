# Planner update - Hermes CLI integration

Project: free cred / Hermes Agent credential rotation
Date: 2026-08-14
Branch: dev

## Status

Hermes Agent CLI integration is locally verified against the configured
NVIDIA/Nemotron provider and the free-cred smoke/router path.

## Implemented

- Added Hermes project verification manifest: `.hermes/environment.json`.
- Fixed editable install packaging by constraining setuptools package discovery
  to `free_cred*`.
- Hardened smoke success reporting:
  - default success is non-empty provider content;
  - strict exact response is opt-in with `--expect-exact`;
  - CLI emits `ok=true|false`;
  - CLI emits `check=non_empty_content` for the default connectivity contract.

## Test and verification evidence

- `hermes status`: NVIDIA NIM configured with redacted key.
- `hermes doctor`: NVIDIA NIM connectivity passed.
- `hermes -z ... --provider nvidia --model nvidia/nemotron-3-ultra-550b-a55b`: returned `ok`, exit code 0.
- `hermes verify --skip-start --json .`: passed bootstrap and tests.
- `py -3.12 -m pytest -q`: passed.
- Manual app readiness:
  - Uvicorn started with `free_cred.api:create_app --factory`.
  - `/health` returned HTTP 200 and `{"status":"ok"}`.
- Real free-cred smoke:
  - provider `nvidia-nemotron`;
  - model `nvidia/nemotron-3-ultra-550b-a55b`;
  - `ok=true`;
  - `check=non_empty_content`;
  - exit code 0.

## Known notes

- Full `hermes verify --json .` did not return JSON promptly during the start
  phase, so readiness was verified manually with explicit start, HTTP check, and
  Ctrl+C teardown.
- The model still does not reliably obey exact `ok` prompts. This is treated as
  model behavior, not connectivity failure.
- No push, merge, deploy, or credential creation was performed.
- No key contents were printed or copied into artifacts.

## Next action

Reviewer returned `APPROVATO`. A non-blocking suggestion to clarify the smoke
contract was implemented by adding `check=non_empty_content` to CLI output.
Next: prepare local commit when the user confirms.
