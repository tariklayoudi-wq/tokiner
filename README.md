free-cred
==========

Minimal local core for a provider-agnostic, quota-aware router for Hermes Agent.

Goals implemented in this project:
- Provider interface with normalized QuotaState (remaining/limit/reset or None)
- Deterministic selection and preventive rotation based on configurable threshold
- Fallback and retry with injectable backoff
- Circuit breaker per-provider

Stack: Python 3.12+, Pydantic, FastAPI, HTTPX (for tests), pytest

The standard test suite uses fake providers or injected fake clients; no network
calls or real tokens are read. The optional NVIDIA smoke test is opt-in and
reads `NVIDIA_API_KEY` from the environment only.

Quick local demo
---------------

1. Create a virtual environment and install deps:

   python -m venv .venv
   .\.venv\Scripts\activate
   python -m pip install -e .

2. Run tests:

   python -m pytest -q

3. Run the demo API locally (no tokens, mock providers):

   set PROVIDERS=a,b
   python -c "from free_cred.api import create_app; import uvicorn; app=create_app(); uvicorn.run(app, host='127.0.0.1', port=8000)"

Endpoints:
- GET /health -> {"status":"ok"}
- POST /route with JSON {"prompt":"..."} -> returns provider_id and result or 503 error when no provider available

Notes:
- Configuration is non-secret (PROVIDERS, THRESHOLD env vars). No tokens, no external calls.

Optional NVIDIA/Nemotron smoke test
-----------------------------------

Set a test key in the environment, then run the integration test:

   set NVIDIA_API_KEY=...
   set RUN_NVIDIA_INTEGRATION=1
   set PROVIDERS=nvidia
   python -m pytest tests/test_nvidia_provider.py -q

The default model is `nvidia/nemotron-3-ultra-550b-a55b`. Override it with
`NVIDIA_MODEL` if needed. The smoke test uses a very small `max_tokens` value
to avoid consuming unnecessary quota.

OpenAI-compatible provider presets
----------------------------------

The router can also create optional OpenAI-compatible providers from
`PROVIDERS` without changing code:

   set PROVIDERS=nvidia,groq,cerebras,together,openrouter,gemini

Supported env vars:
- NVIDIA_API_KEY / NVIDIA_MODEL
- GROQ_API_KEY / GROQ_MODEL
- CEREBRAS_API_KEY / CEREBRAS_MODEL
- TOGETHER_API_KEY / TOGETHER_MODEL
- OPENROUTER_API_KEY / OPENROUTER_MODEL
- GEMINI_API_KEY / GEMINI_MODEL

Provider APIs and free-tier model availability change over time. Treat default
models as local presets, and override `*_MODEL` when a provider account exposes
a different free or trial model.

Credential rotation before free-tier exhaustion
-----------------------------------------------

Each provider preset can expand into multiple credentials. List the credential
environment variable names with `*_API_KEY_ENVS`, then publish non-secret quota
metadata beside each key:

   set PROVIDERS=groq
   set GROQ_API_KEY_ENVS=GROQ_API_KEY_1,GROQ_API_KEY_2
   set GROQ_API_KEY_1=...
   set GROQ_API_KEY_2=...
   set GROQ_API_KEY_1_REMAINING=5
   set GROQ_API_KEY_1_LIMIT=100
   set GROQ_API_KEY_2_REMAINING=90
   set GROQ_API_KEY_2_LIMIT=100

The router treats each credential as a separate provider candidate. With the
default preventive threshold, a credential at or below 10% remaining is skipped
while another credential has healthier quota. If every credential is below the
threshold, the router still uses them as a deterministic last resort.

For a long-running process, prefer an injected runtime quota source over env
vars. A quota probe running in the same process can update an `InMemoryQuotaSource`
as provider usage changes, and the router will use the updated values on the
next selection without restarting the app.

Loading local test keys from a file
-----------------------------------

For local testing, you can keep one API key per line in a file outside the repo
and load it into numbered environment variables without printing the key values:

   python -c "from free_cred.credentials import load_api_keys_file; print(load_api_keys_file('groq', r'C:\\path\\to\\groq_keys.txt'))"

For a two-line Groq key file, this populates:

   GROQ_API_KEY_1
   GROQ_API_KEY_2
   GROQ_API_KEY_ENVS=GROQ_API_KEY_1,GROQ_API_KEY_2

Use provider names matching the presets: `nvidia`, `groq`, `cerebras`,
`together`, `openrouter`, or `gemini`.

The loader preserves existing credential variables by default. Use
`overwrite=True` only when you intentionally reload a local test file; obsolete
numbered variables from the previous load are removed during that reload. Keep
key files outside the repository, do not commit them, do not print their
contents, and prefer a secret manager for anything beyond local testing.

Local real-provider smoke command
---------------------------------

After installing the package locally, run a tiny real-provider smoke test with
a key file that stays outside the repository:

   python -m free_cred.smoke nvidia --keys-file C:\Users\localad\nvm.txt --max-tokens 8

For other presets, replace `nvidia` with `groq`, `cerebras`, `together`,
`openrouter`, or `gemini`, and pass the matching key file. The command loads
the keys only into the current Python process, uses the same router selection
logic as the app, and prints provider/model/content only; it does not print key
values. Use a very small `--max-tokens` value for free-tier checks; the smoke
command rejects values above 64 as a local safety cap.

The command disables retry on the same credential, but it still uses the router
fallback path: if multiple credentials are configured and one candidate fails,
the smoke can try the next candidate. Loaded keys remain available in the
Python process environment until the command exits.
