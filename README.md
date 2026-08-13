free-cred
==========

Minimal local core for a provider-agnostic, quota-aware router for Hermes Agent.

Goals implemented in this project:
- Provider interface with normalized QuotaState (remaining/limit/reset or None)
- Deterministic selection and preventive rotation based on configurable threshold
- Fallback and retry with injectable backoff
- Circuit breaker per-provider

Stack: Python 3.12+, Pydantic, FastAPI, HTTPX (for tests), pytest

Tests use fake providers only; no network calls or real tokens are read.

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
