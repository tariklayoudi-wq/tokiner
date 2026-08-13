free-cred
==========

Minimal local core for a provider-agnostic, quota-aware router for Hermes Agent.

Goals implemented in this project:
- Provider interface with normalized QuotaState (remaining/limit/reset or None)
- Deterministic selection and preventive rotation based on configurable threshold
- Fallback and retry with injectable backoff
- Circuit breaker per-provider

Stack: Python 3.12+, Pydantic, HTTPX (not used at runtime here), pytest

Tests use fake providers only; no network calls or real tokens are read.
