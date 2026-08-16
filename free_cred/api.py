import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from pydantic import BaseModel
from .schemas import RouteRequest, RouteResponse, ErrorResponse
from .credentials import load_api_keys_file
from .providers import providers_from_env
from .core import Router, CircuitBreaker, Provider, QuotaState
from .quota_monitor import QuotaMonitor, get_quota_monitor, create_quota_source, init_quota_monitor
from typing import Any, Optional, Callable, Dict, List
import json
import time
import uuid
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime


OPENAI_COMPATIBLE_MODEL = "free-cred-router"


class ChatMessage(BaseModel):
    role: str
    content: Any = ""


class ChatCompletionRequest(BaseModel):
    model: str = OPENAI_COMPATIBLE_MODEL
    messages: list[ChatMessage]
    tools: Optional[list[dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    extra_body: Optional[dict[str, Any]] = None


class AddCredentialRequest(BaseModel):
    api_key_env: str
    provider_name: str
    api_key: str
    model: Optional[str] = None


_quota_monitor_instance: Optional[QuotaMonitor] = None


async def _init_quota_monitor_on_startup():
    """Initialize quota monitor on app startup."""
    global _quota_monitor_instance
    _quota_monitor_instance = await init_quota_monitor(
        active_interval=int(os.environ.get("QUOTA_ACTIVE_INTERVAL", "30")),
        standby_interval=int(os.environ.get("QUOTA_STANDBY_INTERVAL", "3600")),
    )


async def _shutdown_quota_monitor():
    """Shutdown quota monitor on app shutdown."""
    global _quota_monitor_instance
    if _quota_monitor_instance:
        await _quota_monitor_instance.stop()
        _quota_monitor_instance = None


def _make_quota_source():
    """Create quota source function that reads from the running quota monitor."""
    def quota_source(api_key_env: str) -> QuotaState:
        global _quota_monitor_instance
        if _quota_monitor_instance:
            info = _quota_monitor_instance.get_quota(api_key_env)
            if info:
                return QuotaState(
                    remaining=info.remaining,
                    limit=info.limit,
                    reset=info.reset,
                )
        return QuotaState()
    return quota_source


def create_app(providers=None, preventive_threshold: float = 0.1, max_retries: int = 2, cb_factory: Optional[Callable[[], CircuitBreaker]] = None) -> FastAPI:
    """Factory to create FastAPI app without side effects.
    Reads PROVIDERS and THRESHOLD env variables if providers is None.
    """
    if providers is None:
        _load_provider_keys_file_from_env()
        providers = providers_from_env(quota_source=_make_quota_source())
    # threshold may be provided via env (non-secret) for demo
    try:
        env_thr = float(os.environ.get("THRESHOLD", str(preventive_threshold)))
    except (TypeError, ValueError):
        env_thr = preventive_threshold
    # default cb_factory to CircuitBreaker constructor if not provided
    if cb_factory is None:
        cb_factory = lambda: CircuitBreaker()

    # Router expects cb_factory returning CircuitBreaker instances; pass it through directly
    r = Router(providers, preventive_threshold=env_thr, max_retries=max_retries, cb_factory=cb_factory)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await _init_quota_monitor_on_startup()
        yield
        await _shutdown_quota_monitor()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/health/nvidia")
    def health_nvidia():
        return _health_for_provider_group(providers, "nvidia")

    @app.get("/health/openvino")
    def health_openvino():
        return _health_for_provider_group(providers, "openvino")

    @app.get("/health/copilot")
    def health_copilot():
        return _health_for_provider_group(providers, "copilot")

    @app.get("/providers/status")
    def providers_status():
        checks = [_health_for_provider(provider) for provider in providers]
        return {
            "status": "healthy" if any(check["ok"] for check in checks) else "unhealthy",
            "providers": checks,
        }

    @app.get("/v1/models")
    def openai_models():
        return _models_response()

    @app.get("/api/v1/models")
    def openai_models_alias():
        return _models_response()

    @app.get("/v1/models/{model_id}")
    def openai_model(model_id: str):
        if model_id != OPENAI_COMPATIBLE_MODEL:
            return JSONResponse(status_code=404, content={"error": {"message": "model not found"}})
        return _model_card()

    @app.get("/version")
    def version():
        return {"version": "free-cred-router-0.0.0"}

    @app.get("/props")
    def props():
        return {"models": [OPENAI_COMPATIBLE_MODEL], "openai_compatible": True}

    @app.get("/v1/props")
    def v1_props():
        return props()

    @app.get("/api/tags")
    def ollama_tags():
        return {"models": [{"name": OPENAI_COMPATIBLE_MODEL, "model": OPENAI_COMPATIBLE_MODEL}]}

    @app.post("/api/show")
    def ollama_show():
        return {"model": OPENAI_COMPATIBLE_MODEL, "details": {"family": "free-cred"}}

    def _models_response():
        return {
            "object": "list",
            "data": [_model_card()],
        }

    def _model_card():
        return {
            "id": OPENAI_COMPATIBLE_MODEL,
            "object": "model",
            "created": 0,
            "owned_by": "free-cred",
        }

    @app.post("/route", response_model=RouteResponse, responses={503: {"model": ErrorResponse}})
    def route(req: RouteRequest):
        try:
            res = r.call(req.model_dump())
            if isinstance(res, dict) and "provider" in res:
                provider_id = res.get("provider")
            else:
                provider_id = getattr(res, "provider", "unknown")
            return {"provider_id": provider_id, "result": res}
        except RuntimeError as e:
            err = {"message": str(e)}
            return JSONResponse(status_code=503, content={"error": err})

    @app.post("/v1/chat/completions")
    def openai_chat_completions(req: ChatCompletionRequest):
        try:
            messages = [message.model_dump() for message in req.messages]
            payload = {
                "messages": messages,
                "prompt": _prompt_from_messages(req.messages),
                "metadata": {
                    "model": req.model,
                    "roles": [message.role for message in req.messages],
                },
            }
            kwargs = _provider_kwargs(req)
            res = r.call(payload, **kwargs)
            content = _content_from_provider_result(res)
            provider_id = _provider_id_from_result(res)
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            if req.stream:
                return StreamingResponse(
                    _chat_completion_stream(completion_id, created, req.model, content),
                    media_type="text/event-stream",
                )
            message = {
                "role": "assistant",
                "content": content,
            }
            tool_calls = _tool_calls_from_provider_result(res)
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": req.model,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "free_cred": {"provider": provider_id},
            }
        except RuntimeError as e:
            err = {"message": str(e), "type": "free_cred_provider_error", "code": 503}
            return JSONResponse(status_code=503, content={"error": err})

    # ============================================================================
    # Quota Monitor Endpoints & Dashboard
    # ============================================================================

    @app.get("/quota/status")
    async def quota_status():
        """Get quota status for all registered credentials."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}

        quotas = _quota_monitor_instance.get_all_quotas()
        return {
            "credentials": {
                env: {
                    "provider_id": info.provider_id,
                    "model": info.model,
                    "base_url": info.base_url,
                    "remaining": info.remaining,
                    "limit": info.limit,
                    "reset": info.reset.isoformat() if info.reset else None,
                    "ratio": info.ratio,
                    "status": info.status.value,
                    "last_polled": info.last_polled.isoformat() if info.last_polled else None,
                    "last_error": info.last_error,
                    "model_count": len(info.model_list),
                    "model_list_updated": info.model_list_updated.isoformat() if info.model_list_updated else None,
                }
                for env, info in quotas.items()
            },
            "summary": {
                "total": len(quotas),
                "active": sum(1 for q in quotas.values() if q.status.value == "active"),
                "standby": sum(1 for q in quotas.values() if q.status.value == "standby"),
                "exhausted": sum(1 for q in quotas.values() if q.status.value == "exhausted"),
                "error": sum(1 for q in quotas.values() if q.status.value == "error"),
            }
        }

    @app.get("/quota/status/{api_key_env}")
    async def quota_status_single(api_key_env: str):
        """Get quota status for a specific credential."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}

        info = _quota_monitor_instance.get_quota(api_key_env)
        if not info:
            return {"error": "Credential not found"}, 404

        return {
            "api_key_env": api_key_env,
            "provider_id": info.provider_id,
            "model": info.model,
            "base_url": info.base_url,
            "remaining": info.remaining,
            "limit": info.limit,
            "reset": info.reset.isoformat() if info.reset else None,
            "ratio": info.ratio,
            "status": info.status.value,
            "last_polled": info.last_polled.isoformat() if info.last_polled else None,
            "last_error": info.last_error,
            "models": info.model_list,
            "model_list_updated": info.model_list_updated.isoformat() if info.model_list_updated else None,
        }

    @app.post("/quota/credentials")
    async def add_credential(request: AddCredentialRequest):
        """Add a new credential dynamically."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503

        success = _quota_monitor_instance.add_credential(request.api_key_env, request.provider_name, request.api_key, request.model)
        if success:
            return {"message": "Credential added", "api_key_env": request.api_key_env}
        return {"error": "Failed to add credential"}, 400

    @app.delete("/quota/credentials/{api_key_env}")
    async def remove_credential(api_key_env: str):
        """Remove a credential."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503

        success = _quota_monitor_instance.remove_credential(api_key_env)
        if success:
            return {"message": "Credential removed", "api_key_env": api_key_env}
        return {"error": "Credential not found"}, 404

    @app.post("/quota/credentials/{api_key_env}/activate")
    async def activate_credential(api_key_env: str):
        """Mark a credential as active (poll every 30s)."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503

        _quota_monitor_instance.set_active(api_key_env, True)
        return {"message": "Credential activated", "api_key_env": api_key_env}

    @app.post("/quota/credentials/{api_key_env}/deactivate")
    async def deactivate_credential(api_key_env: str):
        """Mark a credential as standby (poll every hour)."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503

        _quota_monitor_instance.set_active(api_key_env, False)
        return {"message": "Credential deactivated", "api_key_env": api_key_env}

    @app.post("/quota/reload-keys")
    async def reload_keys(nvm_path: Optional[str] = None):
        """Trigger reloading of keys from nvm.txt (tries common locations if nvm_path omitted).

        This calls the running quota monitor's loader so the live server picks up keys immediately.
        """
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503
        try:
            await _quota_monitor_instance.load_keys_from_nvm_file(nvm_path)
            return {"message": "Reloaded keys (if any found)"}
        except Exception as e:
            return {"error": str(e)}, 500

    @app.get("/quota/models")
    async def list_all_models():
        """List all available models from all providers."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503

        all_models = []
        for env, info in _quota_monitor_instance.get_all_quotas().items():
            for model in info.model_list:
                model_copy = dict(model)
                model_copy["provider_env"] = env
                model_copy["provider_id"] = info.provider_id
                all_models.append(model_copy)

        return {
            "models": all_models,
            "total": len(all_models),
        }

    @app.get("/quota/notifications")
    async def get_notifications():
        """Get notifications for new models, quota warnings, etc."""
        global _quota_monitor_instance
        if not _quota_monitor_instance:
            return {"error": "Quota monitor not initialized"}, 503

        notifications = []
        now = datetime.now()

        for env, info in _quota_monitor_instance.get_all_quotas().items():
            if info.ratio is not None and info.ratio <= 0.1 and info.ratio > 0:
                notifications.append({
                    "type": "quota_warning",
                    "severity": "warning",
                    "api_key_env": env,
                    "provider_id": info.provider_id,
                    "message": f"Quota low: {info.remaining}/{info.limit} ({info.ratio*100:.1f}%)",
                    "timestamp": now.isoformat(),
                })
            elif info.ratio is not None and info.ratio <= 0:
                notifications.append({
                    "type": "quota_exhausted",
                    "severity": "critical",
                    "api_key_env": env,
                    "provider_id": info.provider_id,
                    "message": f"Quota exhausted for {info.provider_id}",
                    "timestamp": now.isoformat(),
                })

            if info.model_list_updated:
                if (now - info.model_list_updated).total_seconds() < 3600:
                    pass

        return {
            "notifications": notifications,
            "count": len(notifications),
        }

    @app.get("/dashboard")
    async def quota_dashboard():
        """Serve the quota monitoring dashboard."""
        return HTMLResponse(content=_get_dashboard_html(), media_type="text/html")

    return app


def _load_provider_keys_file_from_env() -> None:
    provider_name = os.environ.get("FREE_CRED_KEYS_PROVIDER")
    keys_file = os.environ.get("FREE_CRED_KEYS_FILE")
    if provider_name and keys_file:
        load_api_keys_file(provider_name, keys_file, overwrite=True)


def _health_for_provider_group(providers: list[Provider], group: str) -> dict[str, Any]:
    checks = [_health_for_provider(provider) for provider in providers if _provider_matches_group(provider, group)]
    return {
        "provider": group,
        "status": "healthy" if any(check["ok"] for check in checks) else "unhealthy",
        "ok": any(check["ok"] for check in checks),
        "checks": checks,
    }


def _provider_matches_group(provider: Provider, group: str) -> bool:
    provider_id = provider.id.lower()
    return provider_id == group or provider_id.startswith(f"{group}-") or provider_id.startswith(f"{group}:")


def _health_for_provider(provider: Provider) -> dict[str, Any]:
    if hasattr(provider, "health_check"):
        try:
            result = provider.health_check()
        except Exception as exc:
            result = {
                "provider": provider.id,
                "status": "unhealthy",
                "ok": False,
                "message": "provider health check failed",
                "error_type": exc.__class__.__name__,
            }
    else:
        result = {
            "provider": provider.id,
            "status": "unknown",
            "ok": False,
            "message": "provider does not implement health_check",
        }
    if not isinstance(result, dict):
        result = {
            "provider": provider.id,
            "status": "unhealthy",
            "ok": False,
            "message": "provider health check returned invalid result",
        }
    result.setdefault("provider", provider.id)
    result["ok"] = bool(result.get("ok"))
    if result.get("status") not in {"healthy", "unhealthy", "unknown"}:
        result["status"] = "healthy" if result["ok"] else "unhealthy"
    result["quota"] = _quota_payload(_safe_quota(provider))
    return result


def _safe_quota(provider: Provider) -> QuotaState:
    try:
        return provider.get_quota()
    except Exception:
        return QuotaState()


def _quota_payload(quota: QuotaState) -> dict[str, Any]:
    ratio = None
    if quota.limit and quota.remaining is not None:
        ratio = quota.remaining / quota.limit
    return {
        "remaining": quota.remaining,
        "limit": quota.limit,
        "reset": quota.reset.isoformat() if quota.reset else None,
        "ratio": ratio,
    }


def _provider_kwargs(req: ChatCompletionRequest) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if req.temperature is not None:
        kwargs["temperature"] = req.temperature
    if req.top_p is not None:
        kwargs["top_p"] = req.top_p
    if req.max_tokens is not None:
        kwargs["max_tokens"] = req.max_tokens
    if req.extra_body is not None:
        kwargs["extra_body"] = req.extra_body
    if req.tools is not None:
        kwargs["tools"] = req.tools
    if req.tool_choice is not None:
        kwargs["tool_choice"] = req.tool_choice
    if req.parallel_tool_calls is not None:
        kwargs["parallel_tool_calls"] = req.parallel_tool_calls
    return kwargs


def _prompt_from_messages(messages: list[ChatMessage]) -> str:
    parts = []
    for message in messages:
        content = _message_content_to_text(message.content)
        if content:
            parts.append(f"{message.role}: {content}")
    return "\n".join(parts).strip() or "Respond with exactly: ok"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(str(item.get("text") or ""))
                elif "text" in item:
                    text_parts.append(str(item.get("text") or ""))
            elif item is not None:
                text_parts.append(str(item))
        return "\n".join(part for part in text_parts if part)
    if content is None:
        return ""
    return str(content)


def _content_from_provider_result(result: Any) -> str:
    if isinstance(result, dict):
        if result.get("content") is not None:
            return str(result.get("content"))
        if result.get("result") is not None:
            return str(result.get("result"))
    return str(result)


def _provider_id_from_result(result: Any) -> str:
    if isinstance(result, dict) and result.get("provider") is not None:
        return str(result.get("provider"))
    return "unknown"


def _tool_calls_from_provider_result(result: Any) -> Optional[Any]:
    if isinstance(result, dict) and result.get("tool_calls"):
        return result["tool_calls"]
    return None


def _chat_completion_stream(completion_id: str, created: int, model: str, content: str):
    first = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(first)}\n\n"
    if content:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
    final = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"


def _get_dashboard_html() -> str:
    """Generate the quota monitoring dashboard HTML."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>free-cred-router - Quota Dashboard</title>
  <style>
    :root{
      --bg:#071018; --panel:#0f1720; --muted:#98a1ab; --accent:#7dd3fc; --good:#2ea043; --warn:#ffb020; --bad:#ff6b6b; --card-border:#1f2933;
      --glass: rgba(255,255,255,0.03);
      --radius:14px; --maxw:1200px;
    }
    /* Light (Apple-style) theme overrides */
    .light-theme{
      --bg: linear-gradient(180deg,#f8fafc,#eef2f7); /* subtle Apple-like light */
      --panel: #ffffff;
      --muted: #4b5563;
      --accent: #0ea5a4; /* teal-ish accent */
      --good: #0b9444;
      --warn: #d97706;
      --bad: #dc2626;
      --card-border: rgba(15,23,42,0.06);
      color-scheme: light;
    }
    /* Smooth theme transitions for dynamic toggling */
    body{transition:background-color .25s ease,color .25s ease}

    *{box-sizing:border-box}
    html,body{height:100%}
    body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial; background:linear-gradient(180deg,var(--bg),#071017); color:#e6edf3; margin:0; padding:24px;}
    .wrap{max-width:var(--maxw);margin:0 auto}

    header{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:20px}
    h1{font-size:1.25rem;margin:0}
    .subtitle{color:var(--muted);font-size:0.9rem}

    .top-controls{display:flex;gap:12px;align-items:center}
    .search{background:var(--panel);border:1px solid var(--card-border);padding:8px 12px;border-radius:10px;color:#cfe9ff}
    .btn{border:0;padding:8px 12px;border-radius:10px;cursor:pointer;font-weight:600}
    .btn-primary{background:linear-gradient(180deg,var(--good),#238636);color:white}
    .btn-ghost{background:transparent;border:1px solid var(--card-border);color:var(--muted)}

    .grid{display:grid;grid-template-columns:1fr 360px;gap:20px}
    .main{background:linear-gradient(180deg,var(--panel),#0e1419);border-radius:var(--radius);padding:18px;border:1px solid var(--card-border)}
    .side{background:var(--panel);border-radius:var(--radius);padding:16px;border:1px solid var(--card-border)}

    .cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
    .card{background:linear-gradient(180deg, rgba(255,255,255,0.02), transparent);border-radius:10px;padding:12px;border:1px solid var(--card-border);transition:transform .12s ease,box-shadow .12s ease}
    .card:hover{transform:translateY(-6px);box-shadow:0 10px 30px rgba(2,6,23,0.6)}
    .card-header{display:flex;align-items:center;justify-content:space-between;gap:8px}
    .provider{display:flex;align-items:center;gap:10px}
    .pill{padding:6px 10px;border-radius:999px;background:var(--glass);font-weight:700;font-size:0.78rem}

    .meter{height:12px;background:rgba(255,255,255,0.03);border-radius:999px;overflow:hidden;margin-top:10px;border:1px solid rgba(255,255,255,0.03)}
    .meter > i{display:block;height:100%;width:0%;transition:width .6s cubic-bezier(.2,.8,.2,1);}
    .meter .good{background:linear-gradient(90deg,var(--good),#238636)}
    .meter .warn{background:linear-gradient(90deg,var(--warn),#d29922)}
    .meter .bad{background:linear-gradient(90deg,var(--bad),#f85149)}
    .meta{display:flex;justify-content:space-between;color:var(--muted);font-size:0.85rem;margin-top:8px}
    .model-list{margin-top:10px;display:flex;flex-wrap:wrap;gap:6px}
    .model-tag{font-size:0.75rem;padding:6px 8px;border-radius:8px;background:rgba(0,0,0,0.16);border:1px solid var(--card-border);}

    .actions{display:flex;gap:8px;margin-top:12px;flex-wrap:wrap}

    /* side panel */
    .side h3{margin-top:0}
    .notif{display:flex;flex-direction:column;gap:8px}
    .notification{background:linear-gradient(180deg,rgba(255,255,255,0.02),transparent);padding:10px;border-radius:10px;border:1px solid var(--card-border);display:flex;justify-content:space-between;align-items:center}
    .notification.small{font-size:0.9rem}

    /* modal */
    .modal-backdrop{position:fixed;inset:0;background:rgba(2,6,23,0.6);display:none;align-items:center;justify-content:center;padding:20px}
    .modal{background:var(--panel);padding:18px;border-radius:12px;border:1px solid var(--card-border);min-width:320px;max-width:720px}
    .form-row{display:flex;gap:8px;margin-bottom:8px}

    footer{margin-top:16px;color:var(--muted);font-size:0.85rem}

    @media(max-width:900px){.grid{grid-template-columns:1fr}.side{order:2}}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <div>
        <h1>free-cred-router</h1>
        <div class="subtitle">Quota dashboard • provider-agnostic rotation</div>
      </div>
      <div class="top-controls">
        <input id="qsearch" class="search" placeholder="Search provider or env..." oninput="debouncedRender()" />
        <button class="btn btn-ghost" onclick="refreshNow()">Refresh</button>
        <button class="btn btn-primary" onclick="openAddModal()">Add Credential</button>
          <button id="theme-toggle" class="btn btn-ghost" onclick="toggleTheme()" title="Toggle theme">🌙</button>
        </div>
    </header>

    <div class="grid">
      <main class="main">
        <section style="margin-bottom:12px;display:flex;justify-content:space-between;align-items:center">
          <div style="display:flex;gap:12px;align-items:center">
            <div class="pill" id="summary-pill">Loading…</div>
            <div style="color:var(--muted)">Auto refresh: <strong style="color:#cfe9ff">5s</strong></div>
          </div>
          <div style="color:var(--muted)">Total credentials: <span id="total-count">—</span></div>
        </section>

        <div class="cards" id="credentials-grid">
          <div class="card"><div style="padding:28px;text-align:center;color:var(--muted)">Loading credentials…</div></div>
        </div>
      </main>

      <aside class="side">
        <h3>Notifications</h3>
        <div class="notif" id="notifications-panel">
          <div style="color:var(--muted);font-size:0.9rem">No notifications yet</div>
        </div>

        <h3 style="margin-top:16px">Quick Actions</h3>
        <div style="display:flex;flex-direction:column;gap:8px">
          <button class="btn btn-ghost" onclick="activateAll()">Activate all first keys</button>
          <button class="btn btn-ghost" onclick="deactivateAll()">Set all to standby</button>
          <button class="btn btn-ghost" onclick="clearNotifications()">Clear notifications</button>
        </div>

        <footer>
          <div style="margin-top:12px">UI inspired by handoff design — improved visuals</div>
        </footer>
      </aside>
    </div>
  </div>

  <!-- Modal -->
  <div id="modal-backdrop" class="modal-backdrop">
    <div class="modal">
      <h3>Add Credential</h3>
      <div style="margin-top:8px">
        <div class="form-row"><input id="env-name" placeholder="API Key Env name (e.g. NVIDIA_API_KEY_2)" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--card-border);background:transparent;color:#e6edf3"/></div>
        <div class="form-row"><select id="provider-select" style="padding:8px;border-radius:8px;border:1px solid var(--card-border);background:transparent;color:#e6edf3"><option value="nvidia">NVIDIA</option><option value="groq">Groq</option><option value="openrouter">OpenRouter</option><option value="vercel">Vercel</option><option value="openvino">OpenVINO</option><option value="copilot">Copilot</option></select></div>
        <div class="form-row"><input id="api-key" placeholder="api key (will be stored in env)" style="flex:1;padding:8px;border-radius:8px;border:1px solid var(--card-border);background:transparent;color:#e6edf3"/></div>
        <div class="form-row"><select id="model-select" style="padding:8px;border-radius:8px;border:1px solid var(--card-border);background:transparent;color:#e6edf3"><option value="">(default)</option></select></div>
        <div style="display:flex;gap:8px;margin-top:10px;justify-content:flex-end">
          <button class="btn btn-ghost" onclick="closeAddModal()">Cancel</button>
          <button class="btn btn-primary" onclick="submitAddCredential()">Add</button>
        </div>
      </div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const POLL_INTERVAL = 5000;
    let quotas = {};
    let notifications = [];
    let debounceTimer = null;

    function debouncedRender(){ clearTimeout(debounceTimer); debounceTimer=setTimeout(()=>renderCredentials(quotas),120);}    

    async function fetchQuotaStatus(){
      try{
        const r = await fetch('/quota/status');
        const data = await r.json();
        quotas = data.credentials || {};
        document.getElementById('total-count').textContent = data.summary.total || 0;
        document.getElementById('summary-pill').textContent = `${data.summary.active} active • ${data.summary.standby} standby`;
        renderCredentials(quotas);
      }catch(e){console.warn('quota fetch',e)}
    }

    async function fetchNotifications(){
      try{
        const r = await fetch('/quota/notifications');
        const d = await r.json();
        notifications = d.notifications || [];
        renderNotifications();
      }catch(e){console.warn('notif fetch',e)}
    }

    function filterBySearch(obj){
      const q = document.getElementById('qsearch').value.trim().toLowerCase();
      if(!q) return obj;
      const out = {};
      Object.entries(obj).forEach(([k,v])=>{ if(k.toLowerCase().includes(q) || (v.provider_id||'').toLowerCase().includes(q)) out[k]=v });
      return out;
    }

    function renderCredentials(list){
      const grid = document.getElementById('credentials-grid');
      const filtered = filterBySearch(list);
      grid.innerHTML = '';
      if(Object.keys(filtered).length===0){ grid.innerHTML='<div class="card"><div style="padding:20px;color:var(--muted)">No credentials</div></div>'; return}

      Object.entries(filtered).forEach(([env,info])=>{
        const card = document.createElement('div'); card.className='card';
        const ratio = info.ratio!=null? Math.round(info.ratio*100): null;
        const pct = ratio==null?0:Math.max(0,Math.min(100,ratio));
        const meterClass = (pct<=0? 'bad' : pct<=10? 'warn' : 'good');

        const modelTags = (info.model_list||[]).slice(0,6).map(m=>`<span class="model-tag">${m.id}</span>`).join('');

        card.innerHTML = `
          <div class="card-header">
            <div class="provider"><div style="width:40px;height:40px;border-radius:10px;background:linear-gradient(135deg,#0f1720,#12202a);display:flex;align-items:center;justify-content:center;font-weight:700">${(info.provider_id||'?').slice(0,2).toUpperCase()}</div>
              <div style="margin-left:12px"><div style="font-weight:700">${info.provider_id}</div><div style="color:var(--muted);font-size:0.85rem">${env}</div></div>
            </div>
            <div class="pill">${info.status}</div>
          </div>
          <div style="display:flex;gap:12px;align-items:center;margin-top:10px">
            <canvas id="chart-${env}" class="chart-small" aria-hidden="true"></canvas>
            <div style="flex:1">
              <div class="meter" aria-hidden="true"><i class="${meterClass}" style="width:${pct}%"></i></div>
              <div class="meta"><div>Quota: <strong style="color:#e6edf3">${info.remaining!=null?info.remaining:'?'} / ${info.limit!=null?info.limit:'?'}</strong></div><div>${pct!=null?pct+'%':'?'}</div></div>
            </div>
          </div>
          <div style="margin-top:8px;color:var(--muted);font-size:0.85rem">Model: <strong style="color:#e6edf3">${info.model||'—'}</strong></div>
          <div class="model-list">${modelTags}</div>
          <div class="actions">
            ${info.status==='standby'?`<button class="btn btn-primary" onclick="activateCredential('${env}')">Activate</button>`:''}
            ${info.status==='active'?`<button class="btn btn-ghost" onclick="deactivateCredential('${env}')">Deactivate</button>`:''}
            <button class="btn btn-ghost" onclick="openDetail('${env}')">Details</button>
            <button class="btn btn-ghost" style="color:var(--bad)" onclick="removeCredential('${env}')">Remove</button>
          </div>
        `;
        grid.appendChild(card);
        // create small doughnut chart for quota
        try{
          const ctx = document.getElementById(`chart-${env}`).getContext('2d');
          if(window._charts && window._charts[env]){ try{ window._charts[env].destroy(); }catch(e){} }
          window._charts = window._charts || {};
          const used = info.limit && info.remaining!=null ? Math.max(0, info.limit - info.remaining) : 1;
          const rem = info.remaining!=null ? info.remaining : 0;
          window._charts[env] = new Chart(ctx, {
            type: 'doughnut',
            data: { labels:['used','rem'], datasets:[{ data:[used, rem], backgroundColor:['#ff6b6b','#2ea043'], hoverOffset:8 }] },
            options:{cutout:'70%',plugins:{legend:{display:false}},responsive:false,maintainAspectRatio:false}
          });
        }catch(e){ /* ignore chart errors */ }
      });
    }

    function renderNotifications(){
      const p = document.getElementById('notifications-panel'); p.innerHTML='';
      if(!notifications || notifications.length===0){ p.innerHTML='<div style="color:var(--muted)">No notifications</div>'; return }
      notifications.slice(0,8).forEach(n=>{
        const div = document.createElement('div'); div.className='notification small'; div.innerHTML=`<div><strong>${n.type.replace('_',' ').toUpperCase()}</strong><div style="color:var(--muted);font-size:0.85rem">${n.message}</div></div><div style="color:${n.severity==='critical'? 'var(--bad)': n.severity==='warning'? 'var(--warn)':'var(--accent)'}">${new Date(n.timestamp).toLocaleTimeString()}</div>`;
        p.appendChild(div);
      })
    }

    function refreshNow(){ fetchQuotaStatus(); fetchNotifications(); showToast('Refreshed'); }
    function showToast(msg){ console.log('toast',msg) }

    // Theme handling: store in localStorage, toggle body class
    function setTheme(t){
      if(t==='light'){
        document.body.classList.add('light-theme');
        document.getElementById('theme-toggle').textContent = '☀️';
        localStorage.setItem('freecred_theme','light');
      } else {
        document.body.classList.remove('light-theme');
        document.getElementById('theme-toggle').textContent = '🌙';
        localStorage.setItem('freecred_theme','dark');
      }
    }
    function toggleTheme(){ const cur = localStorage.getItem('freecred_theme')==='light'? 'dark':'light'; setTheme(cur); }
    // Initialize theme from storage or prefers-color-scheme
    (function(){ const saved=localStorage.getItem('freecred_theme'); if(saved){ setTheme(saved); } else if(window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches){ setTheme('light'); } else { setTheme('dark'); } })();

    function openAddModal(){ document.getElementById('modal-backdrop').style.display='flex'; populateModelPicker(); document.getElementById('provider-select').onchange = populateModelPicker; }
    function closeAddModal(){ document.getElementById('modal-backdrop').style.display='none' }

    async function submitAddCredential(){
      const env = document.getElementById('env-name').value.trim(); const prov=document.getElementById('provider-select').value; const key=document.getElementById('api-key').value.trim(); const model = document.getElementById('model-select').value || undefined;
      if(!env||!key) return alert('env & key required');
      try{
        const res = await fetch('/quota/credentials',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({api_key_env:env,provider_name:prov,api_key:key,model:model})});
        if(res.ok){ closeAddModal(); fetchQuotaStatus(); fetchNotifications(); showToast('Credential added') } else { alert('failed to add') }
      }catch(e){alert('error')}
    }

    async function activateAll(){ for(const e of Object.keys(quotas)){ await fetch(`/quota/credentials/${e}/activate`,{method:'POST'}) } fetchQuotaStatus(); }
    async function deactivateAll(){ for(const e of Object.keys(quotas)){ await fetch(`/quota/credentials/${e}/deactivate`,{method:'POST'}) } fetchQuotaStatus(); }
    function clearNotifications(){ notifications=[]; renderNotifications(); }

    function openDetail(env){ alert('Details: '+env) }

    async function activateCredential(env){ await fetch(`/quota/credentials/${env}/activate`,{method:'POST'}); fetchQuotaStatus(); }
    async function deactivateCredential(env){ await fetch(`/quota/credentials/${env}/deactivate`,{method:'POST'}); fetchQuotaStatus(); }
    async function removeCredential(env){ if(!confirm('Remove '+env+'?')) return; await fetch(`/quota/credentials/'+env+'`,{method:'DELETE'}); fetchQuotaStatus(); }

    // initial
    fetchQuotaStatus(); fetchNotifications(); setInterval(fetchQuotaStatus,POLL_INTERVAL); setInterval(fetchNotifications,POLL_INTERVAL);
  </script>
</body>
</html>'''