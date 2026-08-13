import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .schemas import RouteRequest, RouteResponse, ErrorResponse
from .providers import providers_from_env
from .core import Router, CircuitBreaker
from typing import Optional, Callable


def create_app(providers=None, preventive_threshold: float = 0.1, max_retries: int = 2, cb_factory: Optional[Callable[[], CircuitBreaker]] = None) -> FastAPI:
    """Factory to create FastAPI app without side effects.
    Reads PROVIDERS and THRESHOLD env variables if providers is None.
    """
    if providers is None:
        providers = providers_from_env()
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

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/route", response_model=RouteResponse, responses={503: {"model": ErrorResponse}})
    def route(req: RouteRequest):
        try:
            # router.call may raise RuntimeError when no provider available
            # use model_dump for pydantic v2 compatibility
            res = r.call(req.model_dump())
            # ensure provider id is returned explicitly
            if isinstance(res, dict) and "provider" in res:
                provider_id = res.get("provider")
            else:
                # fallback deterministic provider id
                provider_id = getattr(res, "provider", "unknown")
            return {"provider_id": provider_id, "result": res}
        except RuntimeError as e:
            err = {"message": str(e)}
            return JSONResponse(status_code=503, content={"error": err})

    return app
