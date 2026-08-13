from pydantic import BaseModel
from typing import Optional, Dict, Any


class RouteRequest(BaseModel):
    prompt: str
    metadata: Optional[Dict[str, Any]] = None


class RouteResponse(BaseModel):
    provider_id: str
    result: Any


class ErrorResponse(BaseModel):
    error: Dict[str, Any]
