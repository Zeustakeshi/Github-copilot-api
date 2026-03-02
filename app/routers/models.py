"""
Router: Models endpoints — list và get model detail.
"""

import time

from fastapi import APIRouter, HTTPException, Path, Request

from app.core.config import MODELS, MODELS_BY_ID
from app.core.state import _start_time
from app.services.copilot import _err, extract_token

router = APIRouter(prefix="/v1", tags=["Models"])


def _model_obj(m: dict) -> dict:
    return {
        "id": m["id"],
        "object": "model",
        "created": int(_start_time),
        "owned_by": m["owned_by"],
    }


@router.get("/models", summary="List models")
async def list_models(request: Request):
    """Lists all available models từ GitHub Copilot."""
    extract_token(request)
    return {"object": "list", "data": [_model_obj(m) for m in MODELS]}


@router.get("/models/{model_id}", summary="Get model")
async def get_model(
    request: Request,
    model_id: str = Path(..., description="The ID of the model"),
):
    """Retrieves details about a specific model."""
    extract_token(request)
    m = MODELS_BY_ID.get(model_id)
    if not m:
        raise HTTPException(
            status_code=404,
            detail=_err(f"Model '{model_id}' not found", "invalid_request_error", "model_not_found"),
        )
    return _model_obj(m)
