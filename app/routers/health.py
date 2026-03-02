"""
Router: Health check endpoints.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Utilities"])


@router.get("/health", summary="Health check")
async def health_check():
    return {"status": "ok", "service": "github-copilot-api-gateway"}


@router.get("/", include_in_schema=False)
async def root():
    return {"message": "Copilot API Gateway", "docs": "/docs", "health": "/health"}
