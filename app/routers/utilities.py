"""
Router: Utility endpoints — tokenize và usage statistics.
"""

import time

from fastapi import APIRouter

from app.core.state import _start_time, _stats
from app.models.common import TokenizeRequest
from app.services.tokenizer import count_tokens

router = APIRouter(prefix="/v1", tags=["Utilities"])


@router.post("/tokenize", summary="Count tokens")
async def tokenize(body: TokenizeRequest):
    """Đếm số token trong text bằng tiktoken (local, không cần gọi API)."""
    token_count = count_tokens(body.text, body.model or "gpt-4o")
    return {"model": body.model or "gpt-4o", "token_count": token_count}


@router.get("/usage", summary="Get usage statistics")
async def get_usage():
    """Returns usage statistics: requests, tokens, uptime."""
    uptime = int(time.time() - _start_time)
    return {
        "object": "usage",
        "total_requests": _stats["total_requests"],
        "total_tokens": {
            "input":  _stats["total_input_tokens"],
            "output": _stats["total_output_tokens"],
            "total":  _stats["total_input_tokens"] + _stats["total_output_tokens"],
        },
        "uptime_seconds":  uptime,
        "active_requests": _stats["active_requests"],
    }
