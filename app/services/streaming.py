"""
Streaming service — SSE response builder và forward stream helper.
"""

import json
from typing import AsyncGenerator

import httpx
from fastapi.responses import StreamingResponse

from app.services.copilot import _err


def _sse_response(generator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


async def _forward_stream(
    endpoint: str, payload: dict, headers: dict
) -> AsyncGenerator[str, None]:
    """Forward SSE stream thẳng từ Copilot → client."""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
            if resp.status_code == 401:
                yield f"data: {json.dumps(_err('Invalid Copilot token', code='401'))}\n\n"
                yield "data: [DONE]\n\n"
                return
            if not resp.is_success:
                body_text = await resp.aread()
                yield f"data: {json.dumps(_err(body_text.decode(), code=str(resp.status_code)))}\n\n"
                yield "data: [DONE]\n\n"
                return
            async for line in resp.aiter_lines():
                if not line:
                    continue
                yield f"{line}\n\n"
                if line.strip() == "data: [DONE]":
                    break
