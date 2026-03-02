"""
Router: Text Completions — POST /v1/completions
"""

import json
import time
import uuid
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.state import _stats
from app.models.completion import CompletionRequest
from app.services.copilot import _err, copilot_headers, derive_base_url, extract_token
from app.services.streaming import _sse_response

router = APIRouter(prefix="/v1", tags=["Completions"])


@router.post("/completions", summary="Create text completion")
async def text_completions(request: Request, body: CompletionRequest):
    """
    Text completion — wrap prompt thành chat message rồi gọi Copilot nội bộ.
    """
    token = extract_token(request)
    base_url = derive_base_url(token)
    endpoint = f"{base_url}/chat/completions"

    _stats["total_requests"] += 1
    _stats["active_requests"] += 1

    try:
        payload: Dict[str, Any] = {
            "model": body.model,
            "messages": [{"role": "user", "content": body.prompt}],
            "stream": body.stream,
        }
        if body.max_tokens is not None:
            payload["max_tokens"] = body.max_tokens
        if body.temperature is not None:
            payload["temperature"] = body.temperature

        headers = copilot_headers(token)

        if body.stream:
            async def completion_stream():
                async with httpx.AsyncClient(timeout=120) as client:
                    async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                        if not resp.is_success:
                            body_text = await resp.aread()
                            yield f"data: {json.dumps(_err(body_text.decode()))}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            raw = line[6:]
                            if raw.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                break
                            try:
                                chunk = json.loads(raw)
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                out = {
                                    "id":      chunk.get("id", f"cmpl-{uuid.uuid4().hex}"),
                                    "object":  "text_completion",
                                    "created": chunk.get("created", int(time.time())),
                                    "model":   chunk.get("model", body.model),
                                    "choices": [{"text": content, "index": 0, "finish_reason": None}],
                                }
                                yield f"data: {json.dumps(out)}\n\n"
                            except json.JSONDecodeError:
                                continue

            return _sse_response(completion_stream())

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)

        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=_err(resp.text))

        chat_data = resp.json()
        content = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage   = chat_data.get("usage", {})

        _stats["total_input_tokens"]  += usage.get("prompt_tokens", 0)
        _stats["total_output_tokens"] += usage.get("completion_tokens", 0)

        return {
            "id":      f"cmpl-{uuid.uuid4().hex}",
            "object":  "text_completion",
            "created": int(time.time()),
            "model":   body.model,
            "choices": [{"text": content, "index": 0, "finish_reason": "stop"}],
            "usage":   usage,
        }

    finally:
        _stats["active_requests"] -= 1
