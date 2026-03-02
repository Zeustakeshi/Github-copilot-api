"""
Router: Chat Completions — POST /v1/chat/completions
"""

import time
import uuid
from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.state import _stats
from app.models.chat import ChatCompletionRequest
from app.services.copilot import _err, copilot_headers, derive_base_url, extract_token
from app.services.streaming import _forward_stream, _sse_response

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post("/chat/completions", summary="Create chat completion")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions.
    Hỗ trợ: Streaming (SSE), Function calling/Tools, JSON mode, reasoning_effort.
    """
    token = extract_token(request)
    base_url = derive_base_url(token)
    endpoint = f"{base_url}/chat/completions"

    _stats["total_requests"] += 1
    _stats["active_requests"] += 1

    try:
        # Normalize: developer role → system
        messages = []
        for m in body.messages:
            msg = m.model_dump(exclude_none=True)
            if msg.get("role") == "developer":
                msg["role"] = "system"
            messages.append(msg)

        # max_completion_tokens → max_tokens (GPT-5.x style, auto-normalize)
        max_tokens = body.max_tokens or body.max_completion_tokens

        payload: Dict[str, Any] = {
            "model": body.model,
            "messages": messages,
            "stream": body.stream,
        }

        for field, val in [
            ("temperature",        body.temperature),
            ("max_tokens",         max_tokens),
            ("top_p",              body.top_p),
            ("n",                  body.n),
            ("stop",               body.stop),
            ("presence_penalty",   body.presence_penalty),
            ("frequency_penalty",  body.frequency_penalty),
            ("user",               body.user),
            ("reasoning_effort",   body.reasoning_effort),
        ]:
            if val is not None:
                payload[field] = val

        if body.tools:
            payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
        if body.tool_choice is not None:
            payload["tool_choice"] = body.tool_choice
        if body.response_format:
            payload["response_format"] = body.response_format.model_dump()

        headers = copilot_headers(token)

        # ── Stream ──────────────────────────────────────
        if body.stream:
            return _sse_response(_forward_stream(endpoint, payload, headers))

        # ── Non-stream ──────────────────────────────────
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)

        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail=_err("Invalid Copilot token", code="401"))
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail=_err("Rate limited", code="429"))
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=_err(resp.text))

        data = resp.json()

        # Track usage
        usage = data.get("usage", {})
        _stats["total_input_tokens"]  += usage.get("prompt_tokens", 0)
        _stats["total_output_tokens"] += usage.get("completion_tokens", 0)

        data.setdefault("id",      f"chatcmpl-{uuid.uuid4().hex}")
        data.setdefault("object",  "chat.completion")
        data.setdefault("created", int(time.time()))
        return data

    finally:
        _stats["active_requests"] -= 1
