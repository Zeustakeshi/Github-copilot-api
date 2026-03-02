"""
Router: Chat Completions — POST /v1/chat/completions
"""

import time
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.state import _stats
from app.models.chat import ChatCompletionRequest
from app.services.copilot import _err, copilot_headers, derive_base_url, extract_token
from app.services.streaming import _forward_stream, _sse_response

router = APIRouter(prefix="/v1", tags=["Chat"])

# Optional fields forwarded to the upstream Copilot API as-is
_OPTIONAL_FIELDS = (
    "temperature",
    "max_tokens",
    "top_p",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "user",
    "reasoning_effort",
)


def _build_payload(body: ChatCompletionRequest) -> dict[str, Any]:
    """Construct the upstream request payload from the incoming request body."""
    # Normalize 'developer' role to 'system' for Copilot compatibility
    messages = []
    for m in body.messages:
        msg = m.model_dump(exclude_none=True)
        if msg.get("role") == "developer":
            msg["role"] = "system"
        messages.append(msg)

    # Normalize max_completion_tokens (GPT-4.5+ style) → max_tokens
    field_values = body.model_dump(exclude_none=True)
    field_values["max_tokens"] = body.max_tokens or body.max_completion_tokens

    payload: dict[str, Any] = {
        "model": body.model,
        "messages": messages,
        "stream": body.stream,
    }

    for field in _OPTIONAL_FIELDS:
        val = field_values.get(field)
        if val is not None:
            payload[field] = val

    if body.tools:
        payload["tools"] = [t.model_dump(exclude_none=True) for t in body.tools]
    if body.tool_choice is not None:
        payload["tool_choice"] = body.tool_choice
    if body.response_format:
        payload["response_format"] = body.response_format.model_dump()

    return payload


@router.post("/chat/completions", summary="Create chat completion")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    Supports streaming (SSE), function calling/tools, JSON mode, and reasoning_effort.
    """
    token = extract_token(request)
    endpoint = f"{derive_base_url(token)}/chat/completions"
    headers = copilot_headers(token)
    payload = _build_payload(body)

    _stats["total_requests"] += 1
    _stats["active_requests"] += 1

    try:
        if body.stream:
            return _sse_response(_forward_stream(endpoint, payload, headers))

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)

        if resp.status_code == 401:
            raise HTTPException(status_code=401, detail=_err("Invalid Copilot token", code="401"))
        if resp.status_code == 429:
            raise HTTPException(status_code=429, detail=_err("Rate limited", code="429"))
        if not resp.is_success:
            raise HTTPException(status_code=resp.status_code, detail=_err(resp.text))

        data = resp.json()

        usage = data.get("usage", {})
        _stats["total_input_tokens"] += usage.get("prompt_tokens", 0)
        _stats["total_output_tokens"] += usage.get("completion_tokens", 0)

        data.setdefault("id", f"chatcmpl-{uuid.uuid4().hex}")
        data.setdefault("object", "chat.completion")
        data.setdefault("created", int(time.time()))
        return data

    finally:
        _stats["active_requests"] -= 1