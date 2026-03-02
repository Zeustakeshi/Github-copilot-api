"""
Router: OpenAI 2026 Responses API — POST /v1/responses (Codex CLI compatible)
"""

import json
import time
import uuid
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.state import _stats
from app.models.responses import ResponsesRequest
from app.services.copilot import _err, copilot_headers, derive_base_url, extract_token
from app.services.streaming import _sse_response

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post("/responses", summary="Create response (OpenAI 2026 Responses API)")
async def create_response(request: Request, body: ResponsesRequest):
    """
    OpenAI Responses API — Codex CLI compatible.
    Chuyển đổi nội bộ thành /chat/completions.
    """
    token = extract_token(request)
    base_url = derive_base_url(token)
    endpoint = f"{base_url}/chat/completions"

    _stats["total_requests"] += 1

    messages: List[Dict] = []
    if body.instructions:
        messages.append({"role": "system", "content": body.instructions})
    if isinstance(body.input, str):
        messages.append({"role": "user", "content": body.input})
    else:
        for m in body.input:
            messages.append({"role": m.role, "content": m.content})

    payload: Dict[str, Any] = {
        "model": body.model or "gpt-4o-copilot",
        "messages": messages,
        "stream": body.stream,
    }
    if body.temperature is not None:
        payload["temperature"] = body.temperature
    if body.top_p is not None:
        payload["top_p"] = body.top_p
    if body.max_output_tokens is not None:
        payload["max_tokens"] = body.max_output_tokens
    if body.tools:
        payload["tools"] = body.tools
    if body.tool_choice:
        payload["tool_choice"] = body.tool_choice
    if body.reasoning:
        effort = body.reasoning.get("effort")
        if effort:
            payload["reasoning_effort"] = effort

    headers   = copilot_headers(token)
    resp_id   = f"resp-{uuid.uuid4().hex}"
    created_at = int(time.time())

    if body.stream:
        async def responses_stream():
            item_id      = f"msg-{uuid.uuid4().hex}"
            full_content = ""

            yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'status': 'in_progress', 'created_at': created_at, 'model': payload['model']}})}\n\n"
            yield f"data: {json.dumps({'type': 'response.output_item.added', 'item': {'id': item_id, 'type': 'message', 'role': 'assistant', 'content': []}})}\n\n"

            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    if not resp.is_success:
                        err_body = await resp.aread()
                        yield f"data: {json.dumps({'type': 'error', 'error': err_body.decode()})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw = line[6:]
                        if raw.strip() == "[DONE]":
                            break
                        try:
                            chunk   = json.loads(raw)
                            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                            if content:
                                full_content += content
                                yield f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': content, 'item_id': item_id})}\n\n"
                        except json.JSONDecodeError:
                            continue

            yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': resp_id, 'object': 'response', 'status': 'completed', 'created_at': created_at, 'model': payload['model'], 'output': [{'type': 'message', 'id': item_id, 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_content}]}]}})}\n\n"
            yield "data: [DONE]\n\n"

        return _sse_response(responses_stream())

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)

    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail=_err(resp.text))

    chat_data = resp.json()
    content   = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage     = chat_data.get("usage", {})

    return {
        "id":         resp_id,
        "object":     "response",
        "created_at": created_at,
        "model":      payload["model"],
        "status":     "completed",
        "output": [
            {
                "type":    "message",
                "id":      f"msg-{uuid.uuid4().hex}",
                "role":    "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens":  usage.get("total_tokens", 0),
        },
    }
