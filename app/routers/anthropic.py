"""
Router: Anthropic Messages API — POST /v1/messages
"""

import json
import uuid
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.state import _stats
from app.models.anthropic import AnthropicRequest
from app.services.copilot import _err, copilot_headers, derive_base_url, extract_token
from app.services.streaming import _sse_response

router = APIRouter(prefix="/v1", tags=["Anthropic"])


@router.post("/messages", summary="Anthropic Messages API")
async def anthropic_messages(request: Request, body: AnthropicRequest):
    """
    Anthropic-compatible /v1/messages — chuyển đổi sang Copilot nội bộ.
    SSE events theo chuẩn Anthropic: message_start, content_block_delta, ...
    """
    token = extract_token(request)
    base_url = derive_base_url(token)
    endpoint = f"{base_url}/chat/completions"

    _stats["total_requests"] += 1

    messages: List[Dict] = []
    if body.system:
        messages.append({"role": "system", "content": body.system})
    for m in body.messages:
        messages.append({"role": m.role, "content": m.content})

    payload: Dict[str, Any] = {
        "model":    body.model,
        "messages": messages,
        "stream":   body.stream,
    }
    if body.max_tokens:
        payload["max_tokens"] = body.max_tokens

    headers = copilot_headers(token)
    msg_id  = f"msg-{uuid.uuid4().hex}"

    if body.stream:
        async def anthropic_stream():
            output_tokens = 0

            yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': body.model, 'stop_reason': None}})}\n\n"
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", endpoint, json=payload, headers=headers) as resp:
                    if not resp.is_success:
                        err_body = await resp.aread()
                        yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': err_body.decode()})}\n\n"
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
                                output_tokens += 1
                                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                        except json.JSONDecodeError:
                            continue

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': output_tokens}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

        return _sse_response(anthropic_stream())

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(endpoint, json=payload, headers=headers)

    if not resp.is_success:
        raise HTTPException(status_code=resp.status_code, detail=_err(resp.text))

    chat_data = resp.json()
    content   = chat_data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage     = chat_data.get("usage", {})

    _stats["total_input_tokens"]  += usage.get("prompt_tokens", 0)
    _stats["total_output_tokens"] += usage.get("completion_tokens", 0)

    return {
        "id":          msg_id,
        "type":        "message",
        "role":        "assistant",
        "content":     [{"type": "text", "text": content}],
        "model":       body.model,
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
