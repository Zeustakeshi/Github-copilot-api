"""
Copilot API Gateway - FastAPI
OpenAI-compatible API proxy cho GitHub Copilot.

User truyền Copilot token như OpenAI API key:
  Authorization: Bearer <copilot_token>

Endpoints theo đúng OpenAPI spec:
  POST /v1/chat/completions       — Chat (stream + non-stream, tools, json_mode)
  POST /v1/completions            — Text completion
  POST /v1/responses              — OpenAI 2026 Responses API (Codex CLI compatible)
  GET  /v1/models                 — List models
  GET  /v1/models/{model_id}      — Get model detail
  POST /v1/tokenize               — Count tokens
  GET  /v1/usage                  — Usage statistics
  POST /v1/messages               — Anthropic Messages API compatible
  GET  /health                    — Health check
"""

import json
import re
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
import tiktoken
from fastapi import FastAPI, HTTPException, Path, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# ══════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════

EDITOR_VERSION = "vscode/1.96.2"
USER_AGENT = "CopilotChat/1.0"
DEFAULT_BASE_URL = "https://api.individual.githubcopilot.com"

# ══════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════

_start_time = time.time()
_stats = {
    "total_requests": 0,
    "active_requests": 0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
}

# ══════════════════════════════════════════════════════════
# AVAILABLE MODELS
# ══════════════════════════════════════════════════════════

MODELS: List[Dict] = [
    {"id": "gpt-4o-copilot",         "owned_by": "github-copilot", "context_window": 128000},
    {"id": "gpt-4o",                 "owned_by": "github-copilot", "context_window": 128000},
    {"id": "gpt-4o-mini",            "owned_by": "github-copilot", "context_window": 128000},
    {"id": "gpt-4",                  "owned_by": "github-copilot", "context_window": 8192},
    {"id": "gpt-4-turbo",            "owned_by": "github-copilot", "context_window": 128000},
    {"id": "o1",                     "owned_by": "github-copilot", "context_window": 128000},
    {"id": "o1-preview",             "owned_by": "github-copilot", "context_window": 128000},
    {"id": "o1-mini",                "owned_by": "github-copilot", "context_window": 128000},
    {"id": "o3-mini",                "owned_by": "github-copilot", "context_window": 128000},
    {"id": "claude-3.5-sonnet",      "owned_by": "github-copilot", "context_window": 200000},
    {"id": "claude-3.7-sonnet",      "owned_by": "github-copilot", "context_window": 200000},
    {"id": "gemini-2.0-flash-001",   "owned_by": "github-copilot", "context_window": 1048576},
]
MODELS_BY_ID = {m["id"]: m for m in MODELS}

# ══════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════

app = FastAPI(
    title="Copilot API Gateway",
    description=(
        "Local OpenAI-compatible API server powered by GitHub Copilot. "
        "Provides REST endpoints for chat completions, text completions, "
        "tokenization, and more."
    ),
    version="1.0.0",
    contact={"name": "GitHub Copilot API Gateway"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def extract_token(request: Request) -> str:
    """Lấy Copilot token từ Authorization header (như OpenAI API key)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=_err("Missing or invalid Authorization header. Use: Bearer <copilot_token>",
                        "invalid_request_error", "invalid_api_key"),
        )
    token = auth[len("Bearer "):].strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail=_err("Empty token", "invalid_request_error", "invalid_api_key"),
        )
    return token


def derive_base_url(copilot_token: str) -> str:
    """
    Parse proxy-ep từ Copilot token để lấy đúng base URL.
    Token format: tid=xxx;proxy-ep=proxy.individual.githubcopilot.com;...
    → https://api.individual.githubcopilot.com
    """
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", copilot_token, re.IGNORECASE)
    if not match:
        return DEFAULT_BASE_URL
    proxy_ep = match.group(1).strip()
    host = re.sub(r"^https?://", "", proxy_ep)
    host = re.sub(r"^proxy\.", "api.", host, flags=re.IGNORECASE)
    return f"https://{host}"


def copilot_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Editor-Version": EDITOR_VERSION,
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
        "openai-intent": "conversation-panel",
    }


def _err(message: str, error_type: str = "api_error", code: str = "") -> dict:
    return {"error": {"message": message, "type": error_type, "code": code}}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Đếm token bằng tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _model_obj(m: dict) -> dict:
    return {
        "id": m["id"],
        "object": "model",
        "created": int(_start_time),
        "owned_by": m["owned_by"],
    }


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


# ══════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════

class MessageSchema(BaseModel):
    role: str  # system | developer | user | assistant | tool
    content: Union[str, List[Any]]
    name: Optional[str] = None
    tool_calls: Optional[List[Any]] = None
    tool_call_id: Optional[str] = None


class FunctionSchema(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict] = None


class ToolSchema(BaseModel):
    type: str = "function"
    function: FunctionSchema


class ResponseFormatSchema(BaseModel):
    type: str = "text"  # text | json_object


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o-copilot"
    messages: List[MessageSchema]
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None  # auto-normalize → max_tokens
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    tools: Optional[List[ToolSchema]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    response_format: Optional[ResponseFormatSchema] = None
    reasoning_effort: Optional[str] = None  # low|medium|high|minimal|none|xhigh
    user: Optional[str] = None


class CompletionRequest(BaseModel):
    model: str = "gpt-4o-copilot"
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = None
    stream: Optional[bool] = False


class ResponsesInputMessage(BaseModel):
    role: str
    content: str


class ResponsesRequest(BaseModel):
    model: Optional[str] = "gpt-4o-copilot"
    input: Union[str, List[ResponsesInputMessage]]
    instructions: Optional[str] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[str] = None
    reasoning: Optional[Dict] = None
    truncation: Optional[str] = "disabled"
    store: Optional[bool] = True
    previous_response_id: Optional[str] = None
    metadata: Optional[Dict] = None


class TokenizeRequest(BaseModel):
    model: Optional[str] = "gpt-4o"
    text: str


class AnthropicMessage(BaseModel):
    role: str  # user | assistant
    content: str


class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


# ══════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════

# ── Health ────────────────────────────────────────────────

@app.get("/health", tags=["Utilities"], summary="Health check")
async def health_check():
    return {"status": "ok", "service": "github-copilot-api-gateway"}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Copilot API Gateway", "docs": "/docs", "health": "/health"}


# ── Models ────────────────────────────────────────────────

@app.get("/v1/models", tags=["Models"], summary="List models")
async def list_models(request: Request):
    """Lists all available models từ GitHub Copilot."""
    extract_token(request)
    return {"object": "list", "data": [_model_obj(m) for m in MODELS]}


@app.get("/v1/models/{model_id}", tags=["Models"], summary="Get model")
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


# ── Chat Completions ──────────────────────────────────────

@app.post("/v1/chat/completions", tags=["Chat"], summary="Create chat completion")
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


# ── Text Completions ──────────────────────────────────────

@app.post("/v1/completions", tags=["Completions"], summary="Create text completion")
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


# ── Responses API (OpenAI 2026 / Codex CLI) ──────────────

@app.post("/v1/responses", tags=["Chat"], summary="Create response (OpenAI 2026 Responses API)")
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


# ── Anthropic Messages API ────────────────────────────────

@app.post("/v1/messages", tags=["Anthropic"], summary="Anthropic Messages API")
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

    headers    = copilot_headers(token)
    msg_id     = f"msg-{uuid.uuid4().hex}"

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


# ── Tokenize ──────────────────────────────────────────────

@app.post("/v1/tokenize", tags=["Utilities"], summary="Count tokens")
async def tokenize(body: TokenizeRequest):
    """Đếm số token trong text bằng tiktoken (local, không cần gọi API)."""
    token_count = count_tokens(body.text, body.model or "gpt-4o")
    return {"model": body.model or "gpt-4o", "token_count": token_count}


# ── Usage ─────────────────────────────────────────────────

@app.get("/v1/usage", tags=["Utilities"], summary="Get usage statistics")
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