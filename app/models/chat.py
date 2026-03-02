"""
Schemas cho Chat Completions API (OpenAI-compatible).
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


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
