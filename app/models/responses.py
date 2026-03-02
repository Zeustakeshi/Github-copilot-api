"""
Schemas cho OpenAI 2026 Responses API (Codex CLI compatible).
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


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
