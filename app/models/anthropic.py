"""
Schemas cho Anthropic Messages API compatible endpoint.
"""

from typing import List, Optional

from pydantic import BaseModel


class AnthropicMessage(BaseModel):
    role: str  # user | assistant
    content: str


class AnthropicRequest(BaseModel):
    model: str
    messages: List[AnthropicMessage]
    system: Optional[str] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
