"""
Schemas cho Text Completions API (OpenAI-compatible).
"""

from typing import Optional

from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str = "gpt-4o-copilot"
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = None
    stream: Optional[bool] = False
