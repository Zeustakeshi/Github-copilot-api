"""
Schemas dùng chung cho các utility endpoints.
"""

from typing import Optional

from pydantic import BaseModel


class TokenizeRequest(BaseModel):
    model: Optional[str] = "gpt-4o"
    text: str
