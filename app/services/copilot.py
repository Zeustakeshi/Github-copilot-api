"""
Copilot service — token extraction, base URL derivation, header building, error formatting.
"""

import re

from fastapi import HTTPException, Request

from app.core.config import DEFAULT_BASE_URL, EDITOR_VERSION, USER_AGENT


def extract_token(request: Request) -> str:
    """Lấy Copilot token từ Authorization header (như OpenAI API key)."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=_err(
                "Missing or invalid Authorization header. Use: Bearer <copilot_token>",
                "invalid_request_error",
                "invalid_api_key",
            ),
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
