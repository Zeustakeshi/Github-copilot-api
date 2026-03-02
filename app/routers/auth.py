"""
Router: Auth — POST /v1/auth/token

Accepts a GitHub token (Personal Access Token or OAuth token with Copilot scope),
exchanges it for a Copilot API token via the GitHub API, and returns the result.
"""

import logging
import re

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.services.copilot import _err

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["Auth"])

COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

# Threshold to distinguish seconds vs milliseconds timestamps (year ~2286)
_MILLIS_THRESHOLD = 10_000_000_000


def _extract_github_token(request: Request) -> str:
    """Extract and return the GitHub token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail=_err(
                "Missing or invalid Authorization header. Use: Bearer <github_token>",
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


def _derive_base_url(copilot_token: str) -> str:
    """
    Parse the proxy-ep field from a Copilot token and return the appropriate base URL.

    Falls back to the individual Copilot API URL if proxy-ep is not present.
    """
    match = re.search(r"(?:^|;)\s*proxy-ep=([^;\s]+)", copilot_token, re.IGNORECASE)
    if not match:
        return "https://api.individual.githubcopilot.com"

    proxy_ep = match.group(1).strip()
    # Strip scheme, then replace leading "proxy." with "api."
    host = re.sub(r"^https?://", "", proxy_ep)
    host = re.sub(r"^proxy\.", "api.", host, flags=re.IGNORECASE)
    return f"https://{host}"


@router.post("/token", summary="Exchange GitHub token → Copilot token")
async def exchange_token(request: Request):
    """
    Exchange a GitHub Personal Access Token (`ghp_...`) or OAuth token (`gho_...`)
    with Copilot access for a short-lived Copilot API token.

    **Authorization header**: `Bearer <your_github_token>`

    **Response**:
    - `copilot_token`: Token for authenticating chat/completions requests
    - `expires_at`: Expiry as a Unix timestamp in milliseconds
    - `base_url`: Copilot API base URL derived from the token
    """
    github_token = _extract_github_token(request)

    async with httpx.AsyncClient(timeout=30, headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {github_token}",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36 Edg/145.0.0.0"
    }   # match node-fetch reference)
    ) as client:
        # Use a minimal header set; httpx's default User-Agent is rejected by the GitHub internal API
        resp = await client.get(
            COPILOT_TOKEN_URL,
        )

    if resp.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail=_err(
                "Invalid GitHub token or missing Copilot subscription",
                "invalid_request_error",
                "invalid_api_key",
            ),
        )
    if resp.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail=_err(
                "GitHub token does not have permission to access Copilot",
                "invalid_request_error",
                "forbidden",
            ),
        )
    if not resp.is_success:
        raise HTTPException(
            status_code=502,
            detail=_err(
                f"GitHub API returned HTTP {resp.status_code}: {resp.text}",
                "api_error",
                "upstream_error",
            ),
        )

    data = resp.json()
    raw_token: str = data.get("token", "")
    expires_at_raw: int = data.get("expires_at", 0)

    # GitHub returns seconds; normalize to milliseconds
    expires_at_ms = expires_at_raw if expires_at_raw > _MILLIS_THRESHOLD else expires_at_raw * 1000

    return {
        "copilot_token": raw_token,
        "expires_at": expires_at_ms,
        "base_url": _derive_base_url(raw_token),
    }