"""
Core configuration — constants và available models.
"""

from typing import Dict, List

EDITOR_VERSION = "vscode/1.96.2"
USER_AGENT = "CopilotChat/1.0"
DEFAULT_BASE_URL = "https://api.individual.githubcopilot.com"

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

MODELS_BY_ID: Dict[str, Dict] = {m["id"]: m for m in MODELS}
