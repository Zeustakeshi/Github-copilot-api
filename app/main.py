"""
App factory — tạo FastAPI instance, cấu hình middleware, và đăng ký tất cả routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import anthropic, chat, completions, health, models, responses, utilities

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

app.include_router(health.router)
app.include_router(models.router)
app.include_router(chat.router)
app.include_router(completions.router)
app.include_router(responses.router)
app.include_router(anthropic.router)
app.include_router(utilities.router)
