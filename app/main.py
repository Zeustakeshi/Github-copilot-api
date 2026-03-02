"""
App factory — creates the FastAPI instance, configures middleware, and registers all routers.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.routers import anthropic, auth, chat, completions, health, models, responses, utilities

_DESCRIPTION = (
    "Local OpenAI-compatible API server powered by GitHub Copilot. "
    "Provides REST endpoints for chat completions, text completions, tokenization, and more.\n\n"
    "**Authentication:** Click the **Authorize 🔓** button above, enter your GitHub token or "
    "Copilot token in the `BearerAuth (http, Bearer)` field, then click **Authorize**. "
    "Swagger will automatically send `Authorization: Bearer <token>` on every request."
)

_BEARER_SCHEME = {
    "BearerAuth": {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "Token",
        "description": (
            "Enter a **Copilot token** (used directly with chat/models endpoints) "
            "or a **GitHub Personal Access Token** (used with `POST /v1/auth/token` "
            "to exchange for a Copilot token)."
        ),
    }
}

app = FastAPI(
    title="Copilot API Gateway",
    description=_DESCRIPTION,
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

for _router in (
    auth.router,
    health.router,
    models.router,
    chat.router,
    completions.router,
    responses.router,
    anthropic.router,
    utilities.router,
):
    app.include_router(_router)


def _custom_openapi() -> dict:
    """
    Override the OpenAPI schema to inject an HTTP Bearer security scheme.

    Enables the 'Authorize 🔓' button in Swagger UI so users can enter
    a token once and have it applied to all requests automatically.
    """
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    schema.setdefault("components", {})["securitySchemes"] = _BEARER_SCHEME

    # Apply BearerAuth globally to all operations
    for path_item in schema.get("paths", {}).values():
        for operation in path_item.values():
            if isinstance(operation, dict):
                operation.setdefault("security", [{"BearerAuth": []}])

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = _custom_openapi  # type: ignore[method-assign]