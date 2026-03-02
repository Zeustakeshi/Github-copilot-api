"""
Copilot API Gateway — entry point.`

Khởi chạy server:
  uvicorn main:app --reload
  uvicorn main:app --host 0.0.0.0 --port 8000
"""

# Re-export app từ app package để uvicorn/gunicorn có thể dùng `main:app`
from app.main import app  # noqa: F401  