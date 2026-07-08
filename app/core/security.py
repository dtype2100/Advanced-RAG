"""Authentication and authorisation helpers.

Placeholder module for future API-key / JWT-based auth middleware.
"""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    """Validate the X-API-Key header when ``API_KEY`` is configured."""
    from app.core.config import settings

    if settings.api_key and api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
