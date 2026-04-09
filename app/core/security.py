"""Authentication and authorisation helpers.

Placeholder module for future API-key / JWT-based auth middleware.
"""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(_API_KEY_HEADER)) -> None:
    """Validate the X-API-Key header if a key is configured.

    Currently a no-op stub: configure ``API_KEY`` env var to enable.
    """
    from app.core.config import settings

    configured_key = getattr(settings, "api_key", "")
    if configured_key and api_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
