"""Legacy compatibility router — aggregates all v1 sub-routers.

Kept for backward compatibility with tests importing ``app.api.routes.router``.
New code should import directly from ``app.api.v1.*``.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.ingest import router as ingest_router

router = APIRouter()
router.include_router(health_router)
router.include_router(ingest_router)
router.include_router(chat_router)
