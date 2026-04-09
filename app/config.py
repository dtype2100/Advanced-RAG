"""Backward-compatible re-export; prefer `from app.core.config import settings`."""

from app.core.config import Settings, settings

__all__ = ["Settings", "settings"]
