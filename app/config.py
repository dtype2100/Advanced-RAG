"""Backward-compatibility shim — re-exports settings from app.core.config.

New code should import directly from ``app.core.config``.
"""

from __future__ import annotations

from app.core.config import Settings, settings  # noqa: F401

__all__ = ["Settings", "settings"]
