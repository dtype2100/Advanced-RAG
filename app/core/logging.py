"""Centralised logging and tracing configuration."""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str | None = None) -> None:
    """Configure root logger with a structured format.

    Args:
        level: Log level string, e.g. "DEBUG", "INFO", "WARNING".
               Defaults to ``settings.log_level`` when omitted.
    """
    if level is None:
        from app.core.config import settings

        level = settings.log_level

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger instance.

    Args:
        name: Module or component name used as the logger identifier.
    """
    return logging.getLogger(name)
