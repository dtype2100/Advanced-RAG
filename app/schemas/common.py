"""Shared / generic schema primitives."""

from __future__ import annotations

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error payload."""

    detail: str
