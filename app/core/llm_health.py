"""LLM backend readiness probe used by the health endpoint."""

from __future__ import annotations

import logging

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


def probe_llm() -> str:
    """Return a short status string for the configured LLM backend."""
    if settings.llm_backend == "openai":
        if not settings.openai_api_key:
            return "not_configured"
        return "configured"

    try:
        url = f"{settings.vllm_base_url.rstrip('/')}/models"
        with httpx.Client(timeout=2.0) as client:
            response = client.get(url)
            if response.status_code == 200:
                return "ready"
            return f"error: HTTP {response.status_code}"
    except Exception as exc:
        logger.debug("LLM health probe failed: %s", exc)
        return f"unreachable: {exc}"
