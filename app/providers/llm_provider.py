"""LLM provider — returns a ``ChatOpenAI`` instance for the configured backend.

Supports:
- ``llm_backend=openai``  → OpenAI API
- ``llm_backend=vllm``    → local vLLM server (OpenAI-compatible endpoint)
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_llm: ChatOpenAI | None = None


def get_llm() -> ChatOpenAI:
    """Return a cached ``ChatOpenAI`` instance for the configured LLM backend.

    The instance is reused across requests to avoid repeated initialisation.
    """
    global _llm
    if _llm is not None:
        return _llm

    if settings.using_vllm:
        logger.info("LLM provider: vLLM @ %s  model=%s", settings.vllm_base_url, settings.llm_model)
        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key="EMPTY",  # vLLM does not validate API keys
            openai_api_base=settings.vllm_base_url,
        )
    else:
        logger.info("LLM provider: OpenAI  model=%s", settings.llm_model)
        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )

    return _llm
