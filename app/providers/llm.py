"""LLM provider abstraction.

Supported providers (selected via LLM_BACKEND env var):
- ``vllm``      : Local vLLM server via OpenAI-compatible API (default)
- ``openai``    : OpenAI Chat API (requires OPENAI_API_KEY)
- ``anthropic`` : Anthropic Claude API (requires ANTHROPIC_API_KEY)

Usage::

    from app.providers.llm import get_llm
    llm = get_llm()
    response = llm.invoke([SystemMessage(...), HumanMessage(...)])
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def get_llm() -> Any:
    """Return a LangChain-compatible chat model for the configured LLM_BACKEND.

    The returned object implements the LangChain ``BaseChatModel`` interface
    so callers can use ``.invoke()``, ``.stream()``, etc. uniformly.

    Returns:
        A LangChain chat model instance.

    Raises:
        ValueError: If LLM_BACKEND is set to an unsupported value.
    """
    backend = settings.llm_backend.lower()

    if backend == "vllm":
        logger.debug(
            "Using vLLM backend at %s, model=%s", settings.vllm_base_url, settings.llm_model
        )
        from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key="EMPTY",  # vLLM does not verify API keys
            openai_api_base=settings.vllm_base_url,
        )

    if backend == "openai":
        logger.debug("Using OpenAI backend, model=%s", settings.llm_model)
        from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "temperature": settings.llm_temperature,
            "api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return ChatOpenAI(**kwargs)

    if backend == "anthropic":
        logger.debug("Using Anthropic backend, model=%s", settings.llm_model)
        try:
            from langchain_anthropic import ChatAnthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "langchain-anthropic is required for LLM_BACKEND=anthropic. "
                "Install it with: pip install langchain-anthropic"
            ) from exc

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )

    raise ValueError(
        f"Unsupported LLM_BACKEND='{backend}'. Supported: 'vllm', 'openai', 'anthropic'."
    )
