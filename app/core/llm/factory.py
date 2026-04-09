"""Factory – returns a LangChain BaseChatModel based on LLM_PROVIDER."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)


def get_llm() -> BaseChatModel:
    """Create a new LLM instance based on ``settings.llm_provider``.

    Supported providers:
      - ``vllm``      → OpenAI-compatible local server
      - ``openai``    → OpenAI API
      - ``anthropic`` → Anthropic Claude API

    A new instance is returned on each call so callers may use different
    temperature / model overrides if needed.
    """
    provider = settings.llm_provider

    if provider == "vllm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key="EMPTY",
            openai_api_base=settings.vllm_base_url,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            api_key=settings.anthropic_api_key,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
