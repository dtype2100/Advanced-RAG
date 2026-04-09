"""LLM factory (OpenAI-compatible API: OpenAI or vLLM)."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_chat_llm() -> ChatOpenAI:
    """Create a chat model instance for the configured LLM backend."""
    if settings.using_vllm:
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key="EMPTY",
            openai_api_base=settings.vllm_base_url,
        )
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )
