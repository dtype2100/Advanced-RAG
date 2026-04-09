"""LLM factory – returns a ``ChatOpenAI`` instance for vLLM or OpenAI."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import settings


def get_llm() -> ChatOpenAI:
    """Build a ChatOpenAI instance based on the configured backend.

    ``LLM_BACKEND=vllm`` → points at local vLLM server with a dummy key.
    ``LLM_BACKEND=openai`` → uses the real OpenAI API.
    """
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
