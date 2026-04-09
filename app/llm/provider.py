"""LLM provider helpers."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from app.config import settings


def message_content_to_text(content: Any) -> str:
    """Normalize LangChain message content to plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "".join(chunks).strip()

    return str(content).strip()


def get_chat_model(model_name: str | None = None) -> ChatOpenAI:
    """Build a ChatOpenAI client from the current environment configuration."""
    selected_model = model_name or settings.llm_model

    if settings.using_vllm:
        return ChatOpenAI(
            model=selected_model,
            temperature=settings.llm_temperature,
            openai_api_key="EMPTY",
            openai_api_base=settings.vllm_base_url,
        )

    return ChatOpenAI(
        model=selected_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )
