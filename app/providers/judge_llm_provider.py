"""Judge LLM provider — dedicated model instance for evaluation / grading tasks.

Keeping a separate judge provider allows using a stronger/different model for
LLM-as-judge evaluation without changing the generation model.

Environment variables:
    JUDGE_LLM_BACKEND : ``openai`` | ``vllm``  (default: same as ``llm_backend``)
    JUDGE_LLM_MODEL   : model name              (default: same as ``llm_model``)
    JUDGE_LLM_TEMPERATURE : float               (default: 0.0)
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_judge_llm: ChatOpenAI | None = None


def get_judge_llm() -> ChatOpenAI:
    """Return a cached ``ChatOpenAI`` instance configured for judge tasks.

    Falls back to the main LLM settings when no judge-specific env vars are set.
    """
    global _judge_llm
    if _judge_llm is not None:
        return _judge_llm

    backend = os.getenv("JUDGE_LLM_BACKEND", settings.llm_backend).lower()
    model = os.getenv("JUDGE_LLM_MODEL", settings.llm_model)
    temperature = float(os.getenv("JUDGE_LLM_TEMPERATURE", "0.0"))

    if backend == "vllm":
        base_url = os.getenv("JUDGE_VLLM_BASE_URL", settings.vllm_base_url)
        logger.info("Judge LLM provider: vLLM @ %s  model=%s", base_url, model)
        _judge_llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key="EMPTY",
            openai_api_base=base_url,
        )
    else:
        logger.info("Judge LLM provider: OpenAI  model=%s", model)
        _judge_llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )

    return _judge_llm
