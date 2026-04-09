"""LLM provider abstraction.

Switch providers by setting ``LLM_BACKEND`` env var:
  - ``vllm`` (default) – local vLLM server (OpenAI-compatible)
  - ``openai`` – OpenAI API
"""

from app.core.llm.factory import get_llm

__all__ = ["get_llm"]
