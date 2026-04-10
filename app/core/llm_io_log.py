"""Optional logging of full LLM prompts and model outputs.

Enable with environment variable ``LOG_LLM_IO=true`` (see ``Settings.log_llm_io``).
Logs go to the ``app.llm_io`` logger at INFO — may contain PII; use only in dev/debug.
"""

from __future__ import annotations

import logging

from app.core.config import settings

_llm_io_logger = logging.getLogger("app.llm_io")


def log_llm_io(component: str, **parts: str) -> None:
    """Append a multi-section log entry when ``log_llm_io`` is enabled.

    Args:
        component: Logical step name (e.g. ``generate_answer``, ``query_rewrite``).
        **parts:   Section labels and string bodies (``system``, ``user``, ``assistant``, …).
    """
    if not settings.log_llm_io:
        return
    lines = [f"[LLM_IO] component={component}"]
    for key, val in parts.items():
        if val is not None and val != "":
            lines.append(f"--- {key} ---\n{val}")
    _llm_io_logger.info("\n".join(lines))
