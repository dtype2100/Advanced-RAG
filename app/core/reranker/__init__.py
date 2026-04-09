"""Reranker provider abstraction.

Switch providers by setting ``RERANKER_PROVIDER`` env var:
  - ``none`` (default) – no reranking
  - ``flashrank`` – FlashRank cross-encoder reranking
"""

from app.core.reranker.base import BaseReranker
from app.core.reranker.factory import get_reranker

__all__ = ["BaseReranker", "get_reranker"]
