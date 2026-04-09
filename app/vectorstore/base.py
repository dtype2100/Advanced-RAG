"""Vector store interface."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Document storage and similarity search."""

    def ensure_collection(self) -> None: ...

    def add_documents(
        self, texts: list[str], metadatas: list[dict[str, Any]] | None = None
    ) -> int: ...

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]: ...

    def health_snapshot(self) -> dict[str, str]:
        """Keys: qdrant (or backend status), collection."""
