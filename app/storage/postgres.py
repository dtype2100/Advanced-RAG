"""PostgreSQL connection helper for parent document / metadata storage.

Placeholder for future Postgres integration (docstore, chat history, etc.).
"""

from __future__ import annotations


def get_postgres_engine():
    """Return a SQLAlchemy async engine (not yet implemented).

    Raises:
        NotImplementedError: Until PostgreSQL support is wired up.
    """
    raise NotImplementedError("PostgreSQL storage is not yet configured")
