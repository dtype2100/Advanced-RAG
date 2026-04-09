"""Task queue configuration.

Placeholder for broker / queue setup (Celery, ARQ, etc.).
"""

from __future__ import annotations


def get_queue():
    """Return the configured task queue instance (not yet implemented).

    Raises:
        NotImplementedError: Until a broker is configured.
    """
    raise NotImplementedError("Task queue is not yet configured")
