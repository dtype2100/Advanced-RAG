"""ARQ worker settings — referenced by ``arq app.workers.settings.WorkerSettings``."""

from __future__ import annotations

from arq.connections import RedisSettings

from app.core.config import settings
from app.workers.jobs import ingest_documents_job


def _redis_settings() -> RedisSettings:
    if not settings.redis_url:
        raise RuntimeError("REDIS_URL must be set for the ARQ worker.")
    return RedisSettings.from_dsn(settings.redis_url)


class WorkerSettings:
    """ARQ worker configuration."""

    functions = [ingest_documents_job]
    redis_settings = _redis_settings()
    queue_name = settings.arq_queue_name
    max_jobs = 10
    job_timeout = 3600
    keep_result = 86400
