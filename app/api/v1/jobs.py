"""Background job status API (ARQ)."""

from __future__ import annotations

import logging

from arq.jobs import Job, JobStatus
from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.queue.pool import get_arq_pool
from app.schemas.document import JobStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the status and result of an async ingest job."""
    if not settings.redis_url:
        raise HTTPException(status_code=503, detail="Job status requires REDIS_URL.")

    pool = await get_arq_pool()
    job = Job(job_id, pool, _queue_name=settings.arq_queue_name)
    status = await job.status()

    if status == JobStatus.not_found:
        raise HTTPException(status_code=404, detail="Job not found")

    if status in (JobStatus.queued, JobStatus.deferred, JobStatus.in_progress):
        return JobStatusResponse(job_id=job_id, status=status.value)

    info = await job.result_info()
    if info is None:
        return JobStatusResponse(job_id=job_id, status=status.value)

    if info.success:
        result = info.result or {}
        return JobStatusResponse(
            job_id=job_id,
            status="complete",
            count=result.get("count"),
            message=result.get("message"),
        )

    err = str(info.result) if info.result else "Job failed"
    logger.warning("Job %s failed: %s", job_id, err)
    return JobStatusResponse(job_id=job_id, status="failed", error=err)
