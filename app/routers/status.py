"""Status router — GET /status/{job_id}"""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    from app.routers.process import _jobs

    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")

    job = _jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "error": job.get("error"),
    }
