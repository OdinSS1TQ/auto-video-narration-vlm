"""Status router — GET /status/{job_id}, GET /jobs"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.schemas.response import StatusResponse, ProgressInfo, JobListResponse, JobSummary

router = APIRouter()


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """Get the status and progress of a processing job."""
    from app.routers.process import _jobs, _jobs_lock

    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(404, f"Job not found: {job_id}")

        job = _jobs[job_id]
        progress_data = job["progress"]

        return StatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=ProgressInfo(
                step=progress_data.get("step", "unknown"),
                step_name_vi=progress_data.get("step_name_vi", progress_data.get("step", "unknown")),
                current=progress_data.get("current", 0),
                total=progress_data.get("total", 8),
                percent=progress_data.get("percent", 0),
            ),
            created_at=job.get("created_at", ""),
            error=job.get("error"),
            logs=job.get("logs", []),
            video_path=job.get("video_path"),
            audio_path=job.get("audio_path"),
            pipeline_mode=job.get("pipeline_mode"),
            vlm_mode=job.get("vlm_mode"),
            keep_original_audio=job.get("keep_original_audio"),
            tts_engine=job.get("tts_engine"),
        )


@router.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status: queued, processing, completed, failed, cancelled"),
    limit: int = Query(20, ge=1, le=100, description="Max results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
):
    """List all jobs with optional status filter and pagination."""
    from app.routers.process import _jobs, _jobs_lock

    # Filter by status if provided
    valid_statuses = {"queued", "processing", "completed", "failed", "cancelled"}
    if status and status not in valid_statuses:
        raise HTTPException(422, f"Invalid status filter: {status}. Valid: {valid_statuses}")

    all_jobs = []
    with _jobs_lock:
        for jid, job in _jobs.items():
            if status and job["status"] != status:
                continue
            progress_data = job["progress"]
            all_jobs.append(JobSummary(
                job_id=jid,
                status=job["status"],
                progress=ProgressInfo(
                    step=progress_data.get("step", "unknown"),
                    step_name_vi=progress_data.get("step_name_vi", progress_data.get("step", "unknown")),
                    current=progress_data.get("current", 0),
                    total=progress_data.get("total", 8),
                    percent=progress_data.get("percent", 0),
                ),
                created_at=job.get("created_at", ""),
                video_path=job.get("video_path"),
                audio_path=job.get("audio_path"),
            ))

    # Sort by created_at descending (newest first)
    all_jobs.sort(key=lambda j: j.created_at, reverse=True)

    total = len(all_jobs)
    paginated = all_jobs[offset:offset + limit]

    return JobListResponse(
        jobs=paginated,
        total=total,
        limit=limit,
        offset=offset,
    )
