"""Download router — GET /result/{job_id}"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()


@router.get("/result/{job_id}")
async def download_result(job_id: str):
    """Download the processed video."""
    from app.routers.process import _jobs

    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")

    job = _jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(400, f"Job not ready. Status: {job['status']}")

    result = job.get("result", {})
    output_path = result.get("output_path")

    if not output_path or not Path(output_path).exists():
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        path=output_path,
        filename=Path(output_path).name,
        media_type="video/mp4",
    )


@router.get("/result/{job_id}/srt")
async def download_srt(job_id: str):
    """Download the generated SRT subtitle file."""
    from app.routers.process import _jobs

    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")

    job = _jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(400, f"Job not ready. Status: {job['status']}")

    result = job.get("result", {})
    srt_path = result.get("srt_path")

    if not srt_path or not Path(srt_path).exists():
        raise HTTPException(404, "SRT file not found")

    return FileResponse(
        path=srt_path,
        filename=Path(srt_path).name,
        media_type="text/plain",
    )
