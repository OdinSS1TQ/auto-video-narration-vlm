"""Download router — GET /result/{job_id}, /result/{job_id}/srt, /result/{job_id}/report"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.schemas.response import ResultReportResponse

router = APIRouter()


def _get_completed_job(job_id: str) -> dict:
    """Retrieve a completed job or raise appropriate HTTP error."""
    from app.routers.process import _jobs

    if job_id not in _jobs:
        raise HTTPException(404, f"Job not found: {job_id}")

    job = _jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(400, f"Job not ready. Status: {job['status']}")

    return job


@router.get("/result/{job_id}")
async def download_result(job_id: str):
    """Download the processed dubbed video (MP4)."""
    job = _get_completed_job(job_id)

    result = job.get("result", {})
    output_path = result.get("output_path")

    if not output_path or not Path(output_path).exists():
        raise HTTPException(404, "Output video file not found")

    return FileResponse(
        path=output_path,
        filename=Path(output_path).name,
        media_type="video/mp4",
    )


@router.get("/result/{job_id}/srt")
async def download_srt(job_id: str):
    """Download the generated SRT subtitle file (Vietnamese)."""
    job = _get_completed_job(job_id)

    result = job.get("result", {})
    srt_path = result.get("srt_path")

    if not srt_path or not Path(srt_path).exists():
        raise HTTPException(404, "SRT file not found")

    return FileResponse(
        path=srt_path,
        filename=Path(srt_path).name,
        media_type="text/plain; charset=utf-8",
    )


@router.get("/result/{job_id}/report", response_model=ResultReportResponse)
async def get_result_report(job_id: str):
    """Get detailed result metadata for a completed job."""
    job = _get_completed_job(job_id)

    result = job.get("result", {})

    return ResultReportResponse(
        job_id=job_id,
        video_path=result.get("video_path", ""),
        reference_audio=result.get("reference_audio", ""),
        output_path=result.get("output_path", ""),
        srt_path=result.get("srt_path"),
        mode=result.get("mode", result.get("pipeline_mode", "unknown")),
        elapsed_seconds=result.get("elapsed_seconds", 0.0),
        n_segments_raw=result.get("n_segments_raw"),
        n_segments_after_classify=result.get("n_segments_after_classify"),
        n_segments_merged=result.get("n_segments_merged"),
    )
