"""Process router — POST /process → job id"""

import asyncio
import uuid
from typing import Dict

from fastapi import APIRouter, BackgroundTasks

from app.schemas.request import ProcessRequest
from app.schemas.response import ProcessResponse

router = APIRouter()

# In-memory job storage (replace with Redis/DB for production)
_jobs: Dict[str, dict] = {}


async def _run_pipeline(job_id: str, request: ProcessRequest):
    """Background task to run the dubbing pipeline."""
    _jobs[job_id]["status"] = "processing"

    try:
        from src.m4_pipeline.runner import PipelineRunner
        from src.m4_pipeline.config import PipelineConfig

        config = PipelineConfig()
        runner = PipelineRunner(config)

        def progress_callback(step: str, current: int, total: int):
            _jobs[job_id]["progress"] = {
                "step": step,
                "current": current,
                "total": total,
                "percent": int((current / total) * 100),
            }

        runner.set_progress_callback(progress_callback)

        result = await runner.run(
            video_path=request.video_path,
            reference_audio_path=request.audio_path,
        )

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@router.post("/process", response_model=ProcessResponse)
async def start_processing(
    request: ProcessRequest,
    background_tasks: BackgroundTasks,
):
    """Start a new dubbing job."""
    job_id = str(uuid.uuid4())

    _jobs[job_id] = {
        "status": "queued",
        "progress": {"step": "queued", "current": 0, "total": 8, "percent": 0},
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_run_pipeline, job_id, request)

    return ProcessResponse(
        job_id=job_id,
        status="queued",
        message="Dubbing job queued. Use GET /api/status/{job_id} to track progress.",
    )
