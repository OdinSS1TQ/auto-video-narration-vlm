"""Process router — POST /process, DELETE /process/{job_id}"""

import asyncio
import contextvars
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.schemas.request import ProcessRequest
from app.schemas.response import ProcessResponse

# ContextVar to capture logs per job in async tasks
current_job_id = contextvars.ContextVar("current_job_id", default=None)

router = APIRouter()

import json
import sys
import threading

# In-memory job storage and thread safety lock
_jobs: Dict[str, dict] = {}
_jobs_lock = threading.Lock()
JOBS_FILE = Path("data/jobs.json")

# Concurrency limit (1 job at a time per machine)
MAX_CONCURRENT_JOBS = 1

def load_jobs():
    global _jobs
    if JOBS_FILE.exists():
        try:
            with open(JOBS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Mark interrupted/stale jobs as failed
                for job_id, job in data.items():
                    if job.get("status") in ("queued", "processing"):
                        job["status"] = "failed"
                        job["error"] = "Server restarted or crashed during job execution."
                with _jobs_lock:
                    _jobs.update(data)
            print("Loaded jobs from disk successfully.")
        except Exception as e:
            print(f"Failed to load jobs from disk: {e}", file=sys.stderr)

def save_jobs():
    with _jobs_lock:
        try:
            JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
            temp_file = JOBS_FILE.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(_jobs, f, indent=2, ensure_ascii=False)
            temp_file.replace(JOBS_FILE)
        except Exception as e:
            print(f"Failed to save jobs to disk: {e}", file=sys.stderr)

# Custom loguru sink to forward logs to job objects
def job_log_sink(message):
    if getattr(job_log_sink, "_in_sink", False):
        return
    job_log_sink._in_sink = True
    try:
        record = message.record
        job_id = current_job_id.get()
        if job_id:
            with _jobs_lock:
                if job_id in _jobs:
                    time_str = record["time"].strftime("%H:%M:%S")
                    level = "INF"
                    if record["level"].name == "WARNING":
                        level = "WRN"
                    elif record["level"].name == "ERROR":
                        level = "ERR"
                    elif record["level"].name == "DEBUG":
                        level = "DBG"
                    
                    log_line = f"[{time_str}] {level}: {record['message']}"
                    
                    if "logs" not in _jobs[job_id]:
                        _jobs[job_id]["logs"] = []
                    _jobs[job_id]["logs"].append(log_line)
            save_jobs()
    finally:
        job_log_sink._in_sink = False

# Add custom sink to loguru
logger.add(job_log_sink, level="INFO")

# Vietnamese step names for progress display
STEP_NAMES_VI = {
    # OCR-mode steps
    "caption_ocr_timeline": "Trích xuất caption bằng OCR",
    "vlm_classify_narration": "Phân loại lời thoại",
    "vlm_global_summary": "Tóm tắt tổng quan video",
    "vlm_translate_segments": "Dịch thuật bằng VLM",
    # VLM-mode steps
    "scene_detection": "Phát hiện cảnh",
    "frame_extraction": "Trích xuất khung hình",
    "ocr_extraction": "Trích xuất OCR",
    "vlm_translation": "Dịch thuật bằng VLM",
    # Shared steps
    "srt_generation": "Tạo file phụ đề SRT",
    "voice_cloning": "Nhân bản giọng nói",
    "audio_alignment": "Căn chỉnh audio",
    "video_rendering": "Render video cuối cùng",
    # Meta
    "queued": "Đang chờ xử lý",
}


def _count_active_jobs() -> int:
    """Count jobs that are queued or processing."""
    with _jobs_lock:
        return sum(
            1 for j in _jobs.values()
            if j["status"] in ("queued", "processing")
        )


def _get_step_name_vi(step: str) -> str:
    """Get Vietnamese name for a pipeline step."""
    return STEP_NAMES_VI.get(step, step)


async def _run_pipeline(job_id: str, request: ProcessRequest):
    """Background task to run the dubbing pipeline."""
    current_job_id.set(job_id)
    with _jobs_lock:
        _jobs[job_id]["status"] = "processing"
    save_jobs()

    try:
        from src.m4_pipeline.runner import PipelineRunner
        from src.m4_pipeline.config import PipelineConfig

        config = PipelineConfig()

        # Override config with request values
        import os
        os.environ["PIPELINE_MODE"] = request.pipeline_mode
        os.environ["VLM_MODE"] = request.vlm_mode
        os.environ["TTS_ENGINE"] = request.tts_engine

        runner = PipelineRunner(config)

        def progress_callback(step: str, current: int, total: int):
            with _jobs_lock:
                _jobs[job_id]["progress"] = {
                    "step": step,
                    "step_name_vi": _get_step_name_vi(step),
                    "current": current,
                    "total": total,
                    "percent": int((current / total) * 100),
                }
            save_jobs()

        runner.set_progress_callback(progress_callback)

        result = await runner.run(
            video_path=request.video_path,
            reference_audio_path=request.audio_path,
            voice_source=request.voice_source,
            preset_voice_id=request.preset_voice_id,
        )

        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
        save_jobs()

    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
        save_jobs()


@router.post("/process", response_model=ProcessResponse)
async def start_processing(request: ProcessRequest):
    """Start a new dubbing job.

    Runs the 8-step pipeline in the background.
    Default: pipeline_mode='ocr', vlm_mode='local'.
    Max 1 concurrent job per machine.
    """
    # Validate vlm_mode
    if request.vlm_mode not in ("local", "api"):
        raise HTTPException(422, f"Invalid vlm_mode: {request.vlm_mode}. Must be 'local' or 'api'.")

    # Validate pipeline_mode
    if request.pipeline_mode not in ("ocr", "vlm"):
        raise HTTPException(422, f"Invalid pipeline_mode: {request.pipeline_mode}. Must be 'ocr' or 'vlm'.")

    # Validate voice source
    if request.voice_source not in ("clone", "preset"):
        raise HTTPException(422, f"Invalid voice_source: {request.voice_source}. Must be 'clone' or 'preset'.")

    if request.voice_source == "preset":
        if not request.preset_voice_id:
            raise HTTPException(422, "preset_voice_id is required when voice_source='preset'.")
    else:  # clone
        if not request.audio_path:
            raise HTTPException(422, "audio_path is required when voice_source='clone'.")
        if not Path(request.audio_path).exists():
            raise HTTPException(404, f"Audio file not found: {request.audio_path}")

    # Validate video existence
    if not Path(request.video_path).exists():
        raise HTTPException(404, f"Video file not found: {request.video_path}")

    # Rate limiting
    if _count_active_jobs() >= MAX_CONCURRENT_JOBS:
        raise HTTPException(
            429,
            f"Too many active jobs ({MAX_CONCURRENT_JOBS} max). "
            f"Wait for a job to finish or cancel one."
        )

    job_id = str(uuid.uuid4())
    tz_vn = timezone(timedelta(hours=7))

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": {
                "step": "queued",
                "step_name_vi": _get_step_name_vi("queued"),
                "current": 0,
                "total": 8,
                "percent": 0,
            },
            "result": None,
            "error": None,
            "created_at": datetime.now(tz_vn).isoformat(),
            "logs": [
                f"[{datetime.now(tz_vn).strftime('%H:%M:%S')}] INF: Initializing pipeline for {job_id}",
                f"[{datetime.now(tz_vn).strftime('%H:%M:%S')}] INF: Source Video: {request.video_path}",
                f"[{datetime.now(tz_vn).strftime('%H:%M:%S')}] INF: Reference Audio: {request.audio_path}",
                f"[{datetime.now(tz_vn).strftime('%H:%M:%S')}] INF: Pipeline Mode: {request.pipeline_mode.upper()} | VLM: {request.vlm_mode.upper()} | TTS: {request.tts_engine.upper()}",
                f"[{datetime.now(tz_vn).strftime('%H:%M:%S')}] INF: Queued in processing pool."
            ],
            "video_path": request.video_path,
            "audio_path": request.audio_path,
            "pipeline_mode": request.pipeline_mode,
            "vlm_mode": request.vlm_mode,
            "keep_original_audio": request.keep_original_audio,
            "tts_engine": request.tts_engine,
            "voice_source": request.voice_source,
            "preset_voice_id": request.preset_voice_id,
        }
    save_jobs()

    # Launch background task via a daemon thread so it runs in parallel to the main event loop
    def worker_thread_target(jid: str, req: ProcessRequest):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_pipeline(jid, req))
        finally:
            loop.close()

    threading.Thread(
        target=worker_thread_target,
        args=(job_id, request),
        daemon=True,
    ).start()

    return ProcessResponse(
        job_id=job_id,
        status="queued",
        message="Dubbing job queued. Use GET /api/status/{job_id} to track progress.",
    )


@router.delete("/process/{job_id}", response_model=ProcessResponse)
async def cancel_job(job_id: str):
    """Cancel a queued or processing job."""
    with _jobs_lock:
        if job_id not in _jobs:
            raise HTTPException(404, f"Job not found: {job_id}")

        job = _jobs[job_id]

        if job["status"] in ("completed", "failed", "cancelled"):
            raise HTTPException(
                400,
                f"Cannot cancel job with status '{job['status']}'. "
                f"Only 'queued' or 'processing' jobs can be cancelled."
            )

        job["status"] = "cancelled"
        job["error"] = "Job cancelled by user."
    save_jobs()

    return ProcessResponse(
        job_id=job_id,
        status="cancelled",
        message="Job cancelled successfully.",
    )
