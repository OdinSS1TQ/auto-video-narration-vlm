"""Health router — GET /health"""

import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

from fastapi import APIRouter

from app.schemas.response import HealthResponse

router = APIRouter()


def _check_command(cmd: str) -> dict:
    """Check if a CLI tool is available and get its version."""
    try:
        result = subprocess.run(
            [cmd, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        version_line = result.stdout.strip().split("\n")[0] if result.stdout else ""
        # fallback to stderr (ffmpeg prints to stderr)
        if not version_line and result.stderr:
            version_line = result.stderr.strip().split("\n")[0]
        return {"available": True, "version": version_line}
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"available": False, "version": None}


def _check_gpu() -> dict:
    """Check GPU availability via torch.cuda."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory
            return {
                "available": True,
                "name": name,
                "vram_mb": round(vram / (1024 * 1024)),
            }
        return {"available": False, "name": None, "vram_mb": 0}
    except ImportError:
        return {"available": False, "name": "torch not installed", "vram_mb": 0}


def _get_dir_size_mb(path: str) -> float:
    """Get total size of a directory in MB."""
    dir_path = Path(path)
    if not dir_path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in dir_path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 1)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check — reports dependency status, GPU, disk usage, active jobs."""
    from app.routers.process import _jobs, MAX_CONCURRENT_JOBS

    tz_vn = timezone(timedelta(hours=7))

    # Check dependencies
    ffmpeg_status = _check_command("ffmpeg")
    rubberband_status = _check_command("rubberband")
    gpu_status = _check_gpu()

    # Count active jobs
    active_jobs = sum(
        1 for j in _jobs.values()
        if j["status"] in ("queued", "processing")
    )

    # Disk usage
    disk_usage = {
        "uploads_mb": _get_dir_size_mb("./data/uploads"),
        "outputs_mb": _get_dir_size_mb("./data/outputs"),
    }

    # Determine overall health
    all_ok = ffmpeg_status["available"] and rubberband_status["available"]
    if all_ok:
        overall = "healthy"
    elif ffmpeg_status["available"]:
        overall = "degraded"
    else:
        overall = "unhealthy"

    return HealthResponse(
        status=overall,
        version="0.1.0",
        timestamp=datetime.now(tz_vn).isoformat(),
        dependencies={
            "ffmpeg": ffmpeg_status,
            "rubberband": rubberband_status,
            "gpu": gpu_status,
        },
        active_jobs=active_jobs,
        max_concurrent_jobs=MAX_CONCURRENT_JOBS,
        disk_usage=disk_usage,
    )
