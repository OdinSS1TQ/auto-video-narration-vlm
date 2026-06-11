"""Upload router — POST /upload/video, /upload/audio"""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.response import UploadResponse

router = APIRouter()

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Size limits (bytes)
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB
MAX_AUDIO_SIZE = 10 * 1024 * 1024   # 10 MB


async def _read_upload(file: UploadFile, max_size: int) -> bytes:
    """Read uploaded file content, enforcing size limit."""
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(
            413,
            f"File too large: {len(content) / (1024 * 1024):.1f} MB. "
            f"Maximum: {max_size / (1024 * 1024):.0f} MB"
        )
    return content


@router.post("/upload/video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing.

    Accepts: .mp4, .avi, .mkv, .webm, .mov — max 200 MB (~5 min video).
    """
    allowed_ext = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {allowed_ext}")

    content = await _read_upload(file, MAX_VIDEO_SIZE)

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        path=str(save_path),
        size_mb=save_path.stat().st_size / (1024 * 1024),
    )


@router.post("/upload/audio", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload a reference audio file for voice cloning.

    Accepts: .wav, .mp3, .flac, .ogg, .m4a — max 10 MB.
    Recommended: 3-10 seconds, mono/stereo, ≥16kHz.
    """
    allowed_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {allowed_ext}")

    content = await _read_upload(file, MAX_AUDIO_SIZE)

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        f.write(content)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        path=str(save_path),
        size_mb=save_path.stat().st_size / (1024 * 1024),
    )
