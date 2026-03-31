"""Upload router — POST /upload/video, /upload/audio"""

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.response import UploadResponse

router = APIRouter()

UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload/video", response_model=UploadResponse)
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    allowed_ext = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {allowed_ext}")

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        path=str(save_path),
        size_mb=save_path.stat().st_size / (1024 * 1024),
    )


@router.post("/upload/audio", response_model=UploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload a reference audio file for voice cloning."""
    allowed_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_ext:
        raise HTTPException(400, f"Unsupported format: {ext}. Allowed: {allowed_ext}")

    file_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    return UploadResponse(
        file_id=file_id,
        filename=file.filename,
        path=str(save_path),
        size_mb=save_path.stat().st_size / (1024 * 1024),
    )
