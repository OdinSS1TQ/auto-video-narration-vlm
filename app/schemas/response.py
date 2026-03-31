"""Pydantic output models for API responses."""

from typing import Any, Dict, Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Response for file upload."""

    file_id: str
    filename: str
    path: str
    size_mb: float


class ProcessResponse(BaseModel):
    """Response for starting a processing job."""

    job_id: str
    status: str
    message: str


class StatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: str
    progress: Dict[str, Any]
    error: Optional[str] = None


class ResultResponse(BaseModel):
    """Response with processing results."""

    job_id: str
    output_path: str
    srt_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    elapsed_seconds: float
