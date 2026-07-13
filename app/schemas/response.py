"""Pydantic output models for API responses."""

from typing import Any, Dict, List, Optional
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


class ProgressInfo(BaseModel):
    """Progress details for a running job."""

    step: str
    step_name_vi: str
    current: int
    total: int
    percent: int


class StatusResponse(BaseModel):
    """Response for job status query."""

    job_id: str
    status: str
    progress: ProgressInfo
    created_at: str
    error: Optional[str] = None
    logs: List[str] = []
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    pipeline_mode: Optional[str] = None
    vlm_mode: Optional[str] = None
    keep_original_audio: Optional[bool] = None
    tts_engine: Optional[str] = None


class JobSummary(BaseModel):
    """Summary of a single job for listing."""

    job_id: str
    status: str
    progress: ProgressInfo
    created_at: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None


class JobListResponse(BaseModel):
    """Response for listing all jobs."""

    jobs: List[JobSummary]
    total: int
    limit: int
    offset: int


class ResultReportResponse(BaseModel):
    """Response with detailed processing result metadata."""

    job_id: str
    video_path: str
    reference_audio: str
    output_path: str
    srt_path: Optional[str] = None
    mode: str
    elapsed_seconds: float
    n_segments_raw: Optional[int] = None
    n_segments_after_classify: Optional[int] = None
    n_segments_merged: Optional[int] = None


class HealthResponse(BaseModel):
    """Response for health check."""

    status: str
    version: str
    timestamp: str
    dependencies: Dict[str, Any]
    active_jobs: int
    max_concurrent_jobs: int
    disk_usage: Dict[str, float]


class ConfigResponse(BaseModel):
    """Response for config query (sensitive data masked)."""

    vlm: Dict[str, Any]
    tts: Dict[str, Any]
    pipeline: Dict[str, Any]
    sync: Dict[str, Any]
    limits: Dict[str, Any]
    output: Dict[str, Any]
