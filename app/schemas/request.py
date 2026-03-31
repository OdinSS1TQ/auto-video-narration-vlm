"""Pydantic input models for API requests."""

from typing import Optional
from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Request body for starting a dubbing job."""

    video_path: str = Field(..., description="Path to uploaded video file")
    audio_path: str = Field(..., description="Path to reference audio file")
    vlm_mode: str = Field(
        default="api",
        description="VLM mode: 'api' (Gemini) or 'local' (Qwen2-VL)"
    )
    tts_engine: str = Field(
        default="f5-tts",
        description="TTS engine: 'f5-tts' or 'vixtts'"
    )
    source_lang: str = Field(default="English", description="Source language")
    target_lang: str = Field(default="Vietnamese", description="Target language")
    keep_original_audio: bool = Field(
        default=False,
        description="Keep original audio at low volume"
    )
