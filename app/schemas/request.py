"""Pydantic input models for API requests."""

from typing import Optional
from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Request body for starting a dubbing job."""

    video_path: str = Field(..., description="Path to uploaded video file")
    audio_path: str = Field(..., description="Path to reference audio file")
    vlm_mode: str = Field(
        default="local",
        description="VLM mode: 'local' (Qwen3.5-2B default) or 'api' (Gemini)"
    )
    tts_engine: str = Field(
        default="vieneu",
        description="TTS engine: 'vieneu' (VieNeu-TTS v2 Turbo)"
    )
    pipeline_mode: str = Field(
        default="ocr",
        description="Pipeline mode: 'ocr' (OCR-driven, default) or 'vlm' (VLM-driven)"
    )
    source_lang: str = Field(default="English", description="Source language")
    target_lang: str = Field(default="Vietnamese", description="Target language")
    keep_original_audio: bool = Field(
        default=False,
        description="Keep original audio at low volume"
    )
