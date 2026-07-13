"""Pydantic input models for API requests."""

from typing import Optional
from pydantic import BaseModel, Field


class ProcessRequest(BaseModel):
    """Request body for starting a dubbing job."""

    video_path: str = Field(..., description="Path to uploaded video file")
    audio_path: Optional[str] = Field(
        default=None,
        description="Path to reference audio file (required when voice_source='clone')",
    )
    voice_source: str = Field(
        default="clone",
        description="Voice source: 'clone' (clone reference audio) or 'preset' (VieNeu built-in voice)",
    )
    preset_voice_id: Optional[str] = Field(
        default=None,
        description="VieNeu preset voice id (required when voice_source='preset')",
    )
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
