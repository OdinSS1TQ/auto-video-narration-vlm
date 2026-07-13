"""Config router — GET /config"""

from fastapi import APIRouter

from app.schemas.response import ConfigResponse

router = APIRouter()


@router.get("/config", response_model=ConfigResponse)
async def get_config():
    """Return current pipeline configuration (sensitive data masked)."""
    from src.m4_pipeline.config import PipelineConfig

    config = PipelineConfig()

    return ConfigResponse(
        vlm={
            "mode": config.vlm_mode,
            "model_name": config.vlm_model_name,
            "temperature": config.vlm_temperature,
            "max_tokens": config.vlm_max_tokens,
        },
        tts={
            "engine": config.tts_engine,
            "sample_rate": config.tts_sample_rate,
            "vieneu_mode": config.tts_vieneu_mode,
            "backbone_device": config.tts_backbone_device,
            "codec_device": config.tts_codec_device,
        },
        pipeline={
            "mode": config.pipeline_mode,
            "chunk_duration_sec": config.chunk_duration,
            "scene_threshold": config.scene_threshold,
            "ssim_threshold": config.ssim_threshold,
        },
        sync={
            "max_speedup": config.m3_max_speedup,
            "min_gap_sec": config.m3_min_gap_sec,
        },
        limits={
            "max_video_size_mb": 200,
            "max_audio_size_mb": 10,
            "max_video_duration_min": 5,
            "max_concurrent_jobs": 5,
        },
        output={
            "dir": config.output_dir,
            "format": "mp4",
        },
    )
