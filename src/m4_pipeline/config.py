"""
Config — Load pipeline settings from .env and YAML config files.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv


class PipelineConfig:
    """Pipeline configuration loaded from environment and YAML files."""

    def __init__(self, env_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Args:
            env_path: Path to .env file (default: project root .env).
            config_path: Path to YAML config file (default: configs/default.yaml).
        """
        # Load .env
        load_dotenv(env_path or ".env")

        # Load YAML config if provided
        self._yaml_config = {}
        if config_path and Path(config_path).exists():
            with open(config_path, "r", encoding="utf-8") as f:
                self._yaml_config = yaml.safe_load(f) or {}

    # === API Keys ===
    @property
    def gemini_api_key(self) -> Optional[str]:
        return os.getenv("GEMINI_API_KEY")

    # === VLM Settings ===
    @property
    def vlm_mode(self) -> str:
        return os.getenv("VLM_MODE", "api")

    @property
    def vlm_model_name(self) -> str:
        return os.getenv("VLM_MODEL_NAME", "gemini-1.5-flash")

    @property
    def vlm_temperature(self) -> float:
        return float(os.getenv("VLM_TEMPERATURE", "0.2"))

    @property
    def vlm_max_tokens(self) -> int:
        return int(os.getenv("VLM_MAX_TOKENS", "4096"))

    # === Local Model Paths ===
    @property
    def qwen_model_path(self) -> str:
        return os.getenv("QWEN_MODEL_PATH", "./models/qwen3.5-2b")

    @property
    def glm_ocr_model_path(self) -> str:
        return os.getenv("GLM_OCR_MODEL_PATH", "./models/glm-ocr")

    @property
    def tts_model_path(self) -> str:
        return os.getenv("TTS_MODEL_PATH", "./models/vitts")

    # === TTS Settings ===
    @property
    def tts_engine(self) -> str:
        return os.getenv("TTS_ENGINE", "vieneu")

    @property
    def tts_sample_rate(self) -> int:
        return int(os.getenv("TTS_SAMPLE_RATE", "24000"))  # VieNeu native: 24 kHz

    @property
    def tts_speed(self) -> float:
        return float(os.getenv("TTS_SPEED", "1.0"))

    # --- VieNeu-TTS v2 Turbo specific ---
    @property
    def tts_backbone_repo(self) -> Optional[str]:
        """HuggingFace repo for VieNeu backbone LLM. None = let vieneu SDK
        pick its internal defaults (which are version-aware and known-good).
        Override only if you have a custom backbone."""
        repo = os.getenv("TTS_BACKBONE_REPO", "")
        return repo if repo else None

    @property
    def tts_backbone_device(self) -> str:
        """Device for backbone LLM inference ('cuda' or 'cpu')."""
        return os.getenv("TTS_BACKBONE_DEVICE", "cuda")

    @property
    def tts_codec_device(self) -> str:
        """Device for NeuCodec decoder ('cuda' or 'cpu'). CPU is fine — codec
        is small ONNX, GPU saves only ~50ms but contends with backbone VRAM."""
        return os.getenv("TTS_CODEC_DEVICE", "cpu")

    @property
    def tts_vieneu_mode(self) -> str:
        """Vieneu() factory mode: 'turbo' (default, GGUF+ONNX) | 'standard'
        (PyTorch, needs neucodec pkg) | 'fast' | 'remote'."""
        return os.getenv("TTS_VIENEU_MODE", "turbo")

    @property
    def tts_hf_token(self) -> Optional[str]:
        """HuggingFace token for private/gated model repos (None if unset)."""
        token = os.getenv("TTS_HF_TOKEN", "")
        return token if token else None

    # === Processing ===
    @property
    def chunk_duration(self) -> float:
        return float(os.getenv("CHUNK_DURATION_SEC", "45"))

    @property
    def max_concurrent_chunks(self) -> int:
        return int(os.getenv("MAX_CONCURRENT_CHUNKS", "4"))

    @property
    def scene_threshold(self) -> float:
        return float(os.getenv("SCENE_THRESHOLD", "27.0"))

    @property
    def chunk_overlap(self) -> float:
        """Overlap duration (seconds) between consecutive chunks.
        Prevents context loss at chunk boundaries."""
        return float(os.getenv("CHUNK_OVERLAP_SEC", "5"))

    @property
    def ssim_threshold(self) -> float:
        """SSIM threshold for frame deduplication.
        Frames with SSIM > threshold are considered duplicates.
        0.85 = balanced for tutorial videos."""
        return float(os.getenv("SSIM_THRESHOLD", "0.85"))

    # === Data Paths ===
    @property
    def output_dir(self) -> str:
        return os.getenv("OUTPUT_DIR", "./data/outputs")

    @property
    def processed_dir(self) -> str:
        return os.getenv("PROCESSED_DIR", "./data/processed")

    # === Server ===
    @property
    def host(self) -> str:
        return os.getenv("HOST", "0.0.0.0")

    @property
    def port(self) -> int:
        return int(os.getenv("PORT", "8000"))

    @property
    def debug(self) -> bool:
        return os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

    def get_yaml(self, key: str, default=None):
        """Get a value from the YAML config."""
        keys = key.split(".")
        value = self._yaml_config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
