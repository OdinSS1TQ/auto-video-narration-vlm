"""TTS router — GET /tts/voices (list VieNeu preset voices, no model load)."""

import json
from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter
from loguru import logger

router = APIRouter()

DEFAULT_BACKBONE_REPO = "pnnbao-ump/VieNeu-TTS-v2-Turbo-GGUF"


def _parse_voices_json(path: Path) -> Dict[str, Any]:
    """Parse a VieNeu voices.json into {voices:[{id,label}], default}.

    The preset key IS the voice id; label = description if present else the id.
    Returns empty structure on any malformed input.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"Could not read voices.json at {path}: {exc}")
        return {"voices": [], "default": None}

    presets = data.get("presets") or {}
    voices = []
    for key, val in presets.items():
        label = key
        if isinstance(val, dict) and val.get("description"):
            label = val["description"]
        voices.append({"id": key, "label": label})
    return {"voices": voices, "default": data.get("default_voice")}


def _locate_voices_file() -> Path | None:
    """Resolve the cached voices.json path without contacting the network."""
    from src.m4_pipeline.config import PipelineConfig

    repo_id = PipelineConfig().tts_backbone_repo or DEFAULT_BACKBONE_REPO
    try:
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(
            repo_id=repo_id,
            filename="voices.json",
            repo_type="model",
            local_files_only=True,
        ))
    except Exception as exc:
        logger.warning(f"voices.json not in cache for repo '{repo_id}': {exc}")
        return None


@router.get("/tts/voices")
async def list_preset_voices():
    """List VieNeu preset voices for the UI. Empty list if cache/parse fails."""
    path = _locate_voices_file()
    if path is None:
        return {"voices": [], "default": None}
    return _parse_voices_json(path)
