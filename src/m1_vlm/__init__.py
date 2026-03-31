"""
VLM Module — Video Language Model for subtitle generation.

Pipeline: Video → Scene Detection → Frame Extraction → VLM (Extract + Translate) → SRT
The VLM handles text extraction and translation directly — no separate OCR needed.
GLM-OCR is available as an optional supplementary tool.
"""

from src.m1_vlm.scene_detector import SceneDetector
from src.m1_vlm.frame_extractor import FrameExtractor
from src.m1_vlm.vlm_client import VLMClient
from src.m1_vlm.srt_builder import SRTBuilder

__all__ = [
    "SceneDetector",
    "FrameExtractor",
    "VLMClient",
    "SRTBuilder",
]
