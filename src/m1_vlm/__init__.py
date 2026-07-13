"""
VLM Module — Video Language Model for subtitle generation.

Pipeline: Video → Scene Detection → Adaptive Frame Extraction → SSIM Dedup → VLM → SRT

Architecture (AI Technical Leader):
    Module 1 (m1_vlm) is the core intelligence layer. It transforms raw video
    into structured subtitle data through a multi-stage pipeline:

    1. SceneDetector: Content-aware video segmentation
    2. FrameExtractor: Adaptive frame sampling at scene boundaries
    3. FrameDeduplicator: SSIM-based redundancy removal
    4. ContextWindow: Hierarchical context (global + sliding window)
    5. PromptChain: Structured prompts with global context injection
    6. VLMClient: Multi-backend inference (Gemini API / Local Qwen)
    7. SubtitleValidator + SRTBuilder: Output validation and formatting
"""

from src.m1_vlm.scene_detector import SceneDetector
from src.m1_vlm.frame_extractor import FrameExtractor
from src.m1_vlm.frame_dedup import FrameDeduplicator
from src.m1_vlm.vlm_client import VLMClient
from src.m1_vlm.srt_builder import SRTBuilder
from src.m1_vlm.context_window import ContextWindow
from src.m1_vlm.prompt_chain import PromptChain

__all__ = [
    "SceneDetector",
    "FrameExtractor",
    "FrameDeduplicator",
    "VLMClient",
    "SRTBuilder",
    "ContextWindow",
    "PromptChain",
]
