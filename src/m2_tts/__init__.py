"""
TTS Module — VieNeu-TTS v2 Turbo voice cloning engine.

Pipeline: Vietnamese text + reference audio (3–5s) → cloned voice audio (24 kHz WAV).

Engine  : pnnbao-ump/VieNeu-TTS-v2-Turbo
SDK     : pip install vieneu
Features: Zero-shot voice cloning · Bilingual (Vi/En) · 24 kHz · GPU-accelerated
"""

from src.m2_tts.tts_client import TTSClient
from src.m2_tts.speaker_encoder import SpeakerEncoder
from src.m2_tts.batch_inference import BatchInference

__all__ = ["TTSClient", "SpeakerEncoder", "BatchInference"]
