"""
TTS Module — Text-to-Speech voice cloning.

Pipeline: Vietnamese text + reference audio → cloned voice audio.
"""

from src.m2_tts.tts_client import TTSClient
from src.m2_tts.speaker_encoder import SpeakerEncoder

__all__ = ["TTSClient", "SpeakerEncoder"]
