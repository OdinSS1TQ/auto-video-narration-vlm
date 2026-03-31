"""
Sync Module — Audio synchronization and video merging.

Handles time-stretching, alignment, and FFmpeg rendering.
"""

from src.m3_sync.time_stretcher import TimeStretcher
from src.m3_sync.audio_aligner import AudioAligner
from src.m3_sync.ffmpeg_renderer import FFmpegRenderer

__all__ = ["TimeStretcher", "AudioAligner", "FFmpegRenderer"]
