"""
Sync Accuracy — Measure audio onset delay vs subtitle timestamps.
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger


class SyncAccuracy:
    """Evaluate synchronization accuracy between audio and subtitles."""

    def __init__(self):
        pass

    def detect_onset(self, audio_path: str | Path) -> float:
        """
        Detect the onset time of speech in an audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Onset time in seconds.
        """
        import librosa

        y, sr = librosa.load(str(audio_path), sr=None)
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units="time")

        if len(onset_frames) > 0:
            return float(onset_frames[0])
        return 0.0

    def compute_delay(
        self,
        expected_start: float,
        audio_path: str | Path,
    ) -> float:
        """
        Compute delay between expected start time and actual audio onset.

        Args:
            expected_start: Expected start time from subtitle.
            audio_path: Path to audio chunk.

        Returns:
            Absolute delay in seconds.
        """
        actual_onset = self.detect_onset(audio_path)
        delay = abs(actual_onset - expected_start)
        return delay

    def evaluate_segments(
        self,
        segments: List[Dict[str, Any]],
    ) -> dict:
        """
        Evaluate sync accuracy for all segments.

        Args:
            segments: List of segments with 'start_sec' and 'aligned_audio_path'.

        Returns:
            Dict with mean, std, max absolute delay.
        """
        delays = []

        for segment in segments:
            audio_path = segment.get("aligned_audio_path")
            start_sec = segment.get("start_sec", 0)

            if not audio_path or not Path(audio_path).exists():
                continue

            try:
                delay = self.compute_delay(0.0, audio_path)
                delays.append(delay)
            except Exception as e:
                logger.warning(f"Failed onset detection: {e}")

        if not delays:
            return {"mean_delay": 0.0, "count": 0}

        result = {
            "mean_delay": float(np.mean(delays)),
            "std_delay": float(np.std(delays)),
            "max_delay": float(np.max(delays)),
            "count": len(delays),
        }

        logger.info(
            f"Sync accuracy: mean delay={result['mean_delay']:.4f}s "
            f"(n={result['count']})"
        )
        return result
