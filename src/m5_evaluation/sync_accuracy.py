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

            if not audio_path or not Path(audio_path).exists():
                continue

            try:
                expected = float(segment.get("srt_start_sec", segment.get("start_sec", 0.0)))
                delay = self.compute_delay(expected, audio_path)
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

    def evaluate_against_merged_audio(
        self,
        srt_entries: List[Dict[str, Any]],
        merged_audio_path: str | Path,
        window_sec: float = 1.0,
    ) -> dict:
        """For each SRT entry, detect onset within a window around srt_start
        in the merged audio and report signed delay (positive = audio late)."""
        import librosa

        y, sr = librosa.load(str(merged_audio_path), sr=None)
        total_dur = len(y) / sr
        delays = []
        missed = 0

        for entry in srt_entries:
            start = self._timestamp_to_seconds(entry["start_time"])
            end = self._timestamp_to_seconds(entry["end_time"])
            win_start = max(0, start - window_sec)
            win_end = min(total_dur, end + window_sec)
            seg = y[int(win_start * sr):int(win_end * sr)]

            if len(seg) == 0:
                missed += 1
                continue

            onsets = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
            if len(onsets) == 0:
                missed += 1
                continue

            actual = float(onsets[0]) + win_start
            delays.append(actual - start)

        if not delays:
            return {"mean_delay": 0.0, "p50": 0.0, "p95": 0.0, "n_missed": missed, "count": 0}

        arr = np.array(delays)
        return {
            "mean_delay": float(arr.mean()),
            "p50": float(np.percentile(np.abs(arr), 50)),
            "p95": float(np.percentile(np.abs(arr), 95)),
            "n_missed": missed,
            "count": len(delays),
        }

    @staticmethod
    def _timestamp_to_seconds(ts: str) -> float:
        parts = ts.replace(",", ".").split(":")
        h, m = int(parts[0]), int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
