"""
Audio Aligner — Calculate timing deltas and apply alignment strategy.

Handles the 3 scenarios: audio shorter, audio longer, and audio matching target.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from src.m3_sync.time_stretcher import TimeStretcher


class AudioAligner:
    """Align audio chunks to subtitle timestamps."""

    def __init__(
        self,
        time_stretcher: Optional[TimeStretcher] = None,
        tolerance_sec: float = 0.2,
        max_stretch_ratio: float = 1.5,
    ):
        """
        Args:
            time_stretcher: TimeStretcher instance.
            tolerance_sec: Acceptable timing difference in seconds.
            max_stretch_ratio: Maximum time-stretch ratio before switching to pad/truncate.
        """
        self.time_stretcher = time_stretcher or TimeStretcher()
        self.tolerance_sec = tolerance_sec
        self.max_stretch_ratio = max_stretch_ratio

    def calculate_deltas(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Calculate time delta for each audio segment vs its target timestamp.

        Args:
            segments: List of segments with 'start_time', 'end_time', 'audio_path'.

        Returns:
            Segments with added 'delta', 'target_duration', 'audio_duration' fields.
        """
        import soundfile as sf

        results = []
        for segment in segments:
            audio_path = segment.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                logger.warning(f"Missing audio for segment {segment.get('index')}")
                continue

            # Calculate target duration from timestamps
            start_sec = self._timestamp_to_seconds(segment["start_time"])
            end_sec = self._timestamp_to_seconds(segment["end_time"])
            target_duration = end_sec - start_sec

            # Get actual audio duration
            info = sf.info(audio_path)
            audio_duration = info.duration

            delta = audio_duration - target_duration

            result = segment.copy()
            result["target_duration"] = target_duration
            result["audio_duration"] = audio_duration
            result["delta"] = delta
            result["start_sec"] = start_sec
            result["end_sec"] = end_sec

            # Determine strategy
            if abs(delta) <= self.tolerance_sec:
                result["strategy"] = "exact"
            elif delta > 0:
                result["strategy"] = "stretch_compress"  # Audio too long
            else:
                result["strategy"] = "stretch_expand"  # Audio too short

            results.append(result)

        return results

    def align_all(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Align all audio segments to their target timestamps.

        Args:
            segments: Segments with calculated deltas.
            output_dir: Directory for aligned audio files.

        Returns:
            Segments with updated 'aligned_audio_path'.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, segment in enumerate(segments):
            strategy = segment.get("strategy", "exact")
            audio_path = segment.get("audio_path")
            target_duration = segment.get("target_duration", 0)

            if strategy == "exact":
                result = segment.copy()
                result["aligned_audio_path"] = audio_path
                logger.debug(f"Segment {i}: exact match (no adjustment needed)")

            elif strategy in ("stretch_compress", "stretch_expand"):
                aligned_path = output_dir / f"aligned_{i:04d}.wav"
                output, method = self.time_stretcher.stretch_to_fit(
                    audio_path=audio_path,
                    target_duration=target_duration,
                    tolerance=self.tolerance_sec,
                )
                result = segment.copy()
                result["aligned_audio_path"] = str(output)
                result["align_method"] = method
                logger.debug(
                    f"Segment {i}: {method} "
                    f"({segment['audio_duration']:.2f}s → {target_duration:.2f}s)"
                )
            else:
                result = segment.copy()
                result["aligned_audio_path"] = audio_path
                logger.warning(f"Segment {i}: unknown strategy '{strategy}'")

            results.append(result)

        # Summary
        strategies = [r.get("strategy", "unknown") for r in results]
        logger.info(
            f"Aligned {len(results)} segments: "
            f"{strategies.count('exact')} exact, "
            f"{strategies.count('stretch_compress')} compressed, "
            f"{strategies.count('stretch_expand')} expanded"
        )

        return results

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
        parts = timestamp.replace(",", ":").split(":")
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + ms / 1000.0
