"""
Time Stretcher — Rubberband wrapper for pitch-preserving time stretching.

Adjusts audio duration to match subtitle timestamps without changing pitch.
"""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


class TimeStretcher:
    """Time-stretch audio using rubberband for quality pitch preservation."""

    def __init__(self, rubberband_path: Optional[str] = None):
        """
        Args:
            rubberband_path: Path to rubberband CLI binary. If None, reads
                env var RUBBERBAND_PATH, falling back to "rubberband" on PATH.
        """
        self.rubberband_path = (
            rubberband_path or os.getenv("RUBBERBAND_PATH") or "rubberband"
        )
        self._check_rubberband()

    def _check_rubberband(self):
        """Verify rubberband is available."""
        try:
            result = subprocess.run(
                [self.rubberband_path, "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            logger.debug(f"Rubberband available: {result.stdout.strip()}")
        except FileNotFoundError:
            logger.warning(
                "Rubberband CLI not found. Install with: "
                "apt install rubberband-cli (Linux) or "
                "brew install rubberband (macOS)"
            )

    def stretch(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_duration: float,
        pitch_shift: float = 0.0,
    ) -> Path:
        """
        Time-stretch an audio file to target duration.

        Args:
            input_path: Input audio file path.
            output_path: Output audio file path.
            target_duration: Target duration in seconds.
            pitch_shift: Pitch shift in semitones (0 = no change).

        Returns:
            Path to the stretched audio file.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate current duration
        info = sf.info(str(input_path))
        current_duration = info.duration

        if current_duration <= 0:
            raise ValueError(f"Invalid audio duration: {current_duration}")

        # Calculate stretch ratio
        ratio = target_duration / current_duration

        # Limit stretch ratio to avoid extreme distortion
        ratio = max(0.5, min(2.0, ratio))

        cmd = [
            self.rubberband_path,
            "--time", str(ratio),
        ]

        if pitch_shift != 0.0:
            cmd.extend(["--pitch", str(pitch_shift)])

        cmd.extend([str(input_path), str(output_path)])

        logger.debug(
            f"Stretching {input_path.name}: "
            f"{current_duration:.2f}s → {target_duration:.2f}s "
            f"(ratio={ratio:.3f})"
        )

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"Rubberband failed: {result.stderr}")

        return output_path

    def stretch_to_fit(
        self,
        audio_path: str | Path,
        target_duration: float,
        tolerance: float = 0.1,
        output_dir: Optional[str | Path] = None,
    ) -> tuple[Path, str]:
        """
        Stretch audio to fit a target duration, choosing the best strategy.

        Args:
            audio_path: Input audio file path.
            target_duration: Target duration in seconds.
            tolerance: Acceptable duration difference in seconds.
            output_dir: Directory for stretched output. Defaults to input file's
                parent (legacy behavior). Pass an explicit dir to keep input dir
                clean.

        Returns:
            Tuple of (output_path, strategy_used).
            strategy_used: 'exact', 'stretch', 'pad', 'truncate'.
        """
        info = sf.info(str(audio_path))
        current_duration = info.duration
        delta = target_duration - current_duration

        if abs(delta) <= tolerance:
            return Path(audio_path), "exact"

        suffix = Path(audio_path).suffix
        stem = Path(audio_path).stem
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{stem}_stretched{suffix}"
        else:
            output_path = Path(audio_path).with_name(f"{stem}_stretched{suffix}")

        if 0.5 <= target_duration / current_duration <= 2.0:
            # Within acceptable stretch range
            self.stretch(audio_path, output_path, target_duration)
            return output_path, "stretch"
        elif delta > 0:
            # Need padding (audio too short, stretch ratio too extreme)
            self._pad_audio(audio_path, output_path, target_duration)
            return output_path, "pad"
        else:
            # Need truncation
            self._truncate_audio(audio_path, output_path, target_duration)
            return output_path, "truncate"

    def _pad_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_duration: float,
    ):
        """Pad audio with silence to reach target duration."""
        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(input_path))
        current_ms = len(audio)
        target_ms = int(target_duration * 1000)

        if target_ms > current_ms:
            silence = AudioSegment.silent(duration=target_ms - current_ms)
            padded = audio + silence
            padded.export(str(output_path), format=Path(output_path).suffix[1:])

    def _truncate_audio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        target_duration: float,
    ):
        """Truncate audio to target duration."""
        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(input_path))
        target_ms = int(target_duration * 1000)
        truncated = audio[:target_ms]
        truncated.export(str(output_path), format=Path(output_path).suffix[1:])
