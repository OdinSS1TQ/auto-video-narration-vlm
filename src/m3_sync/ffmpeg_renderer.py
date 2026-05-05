"""
FFmpeg Renderer — Final video merge with aligned audio.

Combines original video with generated Vietnamese audio track.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class FFmpegRenderer:
    """Render final dubbed video using FFmpeg."""

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        """
        Args:
            ffmpeg_path: Path to FFmpeg binary.
        """
        self.ffmpeg_path = ffmpeg_path
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is available."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            version = result.stdout.split("\n")[0]
            logger.debug(f"FFmpeg available: {version}")
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install FFmpeg first.")
            raise

    def merge_audio_segments(
        self,
        segments: List[Dict[str, Any]],
        total_duration: float,
        output_path: str | Path,
        sample_rate: int = 22050,
    ) -> Path:
        """
        Merge aligned audio segments into a single audio track.

        Uses FFmpeg's adelay filter to place each segment at the correct timestamp.

        Args:
            segments: Aligned segments with 'aligned_audio_path' and 'start_sec'.
            total_duration: Total video duration in seconds.
            output_path: Output audio file path.
            sample_rate: Audio sample rate.

        Returns:
            Path to merged audio file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not segments:
            raise ValueError("No segments to merge")

        # Build FFmpeg filter complex
        inputs = []
        filter_parts = []

        for i, segment in enumerate(segments):
            audio_path = segment.get("aligned_audio_path")
            if not audio_path or not Path(audio_path).exists():
                continue

            start_ms = int(segment.get("start_sec", 0) * 1000)
            inputs.extend(["-i", str(audio_path)])
            filter_parts.append(
                f"[{i}:a]adelay={start_ms}|{start_ms}[a{i}]"
            )

        if not filter_parts:
            raise ValueError("No valid audio segments to merge")

        # Mix all delayed segments, then pad to full video duration with silence.
        # Without apad, the merged track ends at the last segment's end, so the
        # downstream video render with -shortest truncates the entire video.
        mix_inputs = "".join(f"[a{i}]" for i in range(len(filter_parts)))
        filter_complex = ";".join(filter_parts)
        filter_complex += (
            f";{mix_inputs}amix=inputs={len(filter_parts)}:duration=longest[mixed]"
            f";[mixed]apad=whole_dur={total_duration}[out]"
        )

        cmd = [
            self.ffmpeg_path, "-y",
            *inputs,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-ar", str(sample_rate),
            "-ac", "1",
            "-t", f"{total_duration:.3f}",
            str(output_path),
        ]

        logger.debug(f"Merging {len(filter_parts)} audio segments")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        if result.returncode != 0:
            logger.error(f"FFmpeg merge failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg merge failed: {result.stderr}")

        logger.info(f"Merged audio saved to {output_path}")
        return output_path

    def render_final_video(
        self,
        video_path: str | Path,
        dubbed_audio_path: str | Path,
        output_path: str | Path,
        keep_original_audio: bool = False,
        original_volume: float = 0.1,
    ) -> Path:
        """
        Render final video with dubbed audio.

        Args:
            video_path: Original video file path.
            dubbed_audio_path: Merged dubbed audio track.
            output_path: Final output video path.
            keep_original_audio: Whether to keep original audio (mixed at low volume).
            original_volume: Volume of original audio if kept (0.0-1.0).

        Returns:
            Path to the final video.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if keep_original_audio:
            # Mix original (low volume) + dubbed audio
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", str(video_path),
                "-i", str(dubbed_audio_path),
                "-filter_complex",
                f"[0:a]volume={original_volume}[orig];"
                f"[orig][1:a]amix=inputs=2:duration=first[out]",
                "-map", "0:v",
                "-map", "[out]",
                "-c:v", "copy",
                str(output_path),
            ]
        else:
            # Replace original audio entirely
            cmd = [
                self.ffmpeg_path, "-y",
                "-i", str(video_path),
                "-i", str(dubbed_audio_path),
                "-map", "0:v",
                "-map", "1:a",
                "-c:v", "copy",
                "-shortest",
                str(output_path),
            ]

        logger.info(f"Rendering final video: {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

        if result.returncode != 0:
            logger.error(f"FFmpeg render failed: {result.stderr}")
            raise RuntimeError(f"FFmpeg render failed: {result.stderr}")

        logger.info(f"Final video saved: {output_path}")
        return output_path

    def add_subtitles(
        self,
        video_path: str | Path,
        srt_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Burn subtitles into video (optional).

        Args:
            video_path: Input video path.
            srt_path: SRT subtitle file path.
            output_path: Output video path.

        Returns:
            Path to video with burned-in subtitles.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Escape path for FFmpeg subtitle filter
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")

        cmd = [
            self.ffmpeg_path, "-y",
            "-i", str(video_path),
            "-vf", f"subtitles='{srt_escaped}'",
            "-c:a", "copy",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg subtitle burn failed: {result.stderr}")

        return output_path

    @staticmethod
    def get_video_info(video_path: str | Path) -> Dict[str, Any]:
        """Get video metadata using ffprobe."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")

        import json
        return json.loads(result.stdout)
