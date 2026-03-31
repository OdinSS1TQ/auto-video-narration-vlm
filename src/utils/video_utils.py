"""
Video Utilities — ffprobe, duration, and video info helpers.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional


def get_video_duration(video_path: str | Path) -> float:
    """Get video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def get_video_info(video_path: str | Path) -> Dict[str, Any]:
    """Get full video metadata via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return json.loads(result.stdout)


def get_video_resolution(video_path: str | Path) -> tuple[int, int]:
    """Get video resolution (width, height)."""
    info = get_video_info(video_path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream["width"], stream["height"]
    raise ValueError("No video stream found")


def get_fps(video_path: str | Path) -> float:
    """Get video frame rate."""
    info = get_video_info(video_path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            r_frame_rate = stream.get("r_frame_rate", "30/1")
            num, den = map(int, r_frame_rate.split("/"))
            return num / den if den != 0 else 30.0
    return 30.0


def extract_audio(
    video_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 22050,
) -> Path:
    """Extract audio track from video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-acodec", "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
    return output_path
