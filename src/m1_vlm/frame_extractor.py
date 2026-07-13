"""
Frame Extractor — Extract key frames from video and encode to base64.

Used to prepare visual input for VLM processing.

AI Engineer Design Notes:
    This module serves as the bridge between raw video and VLM input.
    Key design decisions:
    1. All extraction methods return (timestamp, frame) tuples for traceability
       — the VLM needs timestamps to generate accurate SRT entries.
    2. Frames are resized before base64 encoding to reduce token consumption
       without losing semantic content (768px width preserves UI text readability).
    3. JPEG quality=85 balances file size (~60% smaller than PNG) vs. visual fidelity
       — VLMs are robust to JPEG compression artifacts at this quality level.
"""

import base64
import io
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image


class FrameExtractor:
    """Extract and encode frames from video for VLM input."""

    def __init__(self, max_width: int = 768, quality: int = 85):
        """
        Args:
            max_width: Maximum frame width (resized to save tokens).
                       768px preserves text readability for code/UI screenshots.
            quality: JPEG quality for base64 encoding (1-100).
        """
        self.max_width = max_width
        self.quality = quality

    def extract_frames_at_timestamps(
        self,
        video_path: str | Path,
        timestamps: List[float],
    ) -> List[tuple[float, np.ndarray]]:
        """
        Extract frames at specific timestamps.

        Returns (timestamp, frame) tuples for downstream timestamp alignment.
        Changed from original List[np.ndarray] to include timestamps — this is
        essential for the adaptive sampling pipeline where timestamps are
        non-uniform and must be preserved for SRT generation.

        Args:
            video_path: Path to the input video file.
            timestamps: List of timestamps in seconds.

        Returns:
            List of (timestamp, frame) tuples. Frames are BGR numpy arrays.
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            for ts in sorted(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if ret:
                    frames.append((ts, frame))
        finally:
            cap.release()

        return frames

    def extract_frames_in_range(
        self,
        video_path: str | Path,
        start_time: float,
        end_time: float,
        interval: float = 2.0,
    ) -> List[tuple[float, np.ndarray]]:
        """
        Extract frames at regular intervals within a time range.

        This is the legacy fixed-interval method. Prefer extract_frames_at_timestamps()
        with SceneDetector.get_adaptive_timestamps() for content-aware sampling.

        Args:
            video_path: Path to the input video file.
            start_time: Start timestamp in seconds.
            end_time: End timestamp in seconds.
            interval: Time between frames in seconds.

        Returns:
            List of (timestamp, frame) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        current_time = start_time

        try:
            while current_time < end_time:
                cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
                ret, frame = cap.read()
                if ret:
                    frames.append((current_time, frame))
                current_time += interval
        finally:
            cap.release()

        return frames

    def extract_frames_evenly(
        self,
        video_path: str | Path,
        n: int = 15,
    ) -> List[np.ndarray]:
        """
        Sample N frames uniformly distributed across the entire video.

        AI Researcher Note:
            This method is designed for the Global Summary Pass (Pass 0).
            By sampling uniformly, we capture the macroscopic structure of the
            video — introduction, main content sections, and conclusion —
            without being biased toward any particular segment.

            15 frames is the sweet spot for tutorial videos:
            - Enough to cover all major sections (most tutorials have 3-7 sections)
            - Few enough to fit in a single VLM context window
            - Each frame represents ~5-20% of a typical 3-4 minute video

        Args:
            video_path: Path to the input video file.
            n: Number of frames to sample. Default 15.

        Returns:
            List of frames as numpy arrays (BGR).
            Does NOT include timestamps since these are for high-level overview only.
        """
        cap = cv2.VideoCapture(str(video_path))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            if total_frames <= 0 or fps <= 0:
                return []

            duration = total_frames / fps

            # Generate evenly spaced timestamps across the full video
            if n <= 1:
                timestamps = [duration / 2]
            else:
                timestamps = [duration * i / (n - 1) for i in range(n)]
        finally:
            cap.release()

        # Reuse extract method, discard timestamps for global overview
        frame_pairs = self.extract_frames_at_timestamps(video_path, timestamps)
        return [frame for _, frame in frame_pairs]

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert a frame to base64-encoded JPEG string.

        Applies width-based downscaling to reduce VLM token consumption.
        768px width preserves text readability while reducing image tokens
        by ~4x compared to 1920px originals.

        Args:
            frame: Frame as numpy array (BGR format).

        Returns:
            Base64-encoded JPEG string.
        """
        # Resize if needed
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            frame = cv2.resize(frame, (self.max_width, int(h * scale)))

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # Encode to JPEG
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)

        return base64.b64encode(buffer.read()).decode("utf-8")

    def frames_to_base64_batch(self, frames: List[np.ndarray]) -> List[str]:
        """
        Convert multiple frames to base64-encoded strings.

        Args:
            frames: List of frames as numpy arrays.

        Returns:
            List of base64-encoded JPEG strings.
        """
        return [self.frame_to_base64(frame) for frame in frames]

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: str | Path,
    ) -> Path:
        """
        Save a frame to disk.

        Args:
            frame: Frame as numpy array.
            output_path: Output file path.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)
        return output_path
