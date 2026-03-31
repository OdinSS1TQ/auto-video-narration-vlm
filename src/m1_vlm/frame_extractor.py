"""
Frame Extractor — Extract key frames from video and encode to base64.

Used to prepare visual input for VLM processing.
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
            quality: JPEG quality for base64 encoding (1-100).
        """
        self.max_width = max_width
        self.quality = quality

    def extract_frames_at_timestamps(
        self,
        video_path: str | Path,
        timestamps: List[float],
    ) -> List[np.ndarray]:
        """
        Extract frames at specific timestamps.

        Args:
            video_path: Path to the input video file.
            timestamps: List of timestamps in seconds.

        Returns:
            List of frames as numpy arrays (BGR).
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        try:
            for ts in sorted(timestamps):
                cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
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

    def frame_to_base64(self, frame: np.ndarray) -> str:
        """
        Convert a frame to base64-encoded JPEG string.

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
