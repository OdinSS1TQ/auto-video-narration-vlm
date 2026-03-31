"""
Scene Detector — PySceneDetect wrapper for intelligent frame sampling.

Detects scene changes in video to sample frames at content boundaries
instead of fixed intervals, reducing token usage and hallucination.
"""

from pathlib import Path
from typing import List, Tuple

from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector


class SceneDetector:
    """Detect scene changes in video using PySceneDetect."""

    def __init__(self, threshold: float = 27.0, min_scene_len: int = 15):
        """
        Args:
            threshold: Content change threshold (higher = less sensitive).
            min_scene_len: Minimum scene length in frames.
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect_scenes(self, video_path: str | Path) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries in a video.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of (start_time, end_time) tuples in seconds.
        """
        video_path = str(video_path)
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(
            ContentDetector(
                threshold=self.threshold,
                min_scene_len=self.min_scene_len,
            )
        )

        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()

        return [
            (scene[0].get_seconds(), scene[1].get_seconds())
            for scene in scene_list
        ]

    def get_scene_midpoints(self, video_path: str | Path) -> List[float]:
        """
        Get midpoint timestamps of each scene for frame sampling.

        Args:
            video_path: Path to the input video file.

        Returns:
            List of midpoint timestamps in seconds.
        """
        scenes = self.detect_scenes(video_path)
        return [(start + end) / 2 for start, end in scenes]

    def split_into_chunks(
        self,
        video_path: str | Path,
        chunk_duration: float = 45.0,
    ) -> List[Tuple[float, float]]:
        """
        Split video into processing chunks, respecting scene boundaries.

        Args:
            video_path: Path to the input video file.
            chunk_duration: Target chunk duration in seconds.

        Returns:
            List of (start_time, end_time) tuples for each chunk.
        """
        scenes = self.detect_scenes(video_path)
        if not scenes:
            # Fallback: single chunk for entire video
            return [(0.0, scenes[-1][1] if scenes else 0.0)]

        chunks = []
        current_start = scenes[0][0]
        current_end = current_start

        for scene_start, scene_end in scenes:
            if scene_end - current_start > chunk_duration and current_end > current_start:
                chunks.append((current_start, current_end))
                current_start = scene_start
            current_end = scene_end

        # Add remaining
        if current_end > current_start:
            chunks.append((current_start, current_end))

        return chunks
