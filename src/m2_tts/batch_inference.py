"""
Batch Inference — Batch audio generation for improved throughput.

Groups short text segments for efficient TTS processing.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from src.m2_tts.tts_client import TTSClient


class BatchInference:
    """Batch processing for TTS generation."""

    def __init__(
        self,
        tts_client: TTSClient,
        max_batch_size: int = 8,
        max_text_length: int = 200,
    ):
        """
        Args:
            tts_client: Initialized TTS client.
            max_batch_size: Maximum number of segments per batch.
            max_text_length: Maximum text length per segment.
        """
        self.tts_client = tts_client
        self.max_batch_size = max_batch_size
        self.max_text_length = max_text_length

    def prepare_batches(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Group segments into batches for efficient processing.

        Args:
            segments: List of subtitle entries with 'translated_text'.

        Returns:
            List of batches (each batch is a list of segments).
        """
        batches = []
        current_batch = []
        current_length = 0

        for segment in segments:
            text = segment.get("translated_text", "")
            text_len = len(text)

            if (
                len(current_batch) >= self.max_batch_size
                or current_length + text_len > self.max_text_length * self.max_batch_size
            ):
                if current_batch:
                    batches.append(current_batch)
                current_batch = [segment]
                current_length = text_len
            else:
                current_batch.append(segment)
                current_length += text_len

        if current_batch:
            batches.append(current_batch)

        logger.info(
            f"Prepared {len(batches)} batches from {len(segments)} segments"
        )
        return batches

    async def process_all(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str | Path,
        speaker_embedding: Optional[np.ndarray] = None,
        reference_audio: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Process all segments and generate audio files.

        Args:
            segments: List of subtitle entries.
            output_dir: Directory to save audio chunks.
            speaker_embedding: Pre-computed speaker embedding.
            reference_audio: Path to reference audio.

        Returns:
            List of segments with added 'audio_path' field.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, segment in enumerate(segments):
            text = segment.get("translated_text", "")
            if not text.strip():
                logger.warning(f"Skipping empty segment {i}")
                continue

            output_path = output_dir / f"chunk_{i:04d}.wav"

            try:
                self.tts_client.synthesize_to_file(
                    text=text,
                    output_path=output_path,
                    speaker_embedding=speaker_embedding,
                    reference_audio=reference_audio,
                )

                result = segment.copy()
                result["audio_path"] = str(output_path)
                results.append(result)

                logger.debug(f"Generated audio for segment {i}: {output_path}")
            except Exception as e:
                logger.error(f"Failed to generate audio for segment {i}: {e}")
                result = segment.copy()
                result["audio_path"] = None
                result["error"] = str(e)
                results.append(result)

        logger.info(
            f"Generated {sum(1 for r in results if r.get('audio_path'))} / "
            f"{len(results)} audio chunks"
        )
        return results
