"""
Batch Inference — Efficient batch audio generation for VieNeu-TTS.

Optimisation strategy:
  - Reference audio is encoded ONCE per batch run (encode_reference is expensive).
  - The resulting ref_codes object is reused for every segment → no repeated GPU pass.
  - Segments are processed sequentially (VieNeu SDK is not thread-safe by default).
  - Failed segments are logged and skipped; partial results are returned.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from src.m2_tts.tts_client import TTSClient


class BatchInference:
    """Batch TTS processing using VieNeu-TTS v2 Turbo."""

    def __init__(
        self,
        tts_client: TTSClient,
        max_batch_size: int = 8,
        max_text_length: int = 200,
    ):
        """
        Args:
            tts_client     : Initialized TTSClient (engine="vieneu").
            max_batch_size : Max segments per logical batch (for logging / chunking).
            max_text_length: Soft cap on characters per segment.
        """
        self.tts_client = tts_client
        self.max_batch_size = max_batch_size
        self.max_text_length = max_text_length

    # ------------------------------------------------------------------
    # Batch preparation
    # ------------------------------------------------------------------

    def prepare_batches(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Group segments into logical batches for progress tracking.

        Args:
            segments: List of subtitle entries with 'translated_text'.

        Returns:
            List of batches (each batch is a list of segments).
        """
        batches: List[List[Dict[str, Any]]] = []
        current_batch: List[Dict[str, Any]] = []
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

    # ------------------------------------------------------------------
    # Main processing
    # ------------------------------------------------------------------

    async def process_all(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str | Path,
        # VieNeu voice cloning params
        ref_codes: Optional[Any] = None,
        ref_text: Optional[str] = None,
        reference_audio: Optional[str | Path] = None,
        # Legacy compat (not used by VieNeu engine)
        speaker_embedding: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate audio for all subtitle segments using VieNeu-TTS.

        ref_codes encoding strategy:
          - If ref_codes provided → use directly (most efficient, caller pre-encoded).
          - If reference_audio provided but no ref_codes → encode ONCE here, reuse.
          - If neither → VieNeu uses default preset voice.

        Args:
            segments        : List of subtitle entries with 'translated_text' key.
            output_dir      : Directory to write chunk_XXXX.wav files.
            ref_codes       : Pre-encoded speaker ref_codes (from TTSClient.encode_reference).
            ref_text        : Transcript of reference audio (optional for VieNeu Turbo v2).
            reference_audio : Path to reference .wav; encoded here if ref_codes absent.
            speaker_embedding: Legacy param — ignored (VieNeu uses ref_codes instead).

        Returns:
            List of segments with added 'audio_path' field (None on failure).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- One-time reference encoding ---
        # If caller passed a path but not pre-computed ref_codes, encode once here.
        if ref_codes is None and reference_audio is not None:
            logger.info(f"[BatchInference] Encoding reference audio once: {reference_audio}")
            ref_codes = self.tts_client.encode_reference(reference_audio, use_cache=True)

        total = len(segments)
        success_count = 0
        results: List[Dict[str, Any]] = []

        logger.info(f"[BatchInference] Processing {total} segments → {output_dir}")

        for i, segment in enumerate(segments):
            text = segment.get("translated_text", "")

            if not text.strip():
                logger.warning(f"[BatchInference] Segment {i}: empty text, skipping")
                result = segment.copy()
                result["audio_path"] = None
                result["error"] = "empty_text"
                results.append(result)
                continue

            # Truncate over-length text with warning
            if len(text) > self.max_text_length:
                logger.warning(
                    f"[BatchInference] Segment {i}: text too long "
                    f"({len(text)} chars > {self.max_text_length}), truncating"
                )
                text = text[: self.max_text_length]

            output_path = output_dir / f"chunk_{i:04d}.wav"

            try:
                self.tts_client.synthesize_to_file(
                    text=text,
                    output_path=output_path,
                    ref_codes=ref_codes,
                    ref_text=ref_text,
                )
                result = segment.copy()
                result["audio_path"] = str(output_path)
                results.append(result)
                success_count += 1
                logger.debug(f"[BatchInference] [{i+1}/{total}] ✓ {output_path.name}")

            except Exception as exc:
                logger.error(f"[BatchInference] Segment {i} failed: {exc}")
                result = segment.copy()
                result["audio_path"] = None
                result["error"] = str(exc)
                results.append(result)

        logger.info(
            f"[BatchInference] Done: {success_count}/{total} segments synthesized successfully"
        )
        return results
