"""
Speaker Encoder — Reference audio management for VieNeu-TTS voice cloning.

Architecture note:
    With VieNeu-TTS v2 Turbo, speaker identity is captured via NeuCodec's
    encode_reference() which produces discrete ref_codes — not a float vector
    embedding like resemblyzer/ECAPA-TDNN.

    This module provides two paths:
      1. VieNeu path (primary)  : delegates to TTSClient.encode_reference()
         → returns ref_codes for direct use with tts.infer()
      2. resemblyzer path (eval): used by m5_evaluation for MOS/SIM scoring
         → returns float cosine-space embedding (not for synthesis)

    The separation keeps the evaluation pipeline intact while the synthesis
    pipeline fully migrates to VieNeu.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from loguru import logger

if TYPE_CHECKING:
    # Avoid circular import — TTSClient imported lazily below
    from src.m2_tts.tts_client import TTSClient


class SpeakerEncoder:
    """
    Manage speaker identity for voice cloning.

    Supports two backends:
      - "vieneu"     : Uses TTSClient.encode_reference() for synthesis pipeline.
      - "resemblyzer": Classic ECAPA-style float embedding for MOS evaluation.
    """

    def __init__(
        self,
        backend: str = "resemblyzer",
        cache_dir: str = "./.speaker_cache",
    ):
        """
        Args:
            backend  : "resemblyzer" (evaluation) or "vieneu" (synthesis).
            cache_dir: Directory to cache resemblyzer embeddings on disk.
        """
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._encoder = None  # resemblyzer VoiceEncoder (lazy)

    # ------------------------------------------------------------------
    # Primary path — VieNeu ref_codes
    # ------------------------------------------------------------------

    def encode_vieneu_reference(
        self,
        audio_path: str | Path,
        tts_client: "TTSClient",
        use_cache: bool = True,
    ) -> Any:
        """
        Encode reference audio into VieNeu ref_codes via TTSClient.

        This is the recommended path for the synthesis pipeline.
        ref_codes are cached inside TTSClient (in-memory, per session).

        Args:
            audio_path : Path to reference audio (.wav / .mp3 / .flac).
            tts_client : Initialized TTSClient (must have engine="vieneu").
            use_cache  : Passed through to TTSClient.encode_reference().

        Returns:
            ref_codes: VieNeu codec reference codes for voice conditioning.
        """
        audio_path = Path(audio_path)
        logger.info(f"[SpeakerEncoder] Encoding VieNeu reference: {audio_path.name}")
        return tts_client.encode_reference(audio_path, use_cache=use_cache)

    # ------------------------------------------------------------------
    # Evaluation path — resemblyzer float embedding
    # ------------------------------------------------------------------

    def _load_resemblyzer(self) -> None:
        """Lazy-load resemblyzer VoiceEncoder."""
        if self._encoder is not None:
            return
        try:
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            logger.info("[SpeakerEncoder] Loaded resemblyzer VoiceEncoder")
        except ImportError as exc:
            raise ImportError(
                "resemblyzer not found. Install with: pip install resemblyzer"
            ) from exc

    def extract_embedding(
        self,
        audio_path: str | Path,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Extract a float speaker embedding using resemblyzer.

        Used by m5_evaluation for cosine similarity / MOS scoring.
        NOT used for synthesis — pass ref_codes to TTSClient.synthesize() instead.

        Args:
            audio_path : Path to reference audio (3-10 seconds recommended).
            use_cache  : Cache embedding to disk to avoid re-computation.

        Returns:
            Speaker embedding as np.ndarray (float32, shape [256]).
        """
        audio_path = Path(audio_path)
        cache_path = self.cache_dir / f"{audio_path.stem}.npy"

        if use_cache and cache_path.exists():
            logger.debug(f"[SpeakerEncoder] Cache hit: {cache_path.name}")
            return np.load(str(cache_path))

        self._load_resemblyzer()

        from resemblyzer import preprocess_wav
        wav = preprocess_wav(audio_path)
        embedding = self._encoder.embed_utterance(wav)

        if use_cache:
            np.save(str(cache_path), embedding)
            logger.debug(f"[SpeakerEncoder] Cached embedding → {cache_path.name}")

        logger.info(
            f"[SpeakerEncoder] Extracted resemblyzer embedding: "
            f"{audio_path.name} shape={embedding.shape}"
        )
        return embedding

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Cosine similarity between two resemblyzer embeddings.

        Used by m5_evaluation to measure voice cloning fidelity.

        Args:
            embedding1: Reference speaker embedding.
            embedding2: Synthesized audio speaker embedding.

        Returns:
            Cosine similarity score in [−1.0, 1.0] (higher = more similar).
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))

    def clear_cache(self) -> None:
        """Remove all cached resemblyzer embeddings from disk."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[SpeakerEncoder] Cleared resemblyzer embedding cache")
