"""
Speaker Encoder — Extract voice embeddings from reference audio.

Uses ECAPA-TDNN (SpeechBrain) or resemblyzer for speaker verification
and voice cloning input.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger


class SpeakerEncoder:
    """Extract and manage speaker embeddings."""

    def __init__(
        self,
        backend: str = "resemblyzer",
        cache_dir: str = "./.speaker_cache",
    ):
        """
        Args:
            backend: Encoder backend ('resemblyzer' or 'speechbrain').
            cache_dir: Directory to cache speaker embeddings.
        """
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._encoder = None

    def _load_encoder(self):
        """Lazy load the speaker encoder."""
        if self._encoder is not None:
            return

        if self.backend == "resemblyzer":
            from resemblyzer import VoiceEncoder
            self._encoder = VoiceEncoder()
            logger.info("Loaded resemblyzer VoiceEncoder")
        elif self.backend == "speechbrain":
            # TODO: Implement SpeechBrain ECAPA-TDNN
            raise NotImplementedError("SpeechBrain backend not yet implemented")
        else:
            raise ValueError(f"Unknown speaker encoder backend: {self.backend}")

    def extract_embedding(
        self,
        audio_path: str | Path,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Extract speaker embedding from audio file.

        Args:
            audio_path: Path to reference audio (3-10 seconds recommended).
            use_cache: Whether to use cached embeddings.

        Returns:
            Speaker embedding as numpy array.
        """
        audio_path = Path(audio_path)
        cache_path = self.cache_dir / f"{audio_path.stem}.npy"

        # Check cache
        if use_cache and cache_path.exists():
            logger.debug(f"Loading cached embedding: {cache_path}")
            return np.load(str(cache_path))

        self._load_encoder()

        if self.backend == "resemblyzer":
            from resemblyzer import preprocess_wav
            wav = preprocess_wav(audio_path)
            embedding = self._encoder.embed_utterance(wav)
        else:
            raise NotImplementedError

        # Cache the embedding
        if use_cache:
            np.save(str(cache_path), embedding)
            logger.debug(f"Cached embedding to {cache_path}")

        logger.info(
            f"Extracted embedding from {audio_path.name}: "
            f"shape={embedding.shape}"
        )
        return embedding

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two speaker embeddings.

        Args:
            embedding1: First speaker embedding.
            embedding2: Second speaker embedding.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def clear_cache(self):
        """Clear all cached embeddings."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Cleared speaker embedding cache")
