"""
Speaker Similarity — Evaluate voice cloning quality using resemblyzer.
"""

from pathlib import Path
from typing import List

import numpy as np
from loguru import logger

from src.m2_tts.speaker_encoder import SpeakerEncoder


class SpeakerSimilarity:
    """Evaluate speaker similarity between reference and cloned voice."""

    def __init__(self, encoder: SpeakerEncoder | None = None):
        """
        Args:
            encoder: SpeakerEncoder instance (creates default if None).
        """
        self.encoder = encoder or SpeakerEncoder()

    def compute_similarity(
        self,
        reference_audio: str | Path,
        generated_audio: str | Path,
    ) -> float:
        """
        Compute cosine similarity between reference and generated audio.

        Args:
            reference_audio: Path to reference (original voice) audio.
            generated_audio: Path to generated (cloned) audio.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        ref_embedding = self.encoder.extract_embedding(reference_audio)
        gen_embedding = self.encoder.extract_embedding(generated_audio, use_cache=False)

        similarity = self.encoder.compute_similarity(ref_embedding, gen_embedding)
        logger.debug(
            f"Speaker similarity: {similarity:.4f} "
            f"({Path(reference_audio).name} vs {Path(generated_audio).name})"
        )
        return similarity

    def evaluate_batch(
        self,
        reference_audio: str | Path,
        generated_dir: str | Path,
        pattern: str = "*.wav",
    ) -> dict:
        """
        Evaluate speaker similarity for all generated audio files.

        Args:
            reference_audio: Path to reference audio.
            generated_dir: Directory with generated audio files.
            pattern: File glob pattern.

        Returns:
            Dict with mean, std, min, max similarity scores.
        """
        generated_dir = Path(generated_dir)
        generated_files = sorted(generated_dir.glob(pattern))

        if not generated_files:
            logger.warning(f"No generated audio files found in {generated_dir}")
            return {"mean_similarity": 0.0, "count": 0}

        similarities = [
            self.compute_similarity(reference_audio, gen_file)
            for gen_file in generated_files
        ]

        result = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "count": len(similarities),
        }

        logger.info(
            f"Speaker similarity: mean={result['mean_similarity']:.4f} "
            f"(n={result['count']})"
        )
        return result
