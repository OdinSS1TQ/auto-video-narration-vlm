"""
MOS Estimator — Automated Mean Opinion Score estimation.

Uses UTMOS or MOSNet for automated speech quality assessment.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger


class MOSEstimator:
    """Estimate Mean Opinion Score for generated speech."""

    def __init__(self, backend: str = "utmos"):
        """
        Args:
            backend: MOS estimation backend ('utmos' or 'mosnet').
        """
        self.backend = backend
        self._model = None

    def _load_model(self):
        """Lazy load MOS estimation model."""
        if self._model is not None:
            return

        if self.backend == "utmos":
            # TODO: Implement UTMOS loading
            # import torch
            # self._model = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")
            logger.info("Loading UTMOS model...")
            raise NotImplementedError("UTMOS model loading not yet implemented")
        elif self.backend == "mosnet":
            raise NotImplementedError("MOSNet not yet implemented")
        else:
            raise ValueError(f"Unknown MOS backend: {self.backend}")

    def estimate(self, audio_path: str | Path) -> float:
        """
        Estimate MOS for a single audio file.

        Args:
            audio_path: Path to audio file.

        Returns:
            Estimated MOS score (1.0-5.0).
        """
        self._load_model()
        # TODO: Implement actual MOS estimation
        raise NotImplementedError

    def estimate_batch(self, audio_paths: List[str | Path]) -> List[float]:
        """
        Estimate MOS for multiple audio files.

        Args:
            audio_paths: List of audio file paths.

        Returns:
            List of MOS scores.
        """
        return [self.estimate(path) for path in audio_paths]

    def evaluate_pipeline_output(
        self,
        audio_dir: str | Path,
        pattern: str = "*.wav",
    ) -> dict:
        """
        Evaluate all audio files in a directory.

        Args:
            audio_dir: Directory containing generated audio files.
            pattern: File glob pattern.

        Returns:
            Dict with mean, std, min, max MOS scores.
        """
        audio_dir = Path(audio_dir)
        audio_files = sorted(audio_dir.glob(pattern))

        if not audio_files:
            logger.warning(f"No audio files found in {audio_dir}")
            return {"mean_mos": 0.0, "count": 0}

        scores = self.estimate_batch(audio_files)

        result = {
            "mean_mos": float(np.mean(scores)),
            "std_mos": float(np.std(scores)),
            "min_mos": float(np.min(scores)),
            "max_mos": float(np.max(scores)),
            "count": len(scores),
        }

        logger.info(f"MOS evaluation: mean={result['mean_mos']:.2f} (n={result['count']})")
        return result
