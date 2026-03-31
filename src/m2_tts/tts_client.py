"""
TTS Client — F5-TTS / viXTTS interface for Vietnamese voice synthesis.

Generates speech audio from Vietnamese text using a cloned voice.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from loguru import logger


class TTSClient:
    """Unified TTS client supporting F5-TTS and viXTTS."""

    def __init__(
        self,
        engine: str = "f5-tts",
        model_path: str = "./models/vitts",
        device: str = "cuda",
        sample_rate: int = 22050,
        speed: float = 1.0,
    ):
        """
        Args:
            engine: TTS engine to use ('f5-tts' or 'vixtts').
            model_path: Path to model weights.
            device: Device for inference ('cuda' or 'cpu').
            sample_rate: Output audio sample rate.
            speed: Speech speed multiplier.
        """
        self.engine = engine
        self.model_path = model_path
        self.device = device
        self.sample_rate = sample_rate
        self.speed = speed
        self._model = None

    def _load_model(self):
        """Lazy load the TTS model."""
        if self._model is not None:
            return

        if self.engine == "f5-tts":
            self._load_f5tts()
        elif self.engine == "vixtts":
            self._load_vixtts()
        else:
            raise ValueError(f"Unknown TTS engine: {self.engine}")

    def _load_f5tts(self):
        """Load F5-TTS Vietnamese model."""
        # TODO: Implement F5-TTS loading
        # from f5_tts import F5TTS
        # self._model = F5TTS.from_pretrained(self.model_path)
        logger.info(f"Loading F5-TTS from {self.model_path}")
        raise NotImplementedError(
            "F5-TTS loading not yet implemented. "
            "Download model weights first."
        )

    def _load_vixtts(self):
        """Load viXTTS-v2 model."""
        # TODO: Implement viXTTS loading
        # from TTS.api import TTS
        # self._model = TTS(model_path=self.model_path)
        logger.info(f"Loading viXTTS from {self.model_path}")
        raise NotImplementedError(
            "viXTTS loading not yet implemented. "
            "Download model weights first."
        )

    def synthesize(
        self,
        text: str,
        speaker_embedding: Optional[np.ndarray] = None,
        reference_audio: Optional[str] = None,
    ) -> np.ndarray:
        """
        Synthesize speech from text.

        Args:
            text: Vietnamese text to synthesize.
            speaker_embedding: Pre-computed speaker embedding (optional).
            reference_audio: Path to reference audio for voice cloning.

        Returns:
            Audio waveform as numpy array.
        """
        self._load_model()

        # Normalize text before synthesis
        text = self._normalize_text(text)

        # TODO: Implement actual synthesis
        # if self.engine == "f5-tts":
        #     return self._synthesize_f5tts(text, speaker_embedding, reference_audio)
        # else:
        #     return self._synthesize_vixtts(text, speaker_embedding, reference_audio)
        raise NotImplementedError

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        speaker_embedding: Optional[np.ndarray] = None,
        reference_audio: Optional[str] = None,
    ) -> Path:
        """
        Synthesize speech and save to file.

        Args:
            text: Vietnamese text to synthesize.
            output_path: Output audio file path.
            speaker_embedding: Pre-computed speaker embedding.
            reference_audio: Path to reference audio.

        Returns:
            Path to the saved audio file.
        """
        audio = self.synthesize(text, speaker_embedding, reference_audio)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.sample_rate)

        logger.info(f"Saved audio to {output_path} ({len(audio)/self.sample_rate:.2f}s)")
        return output_path

    def _normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text for TTS.

        Handles:
        - Number to word conversion (1000 → "một nghìn")
        - Abbreviation expansion
        - Punctuation normalization
        """
        # TODO: Implement full normalization with underthesea
        # from underthesea import text_normalize
        # Basic normalization for now
        text = text.strip()
        # Replace common patterns
        text = text.replace("...", "…")
        return text

    def unload_model(self):
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("Unloaded TTS model")
