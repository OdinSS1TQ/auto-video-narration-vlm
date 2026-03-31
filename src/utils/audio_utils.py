"""
Audio Utilities — Load, resample, and normalize audio.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf


def load_audio(
    audio_path: str | Path,
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load audio file and optionally resample.

    Args:
        audio_path: Path to audio file.
        target_sr: Target sample rate (None = keep original).

    Returns:
        Tuple of (audio_data, sample_rate).
    """
    import librosa
    y, sr = librosa.load(str(audio_path), sr=target_sr)
    return y, sr


def normalize_audio(
    audio: np.ndarray,
    target_db: float = -20.0,
) -> np.ndarray:
    """
    Normalize audio to target dB level.

    Args:
        audio: Audio waveform.
        target_db: Target dB level.

    Returns:
        Normalized audio.
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio

    current_db = 20 * np.log10(rms)
    gain_db = target_db - current_db
    gain = 10 ** (gain_db / 20)

    return audio * gain


def save_audio(
    audio: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 22050,
) -> Path:
    """Save audio to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sample_rate)
    return output_path


def get_audio_duration(audio_path: str | Path) -> float:
    """Get audio file duration in seconds."""
    info = sf.info(str(audio_path))
    return info.duration


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate."""
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
