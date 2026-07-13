"""
TTS Client — VieNeu-TTS v2 Turbo core engine for Vietnamese voice synthesis.

Implements zero-shot voice cloning using the `vieneu` SDK:
  - Model  : pnnbao-ump/VieNeu-TTS-v2-Turbo  (Turbo mode, 24 kHz)
  - Backend: GGUF + ONNX (no PyTorch/transformers needed)
  - API    : Vieneu() -> encode_reference() -> infer()

Pipeline:
  reference_audio -> encode_reference() -> ref_codes (cached)
  text + ref_codes -> infer()            -> audio waveform (np.ndarray)
  audio waveform  -> save()              -> .wav file (24 kHz)
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from loguru import logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIENEU_TURBO_REPO = "pnnbao-ump/VieNeu-TTS-v2-Turbo"
VIENEU_SAMPLE_RATE = 24_000  # VieNeu native output: 24 kHz


class TTSClient:
    """
    Unified TTS client backed by VieNeu-TTS v2 Turbo.

    Supports three engine modes (for backward-compat & flexibility):
      - "vieneu"  : VieNeu-TTS v2 Turbo via vieneu SDK  ← default / recommended
      - "f5-tts"  : F5-TTS (not implemented, kept as future slot)
      - "vixtts"  : viXTTS-v2 (not implemented, kept as future slot)

    Voice Cloning flow:
      1. call encode_reference(audio_path) once per speaker
      2. call synthesize(text, ref_codes=ref_codes) for every segment
      The ref_codes are cached on disk; repeated calls are instant.
    """

    def __init__(
        self,
        engine: str = "vieneu",
        # VieNeu-specific
        vieneu_mode: str = "turbo",
        backbone_repo: Optional[str] = None,
        backbone_device: str = "cuda",
        codec_device: str = "cpu",
        codec_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        # General
        sample_rate: int = VIENEU_SAMPLE_RATE,
        speed: float = 1.0,
        # Legacy (kept for backward compat signature)
        model_path: str = "./models/vitts",
    ):
        """
        Args:
            engine          : TTS engine — "vieneu" | "f5-tts" | "vixtts".
            vieneu_mode     : Vieneu() factory mode — "turbo" (default, GGUF+ONNX)
                              | "standard" (PyTorch) | "fast" (LMDeploy) | "remote" (API).
            backbone_repo   : HuggingFace repo for backbone. None = SDK default.
            backbone_device : Device for the backbone ("cuda" or "cpu").
            codec_device    : Device for the codec ("cpu" for ONNX).
            codec_repo      : Codec repo. None = SDK default.
            hf_token        : HuggingFace token (for gated/private repos).
            sample_rate     : Output audio sample rate (24000 for VieNeu).
            speed           : Speech speed multiplier (reserved, not used yet).
            model_path      : Legacy — kept for backward compat; not used when engine=vieneu.
        """
        self.engine = engine
        self.vieneu_mode = vieneu_mode
        self.backbone_repo = backbone_repo
        self.backbone_device = backbone_device
        self.codec_device = codec_device
        self.codec_repo = codec_repo
        self.hf_token = hf_token
        self.sample_rate = sample_rate
        self.speed = speed
        self.model_path = model_path

        # Internal state
        self._tts = None          # vieneu.Vieneu instance
        self._ref_codes_cache: Dict[str, Any] = {}  # audio_path_hash -> ref_codes

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the VieNeu TTS model (only on first synthesis call)."""
        if self._tts is not None:
            return

        if self.engine == "vieneu":
            self._load_vieneu()
        elif self.engine in ("f5-tts", "vixtts"):
            raise NotImplementedError(
                f"Engine '{self.engine}' is reserved for future use. "
                "Use engine='vieneu' with VieNeu-TTS v2 Turbo."
            )
        else:
            raise ValueError(
                f"Unknown TTS engine: '{self.engine}'. "
                "Valid options: 'vieneu', 'f5-tts', 'vixtts'."
            )

    def _load_vieneu(self) -> None:
        """
        Initialize VieNeu-TTS via vieneu SDK.

        Default mode="turbo" uses GGUF backbone + ONNX codec —
        no PyTorch/transformers required. Models are auto-downloaded
        from HuggingFace on first run and cached locally.
        """
        try:
            from vieneu import Vieneu
        except ImportError as exc:
            raise ImportError(
                "vieneu package not found. Install with: pip install vieneu"
            ) from exc

        logger.info(
            f"Loading VieNeu-TTS | mode={self.vieneu_mode} | "
            f"backbone_device={self.backbone_device} | codec_device={self.codec_device}"
        )
        t0 = time.perf_counter()

        # Only pass explicitly set params; let SDK defaults handle the rest
        kwargs: Dict[str, Any] = {}
        if self.backbone_repo is not None:
            kwargs["backbone_repo"] = self.backbone_repo
            kwargs["backbone_device"] = self.backbone_device
        if self.codec_repo is not None:
            kwargs["codec_repo"] = self.codec_repo
            kwargs["codec_device"] = self.codec_device
        if self.hf_token:
            kwargs["hf_token"] = self.hf_token

        self._tts = Vieneu(mode=self.vieneu_mode, **kwargs)

        elapsed = time.perf_counter() - t0
        logger.info(f"VieNeu-TTS ready ({elapsed:.1f}s)")

    def unload_model(self) -> None:
        """Unload model and free GPU VRAM."""
        if self._tts is not None:
            try:
                self._tts.close()
            except Exception:
                pass
            self._tts = None
            self._ref_codes_cache.clear()

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("VieNeu-TTS model unloaded, VRAM freed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.unload_model()

    # ------------------------------------------------------------------
    # Voice cloning — reference encoding
    # ------------------------------------------------------------------

    def encode_reference(
        self,
        audio_path: str | Path,
        use_cache: bool = True,
    ) -> Any:
        """
        Encode a reference audio file into VieNeu ref_codes (speaker identity).

        This is the "fingerprint" step of zero-shot voice cloning.
        ref_codes are discrete audio tokens extracted by NeuCodec from the
        reference wav. They are passed to infer() to condition generation
        on the target speaker's voice.

        Args:
            audio_path : Path to reference audio (.wav / .mp3 / .flac).
                         3-5 seconds of clean speech gives best quality.
            use_cache  : If True, re-use cached ref_codes for the same file
                         (avoids re-encoding on every batch call).

        Returns:
            ref_codes: Opaque VieNeu reference codes object.
        """
        self._load_model()

        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {audio_path}")

        # Build a stable cache key from file path + modification time
        cache_key = self._make_cache_key(audio_path)

        if use_cache and cache_key in self._ref_codes_cache:
            logger.debug(f"Using cached ref_codes for: {audio_path.name}")
            return self._ref_codes_cache[cache_key]

        logger.info(f"Encoding reference voice: {audio_path.name}")
        t0 = time.perf_counter()

        ref_codes = self._tts.encode_reference(str(audio_path))

        elapsed = time.perf_counter() - t0
        logger.info(f"Reference encoded ({elapsed:.2f}s): {audio_path.name}")

        if use_cache:
            self._ref_codes_cache[cache_key] = ref_codes

        return ref_codes

    @staticmethod
    def _make_cache_key(audio_path: Path) -> str:
        """Stable cache key: sha1 of (absolute_path + mtime)."""
        stat = audio_path.stat()
        raw = f"{audio_path.resolve()}:{stat.st_mtime}"
        return hashlib.sha1(raw.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        ref_codes: Optional[Any] = None,
        voice: Optional[Any] = None,
        ref_text: Optional[str] = None,
        reference_audio: Optional[str | Path] = None,
        # Legacy compat param (not used by vieneu engine)
        speaker_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Synthesize Vietnamese speech from text.

        Voice cloning priority:
          1. ref_codes (pre-encoded, most efficient for batching)
          2. reference_audio path (encodes on the fly)
          3. No voice input → VieNeu default preset voice

        Args:
            text            : Vietnamese (or bilingual Vi/En) text to synthesize.
            ref_codes       : Pre-computed ref_codes from encode_reference().
            ref_text        : Transcript of reference audio (optional for Turbo v2).
            reference_audio : Path to reference wav; used if ref_codes is None.
            speaker_embedding: Legacy param — ignored for VieNeu engine.

        Returns:
            Audio waveform as np.ndarray (float32, 24 kHz mono).
        """
        self._load_model()

        text = self._normalize_text(text)
        if not text:
            logger.warning("Empty text after normalization, returning silence")
            return np.zeros(self.sample_rate, dtype=np.float32)

        # If no ref_codes/voice but a reference_audio path given, encode on the fly
        if ref_codes is None and voice is None and reference_audio is not None:
            ref_codes = self.encode_reference(reference_audio)

        # Build infer kwargs
        infer_kwargs: Dict[str, Any] = {"text": text}
        if voice is not None:
            infer_kwargs["voice"] = voice
        elif ref_codes is not None:
            infer_kwargs["ref_codes"] = ref_codes
        if ref_text:
            infer_kwargs["ref_text"] = ref_text

        logger.debug(f"Synthesizing ({len(text)} chars): {text[:60]}...")
        t0 = time.perf_counter()

        audio = self._tts.infer(**infer_kwargs)

        elapsed = time.perf_counter() - t0
        duration = len(audio) / self.sample_rate
        logger.debug(f"Synthesized {duration:.2f}s audio in {elapsed:.2f}s (RTF={elapsed/max(duration,1e-6):.2f})")

        # Ensure float32 numpy array
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        return audio.astype(np.float32)

    def synthesize_to_file(
        self,
        text: str,
        output_path: str | Path,
        ref_codes: Optional[Any] = None,
        voice: Optional[Any] = None,
        ref_text: Optional[str] = None,
        reference_audio: Optional[str | Path] = None,
        # Legacy compat
        speaker_embedding: Optional[np.ndarray] = None,
    ) -> Path:
        """
        Synthesize speech and save to a .wav file.

        Args:
            text        : Vietnamese text to synthesize.
            output_path : Path for the output .wav file.
            ref_codes   : Pre-encoded speaker ref_codes (preferred).
            ref_text    : Reference transcript (optional for Turbo v2).
            reference_audio: Reference audio path (if ref_codes not available).
            speaker_embedding: Legacy param — ignored.

        Returns:
            Path to the saved .wav file.
        """
        audio = self.synthesize(
            text=text,
            ref_codes=ref_codes,
            voice=voice,
            ref_text=ref_text,
            reference_audio=reference_audio,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio, self.sample_rate)

        duration = len(audio) / self.sample_rate
        logger.info(f"Saved {duration:.2f}s audio → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Preset voices
    # ------------------------------------------------------------------

    def list_preset_voices(self) -> List[Tuple[str, str]]:
        """
        List available preset voices bundled with VieNeu.

        Returns:
            List of (description, voice_id) tuples.
        """
        self._load_model()
        voices = self._tts.list_preset_voices()
        logger.info(f"Available preset voices: {len(voices)}")
        for desc, vid in voices:
            logger.debug(f"  [{vid}] {desc}")
        return voices

    def get_preset_voice(self, voice_id: str) -> Any:
        """
        Retrieve a preset voice by ID for use in synthesize().

        Args:
            voice_id: Voice identifier from list_preset_voices().

        Returns:
            Voice data object passable to infer(voice=...).
        """
        self._load_model()
        voice = self._tts.get_preset_voice(voice_id)
        logger.debug(f"Loaded preset voice: {voice_id}")
        return voice

    def synthesize_with_preset(
        self,
        text: str,
        voice_id: str,
        output_path: Optional[str | Path] = None,
    ) -> np.ndarray:
        """
        Synthesize text using a VieNeu preset voice (no reference audio needed).

        Args:
            text      : Text to synthesize.
            voice_id  : Voice ID from list_preset_voices().
            output_path: If provided, save to this path.

        Returns:
            Audio waveform as np.ndarray.
        """
        self._load_model()
        text = self._normalize_text(text)
        voice = self.get_preset_voice(voice_id)
        audio = self._tts.infer(text=text, voice=voice)
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(output_path), audio, self.sample_rate)
            logger.info(f"Preset voice audio saved → {output_path}")

        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Text normalization
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """
        Normalize Vietnamese text before TTS synthesis.

        Handles basic cleanup. VieNeu handles Vietnamese phonology internally.
        Future: integrate underthesea for number→word conversion.
        """
        text = text.strip()
        if not text:
            return text

        # Ellipsis normalization
        text = text.replace("...", "…")

        # Remove excess whitespace
        import re
        text = re.sub(r"\s+", " ", text)

        return text
