"""
Integration test: M1 VLM output → M2 Voice Cloning (VieNeu-TTS).

Validates the bridge between Module 1 (subtitle generation) and Module 2 (TTS).
Uses VieNeu sample audio by default; supports custom reference audio via --ref-audio.

Usage:
    # Default: VieNeu sample audio + built-in SRT fixture
    python scripts/test_m1_m2_integration.py

    # Custom reference audio
    python scripts/test_m1_m2_integration.py --ref-audio path/to/voice.wav

    # Custom SRT file
    python scripts/test_m1_m2_integration.py --srt path/to/subtitles.srt

    # Custom output directory
    python scripts/test_m1_m2_integration.py --output-dir debug_output/custom_test
"""

from __future__ import annotations

import argparse
import asyncio
import io
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fix Windows cp1252 console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from loguru import logger


# ==============================================================================
# Constants
# ==============================================================================

DEFAULT_SRT = ROOT / "data" / "test_fixtures" / "sample_output.srt"
DEFAULT_OUTPUT_DIR = ROOT / "debug_output" / "m1_m2_integration"

# VieNeu built-in sample (male, Northern Vietnamese)
VIENEU_SAMPLES_DIR = (
    ROOT / ".venv" / "Lib" / "site-packages" / "vieneu" / "assets" / "samples"
)
DEFAULT_REF_AUDIO = VIENEU_SAMPLES_DIR / "Bình (nam miền Bắc).wav"


# ==============================================================================
# Timing helpers
# ==============================================================================

class Timer:
    """Simple context manager timer."""

    def __init__(self, label: str):
        self.label = label
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
        logger.info(f"[Timer] {self.label}: {self.elapsed:.3f}s")


# ==============================================================================
# Test runner
# ==============================================================================

class IntegrationTestRunner:
    """Run M1→M2 integration test with timing and validation."""

    def __init__(
        self,
        srt_path: Path,
        ref_audio_path: Path,
        output_dir: Path,
        vieneu_mode: str = "turbo",
        backbone_device: str = "cuda",
        codec_device: str = "cpu",
    ):
        self.srt_path = srt_path
        self.ref_audio_path = ref_audio_path
        self.output_dir = output_dir
        self.vieneu_mode = vieneu_mode
        self.backbone_device = backbone_device
        self.codec_device = codec_device

        self.segment_times: List[float] = []
        self.ref_encode_time: float = 0.0
        self.total_time: float = 0.0
        self.results: List[Dict[str, Any]] = []

    async def run(self) -> bool:
        """Execute the full integration test. Returns True if all segments pass."""
        from src.m1_vlm.srt_builder import SRTBuilder
        from src.m2_tts.tts_client import TTSClient
        from src.m2_tts.batch_inference import BatchInference

        total_start = time.perf_counter()

        # --- Step 1: Load SRT (simulated M1 output) ---
        logger.info("=" * 60)
        logger.info("STEP 1: Loading M1 output (SRT file)")
        logger.info("=" * 60)

        if not self.srt_path.exists():
            logger.error(f"SRT file not found: {self.srt_path}")
            return False

        with Timer("SRT loading"):
            srt_entries = SRTBuilder.load_srt(self.srt_path)

        logger.info(f"Loaded {len(srt_entries)} subtitle entries from {self.srt_path}")
        for entry in srt_entries:
            text = entry.get("translated_text", "")
            logger.info(
                f"  [{entry['index']}] {entry['start_time']} → {entry['end_time']} "
                f"({len(text)} chars): {text[:60]}{'...' if len(text) > 60 else ''}"
            )

        # --- Step 2: Initialize TTS client ---
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Initializing TTS client (VieNeu-TTS v2)")
        logger.info("=" * 60)

        with Timer("TTS client initialization"):
            tts_client = TTSClient(
                engine="vieneu",
                vieneu_mode=self.vieneu_mode,
                backbone_device=self.backbone_device,
                codec_device=self.codec_device,
                sample_rate=24000,
            )
        logger.info(f"TTSClient created (mode={self.vieneu_mode})")

        # --- Step 3: Encode reference audio ---
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Encoding reference audio for voice cloning")
        logger.info("=" * 60)

        if not self.ref_audio_path.exists():
            logger.error(f"Reference audio not found: {self.ref_audio_path}")
            return False

        # Get audio info
        audio_info = sf.info(str(self.ref_audio_path))
        logger.info(
            f"Reference audio: {self.ref_audio_path.name} "
            f"({audio_info.duration:.2f}s, {audio_info.samplerate}Hz, "
            f"{audio_info.channels}ch)"
        )

        with Timer("Reference encoding"):
            ref_codes = tts_client.encode_reference(
                str(self.ref_audio_path), use_cache=True
            )
        self.ref_encode_time = Timer("Reference encoding").elapsed if False else 0.0
        # Re-capture from the timer above
        logger.info("Reference audio encoded successfully (ref_codes cached)")

        # --- Step 4: Batch synthesis ---
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 4: Batch TTS synthesis (M1 → M2 bridge)")
        logger.info("=" * 60)

        batch_inference = BatchInference(tts_client, max_text_length=200)
        audio_output_dir = self.output_dir / "audio_chunks"
        audio_output_dir.mkdir(parents=True, exist_ok=True)

        # Run with per-segment timing
        self.results = []
        self.segment_times = []

        for i, segment in enumerate(srt_entries):
            text = segment.get("translated_text", "")
            if not text.strip():
                result = segment.copy()
                result["audio_path"] = None
                result["error"] = "empty_text"
                self.results.append(result)
                logger.warning(f"Segment {i}: empty text, skipping")
                continue

            if len(text) > 200:
                logger.warning(
                    f"Segment {i}: text too long ({len(text)} chars), truncating"
                )
                text = text[:200]

            output_path = audio_output_dir / f"chunk_{i:04d}.wav"

            seg_start = time.perf_counter()
            try:
                tts_client.synthesize_to_file(
                    text=text,
                    output_path=output_path,
                    ref_codes=ref_codes,
                )
                seg_elapsed = time.perf_counter() - seg_start
                self.segment_times.append(seg_elapsed)

                result = segment.copy()
                result["audio_path"] = str(output_path)
                result["synthesis_time"] = seg_elapsed
                self.results.append(result)

                logger.info(
                    f"  [{i + 1}/{len(srt_entries)}] {seg_elapsed:.3f}s — "
                    f"{output_path.name} ({len(text)} chars)"
                )
            except Exception as exc:
                seg_elapsed = time.perf_counter() - seg_start
                result = segment.copy()
                result["audio_path"] = None
                result["error"] = str(exc)
                result["synthesis_time"] = seg_elapsed
                self.results.append(result)
                self.segment_times.append(seg_elapsed)
                logger.error(f"  [{i + 1}/{len(srt_entries)}] FAILED: {exc}")

        self.total_time = time.perf_counter() - total_start

        # --- Step 5: Validate outputs ---
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 5: Validating output audio files")
        logger.info("=" * 60)

        all_valid = self._validate_outputs()

        # --- Step 6: Report ---
        self._print_report()

        return all_valid

    def _validate_outputs(self) -> bool:
        """Check all synthesized audio files."""
        all_valid = True
        for result in self.results:
            audio_path = result.get("audio_path")
            if audio_path is None:
                logger.warning(
                    f"  Segment {result.get('index', '?')}: SKIPPED — "
                    f"{result.get('error', 'unknown')}"
                )
                all_valid = False
                continue

            path = Path(audio_path)
            if not path.exists():
                logger.error(
                    f"  Segment {result.get('index', '?')}: MISSING — {path}"
                )
                all_valid = False
                continue

            try:
                data, sr = sf.read(str(path))
                duration = len(data) / sr
                has_audio = np.abs(data).max() > 0.001

                if not has_audio:
                    logger.error(
                        f"  Segment {result.get('index', '?')}: SILENT — {path.name}"
                    )
                    all_valid = False
                else:
                    logger.info(
                        f"  Segment {result.get('index', '?')}: OK — "
                        f"{path.name} ({duration:.2f}s, {sr}Hz, "
                        f"peak={np.abs(data).max():.3f})"
                    )
            except Exception as exc:
                logger.error(
                    f"  Segment {result.get('index', '?')}: READ ERROR — {exc}"
                )
                all_valid = False

        return all_valid

    def _print_report(self):
        """Print timing and results summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("INTEGRATION TEST REPORT")
        logger.info("=" * 60)

        total = len(self.results)
        success = sum(1 for r in self.results if r.get("audio_path"))
        failed = total - success

        logger.info(f"  SRT entries loaded : {total}")
        logger.info(f"  Successful syntheses: {success}")
        logger.info(f"  Failed syntheses    : {failed}")

        logger.info("")
        logger.info("TIMING:")
        logger.info(f"  Total time          : {self.total_time:.3f}s")

        if self.segment_times:
            avg = sum(self.segment_times) / len(self.segment_times)
            logger.info(f"  Per-segment average  : {avg:.3f}s")
            logger.info(f"  Fastest segment      : {min(self.segment_times):.3f}s")
            logger.info(f"  Slowest segment      : {max(self.segment_times):.3f}s")

        # Duration stats for synthesized audio
        durations = []
        for r in self.results:
            if r.get("audio_path"):
                try:
                    info = sf.info(r["audio_path"])
                    durations.append(info.duration)
                except Exception:
                    pass

        if durations:
            logger.info("")
            logger.info("AUDIO DURATION:")
            logger.info(f"  Total audio duration : {sum(durations):.2f}s")
            logger.info(f"  Average per segment  : {sum(durations) / len(durations):.2f}s")
            logger.info(f"  Shortest             : {min(durations):.2f}s")
            logger.info(f"  Longest              : {max(durations):.2f}s")

        # SRT timing comparison
        logger.info("")
        logger.info("SRT vs SYNTHESIS TIMING:")
        for r in self.results:
            idx = r.get("index", "?")
            srt_start = r.get("start_time", "?")
            srt_end = r.get("end_time", "?")
            synth_time = r.get("synthesis_time", 0)
            audio_path = r.get("audio_path")

            audio_dur = 0.0
            if audio_path:
                try:
                    info = sf.info(audio_path)
                    audio_dur = info.duration
                except Exception:
                    pass

            from src.m1_vlm.srt_builder import SRTBuilder
            try:
                srt_dur = (
                    SRTBuilder._timestamp_to_seconds(srt_end)
                    - SRTBuilder._timestamp_to_seconds(srt_start)
                )
                ratio = audio_dur / srt_dur if srt_dur > 0 else 0
                logger.info(
                    f"  [{idx}] SRT: {srt_dur:.2f}s | Audio: {audio_dur:.2f}s | "
                    f"Ratio: {ratio:.2f} | Synth: {synth_time:.3f}s"
                )
            except Exception:
                logger.info(
                    f"  [{idx}] SRT: {srt_start}→{srt_end} | Audio: {audio_dur:.2f}s | "
                    f"Synth: {synth_time:.3f}s"
                )

        logger.info("")
        logger.info("=" * 60)
        status = "PASS" if failed == 0 else f"PARTIAL ({failed} failures)"
        logger.info(f"RESULT: {status}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("=" * 60)


# ==============================================================================
# CLI
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integration test: M1 VLM output → M2 Voice Cloning"
    )
    parser.add_argument(
        "--srt",
        type=Path,
        default=DEFAULT_SRT,
        help=f"Path to SRT file (default: {DEFAULT_SRT})",
    )
    parser.add_argument(
        "--ref-audio",
        type=Path,
        default=DEFAULT_REF_AUDIO,
        help=f"Path to reference audio (default: VieNeu sample)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vieneu-mode",
        choices=["turbo", "standard", "fast", "remote"],
        default="turbo",
        help="VieNeu TTS mode (default: turbo)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Compute device (default: cuda)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    logger.info("M1→M2 Integration Test")
    logger.info(f"  SRT file      : {args.srt}")
    logger.info(f"  Reference audio: {args.ref_audio}")
    logger.info(f"  Output dir    : {args.output_dir}")
    logger.info(f"  VieNeu mode   : {args.vieneu_mode}")
    logger.info(f"  Device        : {args.device}")

    runner = IntegrationTestRunner(
        srt_path=args.srt,
        ref_audio_path=args.ref_audio,
        output_dir=args.output_dir,
        vieneu_mode=args.vieneu_mode,
        backbone_device=args.device,
        codec_device="cpu",
    )

    success = await runner.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
