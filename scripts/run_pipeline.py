"""
CLI: Process a single video through the dubbing pipeline.

Usage:
    python scripts/run_pipeline.py --video data/raw/sample.mp4 --ref-audio data/reference_audio/speaker.wav
    python scripts/run_pipeline.py --video input.mp4 --ref-audio ref.wav --vlm-mode local --output output.mp4
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Force UTF-8 on stdout/stderr so emoji prints (🎬, 🎤, ✅) don't crash on
# Windows consoles defaulting to cp1252. Must run before any print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m4_pipeline.runner import PipelineRunner
from src.m4_pipeline.config import PipelineConfig
from src.m4_pipeline.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Dubbing Vietnamese — Process a single video"
    )
    parser.add_argument(
        "--video", "-v",
        required=True,
        help="Path to input video file",
    )
    parser.add_argument(
        "--ref-audio", "-a",
        required=True,
        help="Path to reference audio file (3-10s)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output video path (auto-generated if not specified)",
    )
    parser.add_argument(
        "--vlm-mode",
        choices=["api", "local"],
        default=None,
        help="VLM mode override",
    )
    parser.add_argument(
        "--tts-engine",
        choices=["f5-tts", "vixtts"],
        default=None,
        help="TTS engine override",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    setup_logger(log_level=args.log_level)

    config = PipelineConfig(config_path=args.config)

    # Override config with CLI args
    if args.vlm_mode:
        import os
        os.environ["VLM_MODE"] = args.vlm_mode
    if args.tts_engine:
        import os
        os.environ["TTS_ENGINE"] = args.tts_engine

    runner = PipelineRunner(config)

    # Progress callback for CLI
    def progress(step, current, total):
        bar = "█" * (current * 30 // total) + "░" * (30 - current * 30 // total)
        print(f"\r[{bar}] {current}/{total} — {step}", end="", flush=True)

    runner.set_progress_callback(progress)

    print(f"🎬 Processing: {args.video}")
    print(f"🎤 Reference audio: {args.ref_audio}")
    print()

    result = await runner.run(
        video_path=args.video,
        reference_audio_path=args.ref_audio,
        output_path=args.output,
    )

    print()
    print(f"\n✅ Done in {result.get('elapsed_seconds', 0):.1f}s")
    print(f"📹 Output: {result.get('output_path')}")
    print(f"📝 SRT: {result.get('srt_path')}")


if __name__ == "__main__":
    asyncio.run(main())
