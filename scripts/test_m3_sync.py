"""
M3 Sync standalone validator.

Loads M1 SRT + M2 audio chunks, exercises AudioAligner → TimeStretcher → FFmpegRenderer
in isolation so M3 bugs surface independently of the pipeline orchestrator.

Usage:
    python scripts/test_m3_sync.py \
        --srt output/Demo-Module-5/Demo-Module-5_vi.srt \
        --audio-dir debug_output/m1_m2_integration/audio_chunks \
        --video "H:\\Quan\\tanquan\\VNext\\ToyoBeauty\\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" \
        --output-dir debug_output/m3_test_run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.m1_vlm.srt_builder import SRTBuilder
from src.m3_sync.audio_aligner import AudioAligner
from src.m3_sync.ffmpeg_renderer import FFmpegRenderer
from src.m3_sync.time_stretcher import TimeStretcher


def load_segments(srt_path: Path, audio_dir: Path) -> list[dict]:
    """Pair SRT entries with audio chunk files by ordinal index."""
    entries = SRTBuilder.load_srt(srt_path)
    audio_files = sorted(audio_dir.glob("chunk_*.wav"))

    if len(entries) != len(audio_files):
        raise RuntimeError(
            f"Mismatch: {len(entries)} SRT entries vs {len(audio_files)} audio files. "
            f"M1+M2 outputs must produce the same count."
        )

    segments = []
    for entry, audio_path in zip(entries, audio_files):
        seg = dict(entry)
        seg["audio_path"] = str(audio_path)
        segments.append(seg)
    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Standalone M3 (AudioAligner + TimeStretcher + FFmpegRenderer) validator",
    )
    parser.add_argument("--srt", type=Path, required=True,
                        help="Path to Vietnamese SRT from M1")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Directory of chunk_NNNN.wav files from M2")
    parser.add_argument("--video", type=Path, required=True,
                        help="Original video file")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory for intermediate + final outputs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"SRT       : {args.srt}")
    logger.info(f"Audio dir : {args.audio_dir}")
    logger.info(f"Video     : {args.video}")
    logger.info(f"Output    : {args.output_dir}")

    # Stage 0: Load and pair
    segments = load_segments(args.srt, args.audio_dir)
    logger.info(f"Loaded {len(segments)} segments")

    # Stage 1: AudioAligner.calculate_deltas
    stretcher = TimeStretcher()
    aligner = AudioAligner(time_stretcher=stretcher)
    with_deltas = aligner.calculate_deltas(segments)

    logger.info("=== Per-segment deltas ===")
    for seg in with_deltas:
        idx = seg.get("index", "?")
        logger.info(
            f"#{idx:>2} target={seg['target_duration']:.2f}s "
            f"audio={seg['audio_duration']:.2f}s delta={seg['delta']:+.2f}s "
            f"strategy={seg['strategy']}"
        )

    report_path = args.output_dir / "stage1_deltas.json"
    report_path.write_text(
        json.dumps(with_deltas, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Stage 1 report saved: {report_path}")

    # Stage 2: AudioAligner.align_all (calls TimeStretcher.stretch_to_fit per segment)
    aligned_dir = args.output_dir / "aligned_audio"
    aligned = aligner.align_all(with_deltas, output_dir=aligned_dir)

    logger.info("=== Alignment strategies used ===")
    method_counts: dict[str, int] = {}
    for seg in aligned:
        method = seg.get("align_method", "exact")
        method_counts[method] = method_counts.get(method, 0) + 1
    for method, n in method_counts.items():
        logger.info(f"  {method}: {n}")

    # Stage 3: FFmpegRenderer.merge_audio_segments
    renderer = FFmpegRenderer()
    video_info = FFmpegRenderer.get_video_info(args.video)
    duration = float(video_info["format"]["duration"])
    logger.info(f"Source video duration: {duration:.2f}s")

    merged_audio = args.output_dir / "merged_track.wav"
    renderer.merge_audio_segments(
        aligned,
        total_duration=duration,
        output_path=merged_audio,
        sample_rate=24000,  # match VieNeu TTS native rate
    )

    # Stage 4: Final mux
    final_video = args.output_dir / f"{args.video.stem}_vi.mp4"
    renderer.render_final_video(
        video_path=args.video,
        dubbed_audio_path=merged_audio,
        output_path=final_video,
        keep_original_audio=False,
    )

    logger.info("=== Done ===")
    logger.info(f"Merged audio : {merged_audio}")
    logger.info(f"Final video  : {final_video}")


if __name__ == "__main__":
    main()
