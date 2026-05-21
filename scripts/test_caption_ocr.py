"""Smoke test: run CaptionTimeline on a single video, print + save segments.

Does NOT call the VLM or TTS — just sample → crop → OCR → segment → SRT.
Use this to validate that burned-in captions can be recovered before
running the full --mode ocr pipeline.

Usage:
    .\\.venv\\Scripts\\Activate.ps1
    python scripts/test_caption_ocr.py --video data/raw/sample.mp4
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m1_vlm.glm_ocr import GLMOCR
from src.m1_vlm.caption_ocr import CaptionTimeline


def _sec_to_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="CaptionTimeline smoke test")
    parser.add_argument("--video", "-v", required=True, help="Path to video")
    parser.add_argument("--fps", type=float, default=2.0, help="OCR sample fps")
    parser.add_argument("--band", type=float, default=0.22,
                        help="Bottom-crop ratio (0..1)")
    parser.add_argument("--dedup", type=float, default=0.85,
                        help="Similarity threshold")
    parser.add_argument("--min-dur", type=float, default=0.3,
                        help="Drop segments shorter than this (seconds)")
    parser.add_argument("--ocr-model", default="zai-org/GLM-OCR",
                        help="GLM-OCR model id or path")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: ./output/caption_ocr_<stem>)")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path("output") / f"caption_ocr_{video.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = GLMOCR(model_path=args.ocr_model, device="cuda")
    tl = CaptionTimeline(
        ocr=ocr,
        sample_fps=args.fps,
        caption_band_ratio=args.band,
        dedup_ratio=args.dedup,
        min_duration_sec=args.min_dur,
    )
    segments = tl.build(video)

    print(f"\nDetected {len(segments)} caption segments:\n")
    for i, s in enumerate(segments):
        print(f"  [{i:03d}] {s.start_sec:7.2f}s → {s.end_sec:7.2f}s  "
              f"({s.duration_sec:5.2f}s)  {s.en_text!r}")

    manifest = [
        {"idx": i, "start_sec": s.start_sec, "end_sec": s.end_sec, "en_text": s.en_text}
        for i, s in enumerate(segments)
    ]
    (out_dir / "segments.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    srt_path = out_dir / f"{video.stem}_en_captions.srt"
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_sec_to_srt(s.start_sec)} --> {_sec_to_srt(s.end_sec)}")
        lines.append(s.en_text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote: {out_dir/'segments.json'}")
    print(f"Wrote: {srt_path}")


if __name__ == "__main__":
    main()
