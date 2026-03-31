"""
Extract subtitles from a video using VLM (local Qwen2.5-VL or Gemini API).

This is the REAL VLM pipeline — actually sends frames to the model
and gets extracted/translated text back.

Usage:
    # Local Qwen2.5-VL
    python scripts/run_vlm_extract.py --video path/to/video.mp4

    # Gemini API
    python scripts/run_vlm_extract.py --video path/to/video.mp4 --api --api-key YOUR_KEY
    python scripts/run_vlm_extract.py --video path/to/video.mp4 --api  (reads from .env)
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Extract subtitles from video using VLM",
    )
    parser.add_argument("--video", "-v", type=str, required=True,
                        help="Path to video file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: ./output/TIMESTAMP)")
    # Backend selection
    parser.add_argument("--api", action="store_true",
                        help="Use Gemini API instead of local model")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY in .env)")
    parser.add_argument("--api-model", type=str, default="gemini-2.5-flash-lite",
                        help="Gemini model name (default: gemini-2.5-flash-lite)")
    parser.add_argument("--model", type=str, default="./models/qwen2.5-vl-3b",
                        help="Local model path (default: ./models/qwen2.5-vl-3b)")
    # Processing options
    parser.add_argument("--mode", choices=["single", "3step"], default="single",
                        help="Prompt mode: 'single' (all-in-one) or '3step' (chained)")
    parser.add_argument("--no-chunk", action="store_true",
                        help="Do NOT split video into chunks. Process entire video in one pass.")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Frame extraction interval in seconds (default: 2.0)")
    parser.add_argument("--chunk-duration", type=float, default=45.0,
                        help="Max chunk duration in seconds (default: 45)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames per VLM call (default: 4 for API, 16 for local, 32 for local no-chunk)")
    parser.add_argument("--threshold", type=float, default=20.0,
                        help="Scene detection threshold (default: 20.0)")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay between chunks in seconds (default: 30 for free API, 0 for local)")
    args = parser.parse_args()

    # Set smart defaults based on backend and chunking mode
    if args.max_frames is None:
        if args.api:
            args.max_frames = 2  # Free tier: ~2 images/min limit
        elif args.no_chunk:
            args.max_frames = 32  # More frames when processing entire video
        else:
            args.max_frames = 8
    if args.delay is None:
        args.delay = 30.0 if args.api else 0.0

    # Resolve API key
    api_key = args.api_key
    if args.api and not api_key:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("❌ Gemini API key required. Use --api-key YOUR_KEY or set GEMINI_API_KEY in .env")
            sys.exit(1)

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    # Setup output
    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output") / f"extract_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)

    backend_name = f"Gemini API ({args.api_model})" if args.api else f"Local ({args.model})"
    chunk_mode = "NO CHUNK (single pass)" if args.no_chunk else f"chunked ({args.chunk_duration}s)"
    print(f"{'═' * 60}")
    print(f"  VLM SUBTITLE EXTRACTION")
    print(f"{'═' * 60}")
    print(f"  Video:    {video_path.name}")
    print(f"  Backend:  {backend_name}")
    print(f"  Mode:     {args.mode}")
    print(f"  Chunking: {chunk_mode}")
    print(f"  Max frm:  {args.max_frames}")
    print(f"  Output:   {out_dir.resolve()}")
    print()

    # ── Step 1: Analyze video ────────────────────────────────────
    print("▶ Step 1: Analyzing video...")
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print(f"  Resolution: {width}x{height}, FPS: {fps:.1f}, Duration: {duration:.1f}s")

    # ── Step 2: Scene detection & chunking ────────────────────────
    if args.no_chunk:
        # ── No chunking: process entire video as one pass ──
        print("\n▶ Step 2: Skipping scene detection (--no-chunk mode)...")
        chunks = [(0.0, duration)]
        print(f"  Single pass: 0.0s — {duration:.1f}s ({duration:.1f}s)")
    else:
        # ── Scene detection & chunking ──
        print("\n▶ Step 2: Detecting scenes...")
        from src.m1_vlm.scene_detector import SceneDetector

        sd = SceneDetector(threshold=args.threshold)
        scenes = sd.detect_scenes(video_path)

        if len(scenes) == 0:
            # Retry with lower thresholds
            for t in [15.0, 10.0, 5.0]:
                sd2 = SceneDetector(threshold=t)
                scenes = sd2.detect_scenes(video_path)
                if scenes:
                    print(f"  Found {len(scenes)} scenes (threshold={t})")
                    break

        if not scenes:
            scenes = [(0.0, duration)]
            print(f"  No scene cuts → 1 chunk")

        chunks = sd.split_into_chunks(video_path, chunk_duration=args.chunk_duration)
        if not chunks:
            chunks = [(0.0, duration)]

        print(f"  {len(scenes)} scenes → {len(chunks)} chunks")
        for i, (s, e) in enumerate(chunks):
            print(f"    Chunk {i}: {s:.1f}s — {e:.1f}s ({e-s:.1f}s)")

    # ── Step 3: Extract frames ───────────────────────────────────
    print("\n▶ Step 3: Extracting frames...")
    from src.m1_vlm.frame_extractor import FrameExtractor

    fe = FrameExtractor(max_width=768)

    chunk_frames = {}  # chunk_index -> [(timestamp, frame, base64)]
    total_extracted = 0

    for ci, (start, end) in enumerate(chunks):
        results = fe.extract_frames_in_range(
            video_path, start_time=start, end_time=end, interval=args.interval,
        )
        # Limit frames per chunk
        if len(results) > args.max_frames:
            # Sample evenly
            step = len(results) / args.max_frames
            results = [results[int(i * step)] for i in range(args.max_frames)]

        frames_data = []
        for ts, frame in results:
            b64 = fe.frame_to_base64(frame)
            frames_data.append((ts, frame, b64))
            # Save frame
            fname = f"chunk{ci}_t{ts:.1f}s.jpg"
            cv2.imwrite(str(out_dir / "frames" / fname), frame)

        chunk_frames[ci] = frames_data
        total_extracted += len(frames_data)
        print(f"  Chunk {ci}: {len(frames_data)} frames")

    print(f"  Total: {total_extracted} frames extracted & saved")

    # ── Step 4: Initialize VLM ───────────────────────────────────
    print(f"\n▶ Step 4: Initializing VLM ({backend_name})...")
    from src.m1_vlm.vlm_client import VLMClient

    t0 = time.perf_counter()

    if args.api:
        vlm = VLMClient(
            mode="api",
            model_name=args.api_model,
            api_key=api_key,
            temperature=0.1,
            max_tokens=4096,
        )
        vlm._init_api_client()
        load_time = time.perf_counter() - t0
        print(f"  Gemini API ready in {load_time:.1f}s (model: {args.api_model})")
    else:
        vlm = VLMClient(
            mode="local",
            local_model_path=args.model,
            temperature=0.1,
            max_tokens=4096,
        )
        vlm._init_local_client()
        load_time = time.perf_counter() - t0
        import torch
        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        print(f"  Loaded in {load_time:.1f}s, VRAM: {vram:.2f} GB")

    # ── Step 5: Run VLM inference on each chunk ──────────────────
    print(f"\n▶ Step 5: Running VLM inference ({args.mode} mode)...")
    from src.m1_vlm.prompt_chain import PromptChain
    from src.m1_vlm.context_window import ContextWindow

    chain = PromptChain()
    cw = ContextWindow(window_size=3)
    all_results = []

    for ci, (start, end) in enumerate(chunks):
        # Delay between API calls to avoid rate limiting
        if args.api and ci > 0 and args.delay > 0:
            print(f"\n    ⏳ Waiting {args.delay:.0f}s before next chunk (rate limit)...",
                  end="", flush=True)
            time.sleep(args.delay)
            print(" ready", flush=True)

        frames_data = chunk_frames[ci]
        if not frames_data:
            continue

        b64_images = [b64 for _, _, b64 in frames_data]
        timestamps = [ts for ts, _, _ in frames_data]

        print(f"\n  ── Chunk {ci}/{len(chunks)-1}: {start:.1f}s — {end:.1f}s "
              f"({len(b64_images)} frames) ──")

        chunk_info = (
            f"Chunk {ci}: {start:.1f}s — {end:.1f}s\n"
            f"Video: {video_path.name} (total {duration:.1f}s)\n"
            f"Frames at timestamps: {[f'{t:.1f}s' for t in timestamps]}"
        )

        t0 = time.perf_counter()

        if args.mode == "single":
            # All-in-one prompt
            prompt = chain.build_single_prompt(
                chunk_info=chunk_info,
                context_summary=cw.get_context_summary() or None,
                previous_translations=cw.get_previous_translations(limit=3) or None,
            )

            raw_response = asyncio.run(
                vlm.generate(prompt=prompt, images_base64=b64_images)
            )

        else:
            # 3-step chain
            # Step 1: Extract
            p1 = chain.step1_extract(
                chunk_info=chunk_info,
                context_summary=cw.get_context_summary() or None,
            )
            raw_step1 = asyncio.run(
                vlm.generate(prompt=p1, images_base64=b64_images)
            )
            print(f"    Step 1 (extract): {len(raw_step1)} chars")

            # Step 2: Translate
            p2 = chain.step2_translate(
                extracted_text=raw_step1,
                context_summary=cw.get_context_summary() or None,
                previous_translations=cw.get_previous_translations(limit=3) or None,
            )
            raw_step2 = asyncio.run(
                vlm.generate(prompt=p2)  # No images needed for translation
            )
            print(f"    Step 2 (translate): {len(raw_step2)} chars")

            # Step 3: Format SRT
            p3 = chain.step3_format_srt(
                translated_segments=raw_step2,
                video_duration=duration,
                chunk_start=start,
            )
            raw_response = asyncio.run(
                vlm.generate(prompt=p3)  # No images needed
            )
            print(f"    Step 3 (SRT): {len(raw_response)} chars")

        infer_time = time.perf_counter() - t0

        # Parse response
        print(f"    Inference time: {infer_time:.1f}s")
        print(f"    Raw response ({len(raw_response)} chars):")

        # Save raw response
        raw_path = out_dir / f"chunk_{ci:02d}_raw_response.txt"
        raw_path.write_text(raw_response, encoding="utf-8")

        # Try to parse JSON
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        try:
            parsed = json.loads(cleaned.strip())
            print(f"    ✓ Parsed: {len(parsed)} subtitle entries")
            for entry in parsed:
                orig = entry.get("original_text", "")[:60]
                trans = entry.get("translated_text", "")[:60]
                st = entry.get("start_time", "?")
                et = entry.get("end_time", "?")
                print(f"      [{st} → {et}]")
                print(f"        EN: {orig}")
                print(f"        VI: {trans}")

            all_results.append({
                "chunk_index": ci,
                "chunk_start": start,
                "chunk_end": end,
                "entries": parsed,
                "inference_time": infer_time,
            })

            # Feed to context window
            cw.add_chunk_result(ci, parsed, f"Chunk {ci}: {video_path.stem}")

        except json.JSONDecodeError as e:
            print(f"    ⚠ JSON parse failed: {e}")
            print(f"    Response preview: {cleaned[:200]}...")
            all_results.append({
                "chunk_index": ci,
                "chunk_start": start,
                "chunk_end": end,
                "raw_response": raw_response,
                "parse_error": str(e),
                "inference_time": infer_time,
            })

    # ── Step 6: Validate & Build SRT ─────────────────────────────
    print(f"\n▶ Step 6: Building SRT file...")
    from src.m1_vlm.validator import SubtitleValidator
    from src.m1_vlm.srt_builder import SRTBuilder

    validator = SubtitleValidator()
    builder = SRTBuilder()

    total_entries = 0
    for result in all_results:
        entries = result.get("entries", [])
        if not entries:
            continue

        # Validate
        valid, issues = validator.validate_sequence(entries)
        if not valid:
            print(f"  Chunk {result['chunk_index']}: {len(issues)} validation issues")
            for iss in issues[:3]:
                print(f"    ⚠ {iss}")
            # Try to fix overlaps
            entries = validator.fix_overlaps(entries)
            entries = validator.reindex(entries)

        builder.add_entries(entries)
        total_entries += len(entries)

    if total_entries > 0:
        srt_path = builder.save(out_dir / f"{video_path.stem}.srt")
        print(f"  ✓ SRT saved: {srt_path.name} ({total_entries} entries)")
        print()
        # Print full SRT content
        srt_content = srt_path.read_text(encoding="utf-8")
        print(f"  ── SRT Content ──")
        for line in srt_content.split("\n"):
            print(f"  {line}")
    else:
        print("  ⚠ No subtitle entries generated")

    # ── Save all results ─────────────────────────────────────────
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "video": str(video_path),
        "model": args.api_model if args.api else args.model,
        "backend": "gemini_api" if args.api else "local",
        "mode": args.mode,
        "duration": duration,
        "chunks": len(chunks),
        "total_frames": total_extracted,
        "total_entries": total_entries,
        "results": all_results,
    }
    results_path = out_dir / "extraction_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # Cleanup VLM
    vlm.unload_model()

    print(f"\n{'═' * 60}")
    print(f"  DONE")
    print(f"{'═' * 60}")
    print(f"  Video:    {video_path.name} ({duration:.1f}s)")
    print(f"  Chunks:   {len(chunks)}")
    print(f"  Frames:   {total_extracted}")
    print(f"  Entries:  {total_entries}")
    print(f"  Output:   {out_dir.resolve()}")
    print()
    print(f"  Files:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir)
            sz = f.stat().st_size
            print(f"    {rel}  ({sz/1024:.0f} KB)" if sz > 1024 else f"    {rel}  ({sz} B)")


if __name__ == "__main__":
    main()
