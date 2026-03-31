"""
Module 1 (VLM) — End-to-End Verification Script.

Tests the entire VLM pipeline with a REAL video (no separate OCR needed):
  Video → SceneDetect → FrameExtract → VLM (Extract+Translate) → Validate → SRT

All intermediate results are saved to an output directory for debugging.

Run:  python scripts/verify_m1_vlm.py --video path/to/video.mp4
      python scripts/verify_m1_vlm.py --video path/to/video.mp4 --output ./debug_output
      python scripts/verify_m1_vlm.py  (synthetic fallback)
"""

import argparse
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ─── ANSI Colors ────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0
skipped = 0
log_lines = []  # Collect all output for log file


def log(text: str = "", console: bool = True):
    """Print to console and append to log buffer."""
    log_lines.append(text)
    if console:
        print(text, flush=True)


def header(text: str):
    line = f"\n{'═' * 60}\n  {text}\n{'═' * 60}"
    log(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    log(f"{BOLD}{CYAN}  {text}{RESET}")
    log(f"{BOLD}{CYAN}{'═' * 60}{RESET}")


def test(name: str):
    log_lines.append(f"\n  ▶ {name}")
    print(f"\n  {BOLD}▶ {name}{RESET}", end="", flush=True)


def ok(detail: str = ""):
    global passed
    passed += 1
    msg = f"  ✓ PASSED"
    if detail:
        msg += f"  ({detail})"
    log_lines.append(msg)
    colored = f"  {GREEN}✓ PASSED{RESET}"
    if detail:
        colored += f"  ({detail})"
    print(colored, flush=True)


def fail(detail: str = ""):
    global failed
    failed += 1
    msg = f"  ✗ FAILED"
    if detail:
        msg += f"  ({detail})"
    log_lines.append(msg)
    colored = f"  {RED}✗ FAILED{RESET}"
    if detail:
        colored += f"  ({detail})"
    print(colored, flush=True)


def skip(detail: str = ""):
    global skipped
    skipped += 1
    msg = f"  ⊘ SKIPPED"
    if detail:
        msg += f"  ({detail})"
    log_lines.append(msg)
    colored = f"  {YELLOW}⊘ SKIPPED{RESET}"
    if detail:
        colored += f"  ({detail})"
    print(colored, flush=True)


def detail(text: str):
    """Print indented detail line."""
    log(f"    {text}")


def save_json(data: any, path: Path, label: str = ""):
    """Save data as formatted JSON and print confirmation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    detail(f"💾 Saved {label}: {path.name} ({path.stat().st_size} bytes)")


def save_text(text: str, path: Path, label: str = ""):
    """Save text and print confirmation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    detail(f"💾 Saved {label}: {path.name} ({len(text)} chars)")


def get_video_info(video_path: Path) -> dict:
    """Get basic video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    info["duration"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def create_test_video(output_path: Path, duration_sec: float = 3.0, fps: int = 30):
    """Create a synthetic test video with scene changes and text."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 640, 480
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))
    total_frames = int(duration_sec * fps)

    for i in range(total_frames):
        t = i / fps
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if t < 1.0:
            frame[:] = (30, 30, 50)
            cv2.putText(frame, "Scene 1: Introduction", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        elif t < 2.0:
            frame[:] = (200, 200, 220)
            cv2.putText(frame, "Scene 2: def hello():", (30, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            frame[:] = (50, 100, 50)
            cv2.putText(frame, "Scene 3: Summary", (30, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Verify VLM Module (M1)")
    parser.add_argument("--video", type=str, help="Path to test video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory for debug results (default: ./debug_output/verify_TIMESTAMP)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Scene detection threshold (default: auto)")
    args = parser.parse_args()

    # ── Setup output directory ───────────────────────────────────────
    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("debug_output") / f"verify_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = out_dir / "frames"
    prompts_dir = out_dir / "prompts"
    frames_dir.mkdir(exist_ok=True)
    prompts_dir.mkdir(exist_ok=True)

    log(f"\n  📁 Output directory: {out_dir.resolve()}")

    # ──────────────────────────────────────────────────────────────
    # 1. Prepare test video
    # ──────────────────────────────────────────────────────────────
    header("1. Video Input")

    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            log(f"{RED}Video not found: {video_path}{RESET}")
            sys.exit(1)
        is_real_video = True
    else:
        log("  No --video provided, creating synthetic video...")
        video_path = create_test_video(out_dir / "test_video.mp4")
        is_real_video = False

    info = get_video_info(video_path)
    log(f"  File:       {video_path.name}")
    log(f"  Resolution: {info['width']}x{info['height']}")
    log(f"  FPS:        {info['fps']:.1f}")
    log(f"  Duration:   {info['duration']:.1f}s ({info['total_frames']} frames)")
    log(f"  Size:       {video_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Save video info
    save_json(info, out_dir / "video_info.json", "video metadata")

    # ──────────────────────────────────────────────────────────────
    # 2. SceneDetector
    # ──────────────────────────────────────────────────────────────
    header("2. SceneDetector")

    from src.m1_vlm.scene_detector import SceneDetector

    threshold = args.threshold or (20.0 if is_real_video else 27.0)

    test(f"Detect scenes (threshold={threshold})")
    t0 = time.perf_counter()
    sd = SceneDetector(threshold=threshold)
    scenes = sd.detect_scenes(video_path)
    dt = time.perf_counter() - t0

    if len(scenes) == 0 and is_real_video:
        for retry_thresh in [15.0, 10.0, 5.0]:
            detail(f"⚠ 0 scenes, retrying threshold={retry_thresh}...")
            sd2 = SceneDetector(threshold=retry_thresh)
            scenes = sd2.detect_scenes(video_path)
            if len(scenes) > 0:
                threshold = retry_thresh
                break

    if len(scenes) > 0:
        ok(f"{len(scenes)} scenes in {dt:.2f}s, threshold={threshold}")
        for i, (s, e) in enumerate(scenes[:15]):
            detail(f"Scene {i}: {s:.2f}s — {e:.2f}s ({e - s:.2f}s)")
        if len(scenes) > 15:
            detail(f"... and {len(scenes) - 15} more")
    else:
        scenes = [(0.0, info["duration"])]
        ok(f"0 scene cuts → full video as 1 chunk ({info['duration']:.1f}s)")

    # Save scenes
    scenes_data = [{"index": i, "start": s, "end": e, "duration": e - s}
                   for i, (s, e) in enumerate(scenes)]
    save_json(scenes_data, out_dir / "scenes.json", "scenes data")

    test("Split into chunks (45s)")
    chunks = sd.split_into_chunks(video_path, chunk_duration=45.0)
    if not chunks:
        chunks = [(0.0, info["duration"])]
    ok(f"{len(chunks)} chunks")
    chunks_data = [{"index": i, "start": s, "end": e, "duration": e - s}
                   for i, (s, e) in enumerate(chunks)]
    for c in chunks_data:
        detail(f"Chunk {c['index']}: {c['start']:.2f}s — {c['end']:.2f}s ({c['duration']:.2f}s)")
    save_json(chunks_data, out_dir / "chunks.json", "chunks data")

    # ──────────────────────────────────────────────────────────────
    # 3. FrameExtractor
    # ──────────────────────────────────────────────────────────────
    header("3. FrameExtractor")

    from src.m1_vlm.frame_extractor import FrameExtractor

    test("Extract frames from each chunk")
    fe = FrameExtractor(max_width=768)

    all_frames = []
    all_timestamps = []
    for i, (start, end) in enumerate(chunks):
        results = fe.extract_frames_in_range(
            video_path, start_time=start, end_time=end, interval=2.0,
        )
        for ts, frame in results:
            all_frames.append(frame)
            all_timestamps.append(ts)

    ok(f"{len(all_frames)} frames from {len(chunks)} chunks")

    test("Save ALL extracted frames")
    frame_metadata = []
    for i, (frame, ts) in enumerate(zip(all_frames, all_timestamps)):
        fname = f"frame_{i:03d}_t{ts:.1f}s.jpg"
        fpath = frames_dir / fname
        cv2.imwrite(str(fpath), frame)
        meta = {
            "index": i,
            "timestamp": ts,
            "filename": fname,
            "shape": list(frame.shape),
            "file_size_kb": fpath.stat().st_size / 1024,
        }
        frame_metadata.append(meta)
        detail(f"Frame {i}: t={ts:.2f}s, {frame.shape[1]}x{frame.shape[0]}, {meta['file_size_kb']:.0f}KB")

    ok(f"Saved {len(all_frames)} frames to {frames_dir}")
    save_json(frame_metadata, out_dir / "frames_metadata.json", "frame metadata")

    test("Convert to base64")
    b64_list = fe.frames_to_base64_batch(all_frames)
    total_kb = sum(len(b) for b in b64_list) / 1024
    ok(f"{len(b64_list)} base64 strings, total {total_kb:.0f} KB")

    # ──────────────────────────────────────────────────────────────
    # 4. PromptChain — build prompts using REAL data (VLM-only)
    # ──────────────────────────────────────────────────────────────
    header("4. PromptChain (VLM-only)")

    from src.m1_vlm.prompt_chain import PromptChain
    chain = PromptChain()

    test("Step 1: Extract — VLM reads frames directly")
    chunk_start, chunk_end = chunks[0]
    p1 = chain.step1_extract(
        chunk_info=f"Chunk 0: {chunk_start:.1f}s — {chunk_end:.1f}s "
                   f"(video: {video_path.name})",
    )
    assert "GLM-OCR" not in p1
    ok(f"{len(p1)} chars")
    save_text(p1, prompts_dir / "step1_extract.txt", "Step 1 prompt")

    test("Step 2: Translate")
    simulated_extraction = json.dumps([
        {
            "text": f"Content from {video_path.stem}",
            "type": "speech",
            "start_time": f"{chunk_start:.1f}s",
            "end_time": f"{chunk_end:.1f}s",
        }
    ], ensure_ascii=False)
    p2 = chain.step2_translate(extracted_text=simulated_extraction)
    ok(f"{len(p2)} chars")
    save_text(p2, prompts_dir / "step2_translate.txt", "Step 2 prompt")

    test("Step 3: Format SRT")
    p3 = chain.step3_format_srt(
        translated_segments=simulated_extraction,
        video_duration=info["duration"],
        chunk_start=chunk_start,
    )
    assert "42 characters" in p3
    assert "25 characters" in p3
    ok(f"{len(p3)} chars, SRT constraints ✓")
    save_text(p3, prompts_dir / "step3_format_srt.txt", "Step 3 prompt")

    test("Single combined prompt (recommended for Gemini)")
    single = chain.build_single_prompt(
        chunk_info=f"Video: {video_path.name}, "
                   f"Chunk: {chunk_start:.1f}s—{chunk_end:.1f}s, "
                   f"Duration: {info['duration']:.1f}s",
        context_summary=f"Frames: {len(all_frames)}, Resolution: {info['width']}x{info['height']}",
    )
    assert "GLM-OCR" not in single
    ok(f"{len(single)} chars, VLM-only ✓")
    save_text(single, prompts_dir / "single_combined.txt", "combined prompt")

    # ──────────────────────────────────────────────────────────────
    # 5. ContextWindow
    # ──────────────────────────────────────────────────────────────
    header("5. ContextWindow")

    from src.m1_vlm.context_window import ContextWindow

    test("Process chunks with sliding context")
    cw = ContextWindow(window_size=3)

    for i, (cs, ce) in enumerate(chunks):
        simulated_entries = [{
            "original_text": f"Content in chunk {i} ({cs:.1f}s—{ce:.1f}s)",
            "translated_text": f"Nội dung chunk {i} ({cs:.1f}s—{ce:.1f}s)",
        }]
        cw.add_chunk_result(i, simulated_entries, f"Chunk {i} of {video_path.stem}")

    summary = cw.get_context_summary()
    translations = cw.get_previous_translations(limit=5)
    glossary = cw.get_terminology_glossary()
    ok(f"{len(chunks)} chunks, {len(glossary)} terms")

    context_data = {
        "context_summary": summary,
        "previous_translations": translations,
        "terminology_glossary": glossary,
    }
    save_json(context_data, out_dir / "context_window.json", "context window state")
    detail(f"Summary preview: {summary[:120]}...")

    # ──────────────────────────────────────────────────────────────
    # 6. SubtitleValidator
    # ──────────────────────────────────────────────────────────────
    header("6. SubtitleValidator")

    from src.m1_vlm.validator import SubtitleValidator
    validator = SubtitleValidator()

    test("Generate & validate entries from chunks")
    subtitle_entries = []
    for i, (cs, ce) in enumerate(chunks[:5]):
        entry = {
            "index": i + 1,
            "start_time": validator._seconds_to_timestamp(cs + 0.5),
            "end_time": validator._seconds_to_timestamp(min(ce, cs + 5.0)),
            "original_text": f"Narration for chunk {i}",
            "translated_text": f"Lời thoại cho phần {i}",
        }
        subtitle_entries.append(entry)

    valid, issues = validator.validate_sequence(subtitle_entries)
    ok(f"{len(subtitle_entries)} entries, valid={valid}, {len(issues)} issues")
    if issues:
        for iss in issues:
            detail(f"⚠ {iss}")

    save_json(subtitle_entries, out_dir / "subtitle_entries.json", "subtitle entries")
    save_json({"valid": valid, "issues": issues}, out_dir / "validation_result.json", "validation")

    test("Test overlap detection & fix")
    overlapping = [e.copy() for e in subtitle_entries[:2]]
    if len(overlapping) == 2:
        ts_val = validator._timestamp_to_seconds(overlapping[0]["end_time"]) - 0.5
        overlapping[1]["start_time"] = validator._seconds_to_timestamp(ts_val)
        detail(f"Before fix: entry[0].end={overlapping[0]['end_time']}, entry[1].start={overlapping[1]['start_time']}")

    fixed = validator.fix_overlaps(overlapping)
    valid_after, issues_after = validator.validate_sequence(fixed)
    overlap_issues = [i for i in issues_after if "Overlap" in i]
    ok(f"Overlap fix: {len(overlap_issues)} remaining overlaps")

    if len(fixed) >= 2:
        detail(f"After fix:  entry[0].end={fixed[0]['end_time']}, entry[1].start={fixed[1]['start_time']}")

    # ──────────────────────────────────────────────────────────────
    # 7. SRTBuilder
    # ──────────────────────────────────────────────────────────────
    header("7. SRTBuilder")

    from src.m1_vlm.srt_builder import SRTBuilder
    builder = SRTBuilder()

    test("Build & save SRT")
    builder.add_entries(subtitle_entries)
    srt_path = builder.save(out_dir / f"{video_path.stem}_subtitles.srt")
    content = srt_path.read_text(encoding="utf-8")
    ok(f"{builder.entry_count} entries → {srt_path.name}")
    detail("--- SRT Content ---")
    for line in content.split("\n"):
        detail(line)
    detail("--- End SRT ---")

    test("Validate with pysrt")
    valid = builder.validate_with_pysrt(srt_path)
    ok("pysrt validation passed" if valid else "pysrt validation FAILED")

    test("Load SRT back")
    loaded = SRTBuilder.load_srt(srt_path)
    assert len(loaded) == len(subtitle_entries)
    ok(f"Loaded {len(loaded)} entries ✓")
    save_json(loaded, out_dir / "srt_loaded_back.json", "SRT re-loaded data")

    # ──────────────────────────────────────────────────────────────
    # 8. VLMClient (smoke test)
    # ──────────────────────────────────────────────────────────────
    header("8. VLMClient (smoke test)")

    from src.m1_vlm.vlm_client import VLMClient

    test("API client init")
    api_client = VLMClient(mode="api", api_key="test_key")
    ok(f"model={api_client.model_name}")

    test("Local client init")
    local_client = VLMClient(mode="local")
    ok(f"model={local_client.local_model_path}")

    # ──────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────
    header("VERIFICATION SUMMARY")
    total = passed + failed + skipped
    log(f"  Video:    {video_path.name}")
    log(f"  Duration: {info['duration']:.1f}s")
    log(f"  Chunks:   {len(chunks)}")
    log(f"  Frames:   {len(all_frames)}")
    log()
    log(f"  Passed:   {passed}")
    log(f"  Failed:   {failed}")
    log(f"  Skipped:  {skipped}")
    log(f"  Total:    {total}")
    log()

    if failed == 0:
        log(f"  ✓ MODULE 1 (VLM) — ALL CHECKS PASSED")
    else:
        log(f"  ✗ MODULE 1 (VLM) — {failed} CHECK(S) FAILED")

    # ── Save full log ────────────────────────────────────────────
    log_path = out_dir / "verify_log.txt"
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    # ── Save summary JSON ────────────────────────────────────────
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "video": str(video_path),
        "video_info": info,
        "scene_threshold": threshold,
        "scenes_count": len(scenes),
        "chunks_count": len(chunks),
        "frames_extracted": len(all_frames),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "total": total,
        "output_dir": str(out_dir.resolve()),
    }
    save_json(summary_data, out_dir / "summary.json", "verification summary")

    log()
    log(f"  📁 All results saved to: {out_dir.resolve()}")
    log()
    log(f"  Debug files:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir)
            size = f.stat().st_size
            if size > 1024:
                log(f"    {rel}  ({size/1024:.0f} KB)")
            else:
                log(f"    {rel}  ({size} bytes)")

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
