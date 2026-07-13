"""
Full Pipeline: Video → VLM Subtitle Extraction → Voice Cloning.

Phase 1: VLM (Module 1) — Extract & translate narration from video frames.
Phase 2: TTS (Module 2) — Synthesize Vietnamese audio for each subtitle segment.

Two voice modes:
  --voice-mode preset   Use a built-in VieNeu preset voice (no reference audio).
  --voice-mode clone    Zero-shot voice cloning from a reference audio file (3–10s).

Usage:
    # Preset voice, local Qwen3.5-2B (default)
    python scripts/run_tts_voice_cloning.py --video data/raw/sample.mp4 --voice-mode preset

    # Voice cloning from reference audio, Gemini API
    python scripts/run_tts_voice_cloning.py \\
        --video data/raw/sample.mp4 --voice-mode clone \\
        --ref-audio data/reference_audio/speaker.wav --api

    # List preset voices and exit
    python scripts/run_tts_voice_cloning.py --list-voices

    # Skip VLM phase — reuse existing extraction_results.json
    python scripts/run_tts_voice_cloning.py \\
        --skip-vlm --input output/extract_20260423_120000 \\
        --voice-mode clone --ref-audio speaker.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Fix Windows cp1252 console — Vietnamese chars in vieneu SDK logs
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ===================================================================== #
#  Phase 1 — VLM Subtitle Extraction (reuses run_vlm_extract.py logic)  #
# ===================================================================== #

def run_vlm_phase(args) -> tuple[list[dict], Path, dict]:
    """
    Run full VLM extraction pipeline on the video.

    Returns:
        (segments, out_dir, metadata)
        - segments: flat list of {translated_text, start_time, end_time, original_text}
        - out_dir: output directory for this run
        - metadata: video info dict (duration, fps, resolution)
    """
    import cv2
    from src.m1_vlm.scene_detector import SceneDetector
    from src.m1_vlm.frame_extractor import FrameExtractor
    from src.m1_vlm.prompt_chain import PromptChain
    from src.m1_vlm.context_window import ContextWindow

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        sys.exit(1)

    # ── Video analysis ──────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print(f"  PHASE 1: VLM SUBTITLE EXTRACTION")
    print(f"{'─' * 65}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    backend_name = f"Gemini API ({args.api_model})" if args.api else f"Local ({args.model})"
    print(f"  Video:    {video_path.name} ({width}x{height}, {fps:.1f}fps, {duration:.1f}s)")
    print(f"  Backend:  {backend_name}")
    print(f"  Mode:     {args.mode}")

    # ── Output dir ──────────────────────────────────────────────
    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("output") / f"pipeline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)

    # ── Scene detection & chunking ──────────────────────────────
    fe = FrameExtractor(max_width=768)
    chain = PromptChain()
    cw = ContextWindow(window_size=3)

    if args.no_chunk:
        chunks = [(0.0, duration)]
        sd = None
        print(f"  Chunks:   single pass (0.0s — {duration:.1f}s)")
    else:
        print(f"  Detecting scenes (threshold={args.threshold})...")
        sd = SceneDetector(threshold=args.threshold)
        scenes = sd.detect_scenes(video_path)

        if not scenes:
            for t in [15.0, 10.0, 5.0]:
                sd = SceneDetector(threshold=t)
                scenes = sd.detect_scenes(video_path)
                if scenes:
                    break
        if not scenes:
            scenes = [(0.0, duration)]

        chunks = sd.split_into_chunks(
            video_path, chunk_duration=args.chunk_duration, overlap=args.overlap,
        )
        if not chunks:
            chunks = [(0.0, duration)]

        print(f"  {len(scenes)} scenes -> {len(chunks)} chunks (overlap={args.overlap}s)")
        for ci, (s, e) in enumerate(chunks):
            print(f"    Chunk {ci}: {s:.1f}s — {e:.1f}s ({e - s:.1f}s)")

    # ── Global Summary Pass ─────────────────────────────────────
    global_summary_text = None
    vlm_initialized = False

    if not args.no_global_summary:
        print(f"\n  Global Summary Pass ({args.global_frames} frames)...")
        t0 = time.perf_counter()

        global_frames = fe.extract_frames_evenly(video_path, n=args.global_frames)
        if global_frames:
            global_images_b64 = fe.frames_to_base64_batch(global_frames)
            for idx, frame in enumerate(global_frames):
                cv2.imwrite(str(out_dir / "frames" / f"global_frame_{idx:02d}.jpg"), frame)

            if len(global_images_b64) > args.max_frames:
                step = len(global_images_b64) / args.max_frames
                global_images_b64 = [global_images_b64[int(i * step)] for i in range(args.max_frames)]

            global_prompt = chain.build_global_summary_prompt()
            from src.m1_vlm.vlm_client import VLMClient

            if args.api:
                vlm = VLMClient(mode="api", model_name=args.api_model,
                                api_key=args.api_key, temperature=0.1, max_tokens=4096)
                vlm._init_api_client()
            else:
                vlm = VLMClient(mode="local", local_model_path=args.model,
                                temperature=0.1, max_tokens=4096)
                vlm._init_local_client()

            raw_global = asyncio.run(vlm.generate(prompt=global_prompt, images_base64=global_images_b64))
            (out_dir / "global_summary_raw.txt").write_text(raw_global, encoding="utf-8")

            cleaned = raw_global.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]

            try:
                global_parsed = json.loads(cleaned.strip())
                global_summary_text = json.dumps(global_parsed, indent=2, ensure_ascii=False)
                cw.set_global_summary(global_summary_text)
                elapsed = time.perf_counter() - t0
                print(f"    Done in {elapsed:.1f}s — topic: {global_parsed.get('topic', 'N/A')}")
                (out_dir / "global_summary.json").write_text(global_summary_text, encoding="utf-8")
            except json.JSONDecodeError:
                global_summary_text = raw_global
                cw.set_global_summary(global_summary_text)
                print(f"    JSON parse failed, using raw text")

            vlm_initialized = True

    # ── Frame extraction ────────────────────────────────────────
    print(f"\n  Extracting frames ({'ADAPTIVE' if not args.legacy else 'FIXED'})...")

    dedup = None
    if not args.no_dedup:
        from src.m1_vlm.frame_dedup import FrameDeduplicator
        dedup = FrameDeduplicator(ssim_threshold=args.ssim_threshold)

    chunk_frames: dict[int, list] = {}
    total_extracted = 0

    for ci, (start, end) in enumerate(chunks):
        if args.legacy or sd is None:
            results = fe.extract_frames_in_range(video_path, start_time=start, end_time=end, interval=args.interval)
        else:
            timestamps = sd.get_adaptive_timestamps(video_path, start, end,
                                                     min_frames=3, max_frames=args.max_frames * 2)
            results = fe.extract_frames_at_timestamps(video_path, timestamps)

        if dedup and len(results) > 1:
            results = dedup.deduplicate(results)
        if len(results) > args.max_frames:
            step = len(results) / args.max_frames
            results = [results[int(i * step)] for i in range(args.max_frames)]

        frames_data = []
        for ts, frame in results:
            b64 = fe.frame_to_base64(frame)
            frames_data.append((ts, frame, b64))
            cv2.imwrite(str(out_dir / "frames" / f"chunk{ci}_t{ts:.1f}s.jpg"), frame)

        chunk_frames[ci] = frames_data
        total_extracted += len(frames_data)
        print(f"    Chunk {ci}: {len(frames_data)} frames")

    print(f"    Total: {total_extracted} frames")

    # ── VLM inference ───────────────────────────────────────────
    if not vlm_initialized:
        from src.m1_vlm.vlm_client import VLMClient
        if args.api:
            vlm = VLMClient(mode="api", model_name=args.api_model,
                            api_key=args.api_key, temperature=0.1, max_tokens=4096)
            vlm._init_api_client()
        else:
            vlm = VLMClient(mode="local", local_model_path=args.model,
                            temperature=0.1, max_tokens=4096)
            vlm._init_local_client()

    print(f"\n  Running VLM inference ({args.mode} mode)...")

    all_results: list[dict] = []

    for ci, (start, end) in enumerate(chunks):
        if args.api and ci > 0 and args.delay > 0:
            print(f"    Waiting {args.delay:.0f}s (rate limit)...", end="", flush=True)
            time.sleep(args.delay)
            print(" ready", flush=True)

        frames_data = chunk_frames.get(ci, [])
        if not frames_data:
            continue

        b64_images = [b64 for _, _, b64 in frames_data]
        timestamps = [ts for ts, _, _ in frames_data]

        print(f"    Chunk {ci}/{len(chunks) - 1}: {start:.1f}s — {end:.1f}s ({len(b64_images)} frames)")

        chunk_info = (f"Chunk {ci}: {start:.1f}s — {end:.1f}s\n"
                      f"Video: {video_path.name} (total {duration:.1f}s)\n"
                      f"Frames at timestamps: {[f'{t:.1f}s' for t in timestamps]}")

        t0 = time.perf_counter()

        if args.mode == "single":
            prompt = chain.build_single_prompt(
                chunk_info=chunk_info,
                context_summary=cw.get_context_summary() or None,
                previous_translations=cw.get_previous_translations(limit=10) or None,
                global_context=cw.get_global_summary(),
                frame_timestamps=timestamps,
                chunk_start=start, chunk_end=end,
                chunk_index=ci, total_chunks=len(chunks),
            )
            raw_response = asyncio.run(vlm.generate(prompt=prompt, images_base64=b64_images))
        else:
            p1 = chain.step1_extract(chunk_info=chunk_info,
                                     context_summary=cw.get_context_summary() or None,
                                     global_context=cw.get_global_summary())
            raw_step1 = asyncio.run(vlm.generate(prompt=p1, images_base64=b64_images))

            p2 = chain.step2_translate(extracted_text=raw_step1,
                                       context_summary=cw.get_context_summary() or None,
                                       previous_translations=cw.get_previous_translations(limit=3) or None)
            raw_step2 = asyncio.run(vlm.generate(prompt=p2))

            p3 = chain.step3_format_srt(translated_segments=raw_step2,
                                        video_duration=duration, chunk_start=start)
            raw_response = asyncio.run(vlm.generate(prompt=p3))

        infer_time = time.perf_counter() - t0
        (out_dir / f"chunk_{ci:02d}_raw_response.txt").write_text(raw_response, encoding="utf-8")

        # Parse
        cleaned = raw_response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        try:
            parsed = json.loads(cleaned.strip())
            print(f"      {len(parsed)} entries ({infer_time:.1f}s)")
            for entry in parsed:
                print(f"        [{entry.get('start_time', '?')} -> {entry.get('end_time', '?')}] "
                      f"{entry.get('translated_text', '')[:60]}")
            all_results.append({"chunk_index": ci, "chunk_start": start,
                                "chunk_end": end, "entries": parsed,
                                "inference_time": infer_time})
            cw.add_chunk_result(ci, parsed, f"Chunk {ci}: {video_path.stem}")
        except json.JSONDecodeError as e:
            print(f"      JSON parse failed: {e}")
            all_results.append({"chunk_index": ci, "chunk_start": start,
                                "chunk_end": end, "raw_response": raw_response,
                                "parse_error": str(e), "inference_time": infer_time})

    # ── Build SRT ───────────────────────────────────────────────
    print(f"\n  Building SRT...")
    from src.m1_vlm.validator import SubtitleValidator
    from src.m1_vlm.srt_builder import SRTBuilder

    validator = SubtitleValidator()
    builder = SRTBuilder()
    total_entries = 0

    for result in all_results:
        entries = result.get("entries", [])
        if not entries:
            continue
        valid, issues = validator.validate_sequence(entries)
        if not valid:
            entries = validator.fix_overlaps(entries)
            entries = validator.reindex(entries)
        builder.add_entries(entries, chunk_offset=0.0)
        total_entries += len(entries)

    if total_entries > 0:
        builder.deduplicate_entries(overlap_tolerance=0.5)
        total_entries = builder.entry_count
        srt_path = builder.save(out_dir / f"{video_path.stem}.srt")
        print(f"    SRT saved: {srt_path.name} ({total_entries} entries)")
    else:
        srt_path = None
        print("    No subtitle entries generated")

    # ── Flatten segments ────────────────────────────────────────
    segments: list[dict] = []
    for result in all_results:
        for entry in result.get("entries", []):
            text = entry.get("translated_text", "").strip()
            if not text:
                continue
            segments.append({
                "translated_text": text,
                "start_time": entry.get("start_time", 0),
                "end_time": entry.get("end_time", 0),
                "original_text": entry.get("original_text", ""),
            })

    # Save extraction results
    save_data = {
        "timestamp": datetime.now().isoformat(),
        "video": str(video_path),
        "backend": "gemini_api" if args.api else "local",
        "mode": args.mode,
        "duration": duration,
        "total_entries": total_entries,
        "results": all_results,
    }
    with open(out_dir / "extraction_results.json", "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # Unload VLM — free VRAM before loading TTS model
    vlm.unload_model()
    print(f"    VLM unloaded, VRAM freed")

    metadata = {"duration": duration, "fps": fps, "width": width, "height": height}
    return segments, out_dir, metadata


# ===================================================================== #
#  Phase 2 — Voice Cloning (Module 2)                                   #
# ===================================================================== #

def run_tts_phase(segments: list[dict], out_dir: Path, args) -> None:
    """Synthesize audio for all subtitle segments."""
    import soundfile as sf
    from src.m2_tts.tts_client import TTSClient

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    client = TTSClient(
        engine=args.engine,
        vieneu_mode=args.vieneu_mode,
        backbone_device=args.device,
    )

    print(f"\n{'─' * 65}")
    print(f"  PHASE 2: VOICE CLONING ({args.voice_mode.upper()})")
    print(f"{'─' * 65}")
    print(f"  Engine:   {args.engine} ({args.vieneu_mode})")
    print(f"  Device:   {args.device}")
    print(f"  Segments: {len(segments)}")

    if args.voice_mode == "clone":
        print(f"  Ref audio: {args.ref_audio}")
    else:
        # Resolve voice_id
        voice_id = args.voice_id
        if not voice_id:
            print(f"  Loading model to find preset voices...")
            voices = client.list_preset_voices()
            voice_id = voices[0][1]
            print(f"  Voice:    {voice_id} (first available of {len(voices)})")
        else:
            print(f"  Voice:    {voice_id}")

    # ── Synthesize ──────────────────────────────────────────────
    success_count = 0
    fail_count = 0
    total_audio_duration = 0.0
    max_text_length = args.max_text_length

    if args.voice_mode == "clone":
        # Encode reference audio once
        print(f"\n  Encoding reference voice...")
        t0 = time.perf_counter()
        ref_codes = client.encode_reference(args.ref_audio, use_cache=True)
        print(f"    Done in {time.perf_counter() - t0:.2f}s")

        print(f"\n  Synthesizing {len(segments)} segments (cloned voice)...")

        for i, seg in enumerate(segments):
            text = seg["translated_text"]
            if len(text) > max_text_length:
                print(f"    [{i + 1}/{len(segments)}] Truncating {len(text)} -> {max_text_length} chars")
                text = text[:max_text_length]

            chunk_path = audio_dir / f"chunk_{i:04d}.wav"
            t_seg = time.perf_counter()

            try:
                client.synthesize_to_file(
                    text=text, output_path=chunk_path,
                    ref_codes=ref_codes, ref_text=args.ref_text,
                )
                seg_time = time.perf_counter() - t_seg
                audio_dur = sf.info(str(chunk_path)).duration
                total_audio_duration += audio_dur
                success_count += 1

                seg["audio_path"] = str(chunk_path)
                seg["audio_duration"] = round(audio_dur, 3)
                print(f"    [{i + 1}/{len(segments)}] {seg['start_time']} -> {seg['end_time']}  "
                      f"({audio_dur:.1f}s in {seg_time:.1f}s)  {text[:50]}...")
            except Exception as exc:
                fail_count += 1
                seg["audio_path"] = None
                seg["error"] = str(exc)
                print(f"    [{i + 1}/{len(segments)}] FAILED: {exc}")

    else:
        # Preset voice
        print(f"\n  Synthesizing {len(segments)} segments (preset: {voice_id})...")

        for i, seg in enumerate(segments):
            text = seg["translated_text"]
            if len(text) > max_text_length:
                print(f"    [{i + 1}/{len(segments)}] Truncating {len(text)} -> {max_text_length} chars")
                text = text[:max_text_length]

            chunk_path = audio_dir / f"chunk_{i:04d}.wav"
            t_seg = time.perf_counter()

            try:
                client.synthesize_with_preset(
                    text=text, voice_id=voice_id, output_path=chunk_path,
                )
                seg_time = time.perf_counter() - t_seg
                audio_dur = sf.info(str(chunk_path)).duration
                total_audio_duration += audio_dur
                success_count += 1

                seg["audio_path"] = str(chunk_path)
                seg["audio_duration"] = round(audio_dur, 3)
                print(f"    [{i + 1}/{len(segments)}] {seg['start_time']} -> {seg['end_time']}  "
                      f"({audio_dur:.1f}s in {seg_time:.1f}s)  {text[:50]}...")
            except Exception as exc:
                fail_count += 1
                seg["audio_path"] = None
                seg["error"] = str(exc)
                print(f"    [{i + 1}/{len(segments)}] FAILED: {exc}")

    # ── Save manifest ───────────────────────────────────────────
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "voice_mode": args.voice_mode,
        "engine": args.engine,
        "vieneu_mode": args.vieneu_mode,
        "device": args.device,
        "voice_id": voice_id if args.voice_mode == "preset" else None,
        "ref_audio": str(args.ref_audio) if args.voice_mode == "clone" else None,
        "total_segments": len(segments),
        "success_count": success_count,
        "fail_count": fail_count,
        "total_audio_duration": round(total_audio_duration, 2),
        "segments": segments,
    }
    manifest_path = audio_dir / "tts_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    client.unload_model()
    print(f"\n  TTS Results: {success_count}/{len(segments)} OK, {fail_count} failed")
    print(f"  Total audio: {total_audio_duration:.1f}s")


# ===================================================================== #
#  Skip-VLM helper — load from existing extraction results              #
# ===================================================================== #

def load_segments_from_json(path: Path) -> list[dict]:
    """Load segments from an existing extraction_results.json or directory."""
    if path.is_dir():
        json_path = path / "extraction_results.json"
        if not json_path.exists():
            print(f"extraction_results.json not found in: {path}")
            sys.exit(1)
    else:
        json_path = path

    if not json_path.exists():
        print(f"File not found: {json_path}")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments: list[dict] = []
    for chunk in data.get("results", []):
        for entry in chunk.get("entries", []):
            text = entry.get("translated_text", "").strip()
            if not text:
                continue
            segments.append({
                "translated_text": text,
                "start_time": entry.get("start_time", 0),
                "end_time": entry.get("end_time", 0),
                "original_text": entry.get("original_text", ""),
            })
    return segments


# ===================================================================== #
#  Main                                                                  #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser(
        description="Full Pipeline: Video -> VLM Extract -> Voice Cloning",
    )

    # ── Video input ─────────────────────────────────────────────
    parser.add_argument("--video", "-v", type=str, default=None,
                        help="Path to video file")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: ./output/pipeline_TIMESTAMP)")

    # ── Skip VLM (reuse existing results) ───────────────────────
    parser.add_argument("--skip-vlm", action="store_true",
                        help="Skip Phase 1; load segments from existing extraction_results.json")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to extraction_results.json or directory (used with --skip-vlm)")

    # ── VLM backend ─────────────────────────────────────────────
    parser.add_argument("--api", action="store_true",
                        help="Use Gemini API for VLM")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Gemini API key (or set GEMINI_API_KEY in .env)")
    parser.add_argument("--api-model", type=str, default="gemini-2.5-flash-lite",
                        help="Gemini model (default: gemini-2.5-flash-lite)")
    parser.add_argument("--model", type=str, default="./models/qwen3.5-2b",
                        help="Local model path (default: ./models/qwen3.5-2b)")
    parser.add_argument("--qwen25", action="store_true",
                        help="Use legacy Qwen2.5-VL-3B")

    # ── VLM processing ─────────────────────────────────────────
    parser.add_argument("--mode", choices=["single", "3step"], default="single",
                        help="VLM prompt mode (default: single)")
    parser.add_argument("--no-chunk", action="store_true",
                        help="Process entire video in one pass (no chunking)")
    parser.add_argument("--chunk-duration", type=float, default=45.0,
                        help="Max chunk duration in seconds (default: 45)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames per VLM call")
    parser.add_argument("--threshold", type=float, default=20.0,
                        help="Scene detection threshold (default: 20.0)")
    parser.add_argument("--delay", type=float, default=None,
                        help="Delay between chunks in seconds")
    parser.add_argument("--legacy", action="store_true",
                        help="Legacy VLM mode (fixed interval, no improvements)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Frame interval in seconds (legacy mode)")
    parser.add_argument("--ssim-threshold", type=float, default=0.85,
                        help="SSIM dedup threshold (default: 0.85)")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable SSIM frame dedup")
    parser.add_argument("--no-global-summary", action="store_true",
                        help="Skip Global Summary Pass")
    parser.add_argument("--overlap", type=float, default=5.0,
                        help="Chunk overlap in seconds (default: 5.0)")
    parser.add_argument("--global-frames", type=int, default=15,
                        help="Frames for Global Summary Pass (default: 15)")

    # ── Voice mode ──────────────────────────────────────────────
    parser.add_argument("--voice-mode", choices=["preset", "clone"], required=True,
                        help="Voice mode: 'preset' or 'clone'")
    parser.add_argument("--list-voices", action="store_true",
                        help="List preset voices and exit")
    parser.add_argument("--voice-id", type=str, default=None,
                        help="Preset voice ID (default: first available)")
    parser.add_argument("--ref-audio", type=str, default=None,
                        help="Reference audio path for voice cloning (3–10s)")
    parser.add_argument("--ref-text", type=str, default=None,
                        help="Transcript of reference audio (optional)")

    # ── TTS engine ──────────────────────────────────────────────
    parser.add_argument("--engine", type=str, default="vieneu",
                        help="TTS engine (default: vieneu)")
    parser.add_argument("--vieneu-mode", type=str, default="turbo",
                        help="Vieneu mode: turbo | standard | fast | remote (default: turbo)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: cuda | cpu (default: cuda)")
    parser.add_argument("--max-text-length", type=int, default=200,
                        help="Max chars per segment (default: 200)")

    args = parser.parse_args()

    # ── --list-voices ───────────────────────────────────────────
    if args.list_voices:
        from src.m2_tts.tts_client import TTSClient
        client = TTSClient(engine=args.engine, vieneu_mode=args.vieneu_mode)
        voices = client.list_preset_voices()
        print(f"Available preset voices ({len(voices)}):")
        for desc, vid in voices:
            print(f"  [{vid}] {desc}")
        client.unload_model()
        sys.exit(0)

    # ── Validate args ───────────────────────────────────────────
    if args.voice_mode == "clone" and not args.ref_audio:
        print("Error: --ref-audio is required when --voice-mode clone")
        sys.exit(1)

    if args.voice_mode == "clone":
        if not Path(args.ref_audio).exists():
            print(f"Reference audio not found: {args.ref_audio}")
            sys.exit(1)

    if args.skip_vlm:
        if not args.input:
            print("Error: --input is required when --skip-vlm")
            sys.exit(1)
        if not args.video:
            # Need an output dir
            args.output = args.output or str(Path(args.input) if Path(args.input).is_dir()
                                             else Path(args.input).parent)
    else:
        if not args.video:
            print("Error: --video is required (or use --skip-vlm --input PATH)")
            sys.exit(1)

    # Apply --qwen25
    if args.qwen25 and args.model == "./models/qwen3.5-2b":
        args.model = "./models/qwen2.5-vl-3b"

    # Legacy mode disables improvements
    if args.legacy:
        args.no_dedup = True
        args.no_global_summary = True
        args.overlap = 0.0

    # Smart defaults
    if args.max_frames is None:
        if args.api:
            args.max_frames = 2
        elif args.no_chunk:
            args.max_frames = 32
        else:
            args.max_frames = 8
    if args.delay is None:
        args.delay = 30.0 if args.api else 0.0

    # Resolve API key
    if args.api and not args.api_key:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        args.api_key = os.getenv("GEMINI_API_KEY")
        if not args.api_key:
            print("Error: Gemini API key required. Use --api-key or set GEMINI_API_KEY in .env")
            sys.exit(1)

    # ════════════════════════════════════════════════════════════
    #  Run pipeline
    # ════════════════════════════════════════════════════════════

    pipeline_start = time.perf_counter()

    print(f"\n{'═' * 65}")
    print(f"  FULL PIPELINE: Video -> VLM -> Voice Cloning")
    print(f"{'═' * 65}")

    if args.skip_vlm:
        input_path = Path(args.input)
        out_dir = Path(args.output) if args.output else (
            input_path if input_path.is_dir() else input_path.parent)
        segments = load_segments_from_json(input_path)
        print(f"  [SKIP VLM] Loaded {len(segments)} segments from {input_path}")
    else:
        segments, out_dir, _ = run_vlm_phase(args)

    if not segments:
        print("\nNo subtitle segments to synthesize. Exiting.")
        sys.exit(1)

    # Phase 2: TTS
    run_tts_phase(segments, out_dir, args)

    pipeline_elapsed = time.perf_counter() - pipeline_start

    # ── Final summary ───────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'═' * 65}")
    print(f"  Output:    {out_dir.resolve()}")
    print(f"  Segments:  {len(segments)}")
    print(f"  Elapsed:   {pipeline_elapsed:.1f}s")
    print()
    print(f"  Files:")
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir)
            sz = f.stat().st_size
            print(f"    {rel}  ({sz / 1024:.0f} KB)" if sz > 1024 else f"    {rel}  ({sz} B)")


if __name__ == "__main__":
    main()
