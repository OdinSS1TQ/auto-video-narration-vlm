# Phase 2.A — End-to-end Vietnamese Dubbing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the first complete English → Vietnamese dubbed video output (`Demo-Module-5_vi.mp4`) by validating Module 3, fixing the `runner.py:175` chunk_offset bug, then refreshing user-facing docs.

**Architecture:** Hybrid validation — run M1 + M2 with existing scripts to capture artifacts, then exercise M3 (`AudioAligner` → `TimeStretcher` → `FFmpegRenderer`) via a new standalone script before fixing the orchestrator. M3 components already exist in `src/m3_sync/`; the runner already wires them at `runner.py:215-244`.

**Tech Stack:** Python 3.10+, PyTorch (Qwen3.5-2B local VLM), VieNeu-TTS v2 Turbo, rubberband 4.0 CLI, FFmpeg, soundfile, loguru.

**Spec:** `docs/superpowers/specs/2026-05-05-phase2a-end-to-end-design.md`

**Test inputs:**
- Video: `H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4` (60.4s, 1920×1080, 30fps)
- Reference audio: `data/reference_audio/speaker.wav` (10.0s, 16kHz mono)
- VLM: Qwen3.5-2B local
- rubberband: `C:\Tools\rubberband\rubberband.exe` v4.0.0 on PATH
- ffmpeg: `C:\ffmpeg\bin\ffmpeg.exe` on PATH

**Audit findings used by this plan:**
- `AudioAligner.calculate_deltas(segments)` — expects each segment dict to have keys `start_time`, `end_time` (SRT format `HH:MM:SS,mmm`), and `audio_path`. Returns segments enriched with `target_duration`, `audio_duration`, `delta`, `start_sec`, `end_sec`, `strategy`.
- `AudioAligner.align_all(segments_with_deltas, output_dir)` — produces `aligned_audio_path` per segment.
- `FFmpegRenderer.merge_audio_segments(segments, total_duration, output_path, sample_rate=22050)` — default sample rate is 22050, but TTS outputs **24000**. Must pass `sample_rate=24000` explicitly.
- `FFmpegRenderer.render_final_video(video_path, dubbed_audio_path, output_path)` — replaces audio entirely by default.
- `runner.py:175` uses `chunk_offset=start` which double-shifts timestamps because prompts already request absolute time. Must change to `chunk_offset=0.0`.
- `runner.py:234` calls `merge_audio_segments` without `sample_rate=24000` — same bug.
- `BatchInference.process_all(...)` returns `List[Dict]` where each segment preserves SRT keys and adds `audio_path`. This shape is directly consumable by `AudioAligner.calculate_deltas`.

---

## File structure

**New files:**
- `scripts/test_m3_sync.py` — standalone M3 validator
- `docs/RUNNING.md` — first-run walkthrough (Vietnamese)

**Modified files:**
- `src/m4_pipeline/runner.py` — fix line 175 chunk_offset; pass sample_rate=24000 to renderer
- `README.md` — refresh install/run sections, Windows-specific notes

**Read-only (audited, no changes expected unless tests reveal bugs):**
- `src/m3_sync/audio_aligner.py`
- `src/m3_sync/time_stretcher.py`
- `src/m3_sync/ffmpeg_renderer.py`

---

## Task 1: Verify rubberband 4.0 CLI flag compatibility

**Files:** none (verification only).

- [ ] **Step 1: Verify rubberband is on PATH and reports v4**

Run: `rubberband --version`
Expected output (or similar): `Rubber Band Library v4.0.0`

- [ ] **Step 2: Verify `--time` and `--pitch` flags still exist in v4**

Run: `rubberband --help 2>&1 | findstr /i "time pitch"`
Expected: lines mentioning `-T --time <X>` and `-p --pitch <S>` (or similar). The flags `time_stretcher.py:80-86` uses (`--time`, `--pitch`) must be present.

If any flag is missing, patch `src/m3_sync/time_stretcher.py:80-86` to use the v4 equivalent and document the change in a commit.

- [ ] **Step 3: Smoke test on a known WAV**

Run:
```powershell
$ref = "H:\Quan\tanquan\Graduation\auto-video-narration-vlm\data\reference_audio\speaker.wav"
rubberband --time 1.5 $ref "$env:TEMP\rb_smoke.wav"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$env:TEMP\rb_smoke.wav"
```
Expected: prints a duration close to `15.0` (1.5 × 10s).

- [ ] **Step 4: Commit nothing (read-only verification)**

If a patch was needed in step 2:
```bash
git add src/m3_sync/time_stretcher.py
git commit -m "fix(m3): update rubberband CLI flags for v4 compatibility"
```

---

## Task 2: Run M1 on `Demo-Module-5.mp4` to produce SRT

**Files:** none created in repo (artifact lands in `output/`).

- [ ] **Step 1: Confirm `.env` has Qwen path set**

Read `.env`. The variable `VLM_MODE` should be `local` and `QWEN_MODEL_PATH` (or equivalent — check `configs/qwen_local.yaml`) must point at the downloaded Qwen3.5-2B weights. If models aren't downloaded:
```powershell
python scripts/download_models.py
```

- [ ] **Step 2: Run M1 extraction**

```powershell
python scripts/run_vlm_extract.py `
    --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
    --mode local
```

Expected:
- Console shows scene detection, frame extraction, deduplication, then VLM inference per chunk.
- Output file: `output/Demo-Module-5_vi.srt` (or whatever path the script reports — verify it exists).
- Final SRT contains numbered Vietnamese subtitles with `HH:MM:SS,mmm` timestamps.

- [ ] **Step 3: Sanity-check the SRT**

Read the file. Confirm:
- At least 2–3 entries (60s video produces several segments).
- Last entry's end timestamp is close to but not beyond `00:01:00,415` (video duration 60.415s).
- All entries have non-empty `translated_text` (Vietnamese characters present).

If timestamps are clearly broken (all `00:00:00,000` or all beyond video length), abort and investigate before proceeding — running M2/M3 on bad SRT wastes time.

- [ ] **Step 4: Commit nothing — SRT is a generated artifact, not source.**

---

## Task 3: Run M1+M2 integration to produce audio chunks

**Files:** artifacts in `debug_output/m1_m2_integration/audio_chunks/`.

- [ ] **Step 1: Locate the SRT path produced by Task 2**

Confirm `output/Demo-Module-5_vi.srt` exists (or note actual path).

- [ ] **Step 2: Run integration script**

```powershell
python scripts/test_m1_m2_integration.py `
    --srt "output/Demo-Module-5_vi.srt" `
    --ref-audio "data/reference_audio/speaker.wav" `
    --output-dir "debug_output/m1_m2_integration"
```

Expected:
- Each SRT entry produces one `chunk_NNNN.wav` (24kHz, mono, float32) in `debug_output/m1_m2_integration/audio_chunks/`.
- Console reports per-segment synthesis time and total elapsed.
- Final summary lists 100% success (all segments synthesized).

- [ ] **Step 3: Verify chunk count matches SRT entry count**

```powershell
$srtEntries = (Select-String -Path "output/Demo-Module-5_vi.srt" -Pattern "^\d+$" -AllMatches).Matches.Count
$wavCount = (Get-ChildItem "debug_output/m1_m2_integration/audio_chunks/chunk_*.wav").Count
Write-Host "SRT entries: $srtEntries, WAV chunks: $wavCount"
```
Expected: same number. If not, the bridge has a mismatch — investigate before proceeding.

- [ ] **Step 4: Spot-check one WAV is valid 24kHz mono**

```powershell
ffprobe -v error -show_entries stream=sample_rate,channels -of default=noprint_wrappers=1 "debug_output/m1_m2_integration/audio_chunks/chunk_0000.wav"
```
Expected: `sample_rate=24000`, `channels=1`.

- [ ] **Step 5: Commit nothing — audio chunks are generated artifacts.**

---

## Task 4: Build `scripts/test_m3_sync.py` skeleton with AudioAligner pass

**Files:**
- Create: `scripts/test_m3_sync.py`

- [ ] **Step 1: Write the script with AudioAligner stage only (no stretch/render yet)**

Create `scripts/test_m3_sync.py`:

```python
"""
M3 Sync standalone validator.

Loads M1 SRT + M2 audio chunks, exercises AudioAligner → TimeStretcher → FFmpegRenderer
in isolation so M3 bugs surface independently of the pipeline orchestrator.

Usage:
    python scripts/test_m3_sync.py \
        --srt output/Demo-Module-5_vi.srt \
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
            f"Mismatch: {len(entries)} SRT entries vs {len(audio_files)} audio files"
        )

    segments = []
    for entry, audio_path in zip(entries, audio_files):
        seg = dict(entry)
        seg["audio_path"] = str(audio_path)
        segments.append(seg)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--srt", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
        logger.info(
            f"#{seg.get('index'):02d} target={seg['target_duration']:.2f}s "
            f"audio={seg['audio_duration']:.2f}s delta={seg['delta']:+.2f}s "
            f"strategy={seg['strategy']}"
        )

    # Save intermediate report for inspection
    report_path = args.output_dir / "stage1_deltas.json"
    report_path.write_text(json.dumps(with_deltas, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Stage 1 report saved: {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```powershell
python scripts/test_m3_sync.py `
    --srt "output/Demo-Module-5_vi.srt" `
    --audio-dir "debug_output/m1_m2_integration/audio_chunks" `
    --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
    --output-dir "debug_output/m3_test_run"
```

Expected:
- Per-segment deltas printed (target, audio, delta, strategy).
- `debug_output/m3_test_run/stage1_deltas.json` written.
- No exceptions.

- [ ] **Step 3: Inspect the report**

Read `debug_output/m3_test_run/stage1_deltas.json`. Verify:
- Each entry has `target_duration`, `audio_duration`, `delta`, `strategy`.
- Most strategies will be `stretch_compress` (Vietnamese narration tends to be longer than English subtitle slot) or `stretch_expand`.

- [ ] **Step 4: Commit**

```bash
git add scripts/test_m3_sync.py
git commit -m "feat(m3): add standalone M3 validator (stage 1: AudioAligner)"
```

---

## Task 5: Extend `test_m3_sync.py` with align_all + render stages

**Files:**
- Modify: `scripts/test_m3_sync.py`

- [ ] **Step 1: Add the alignment + render stages**

Edit `scripts/test_m3_sync.py`. After the `Stage 1: AudioAligner.calculate_deltas` block, append:

```python
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

    logger.info(f"=== Done ===")
    logger.info(f"Merged audio : {merged_audio}")
    logger.info(f"Final video  : {final_video}")
```

- [ ] **Step 2: Run the extended script**

```powershell
python scripts/test_m3_sync.py `
    --srt "output/Demo-Module-5_vi.srt" `
    --audio-dir "debug_output/m1_m2_integration/audio_chunks" `
    --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
    --output-dir "debug_output/m3_test_run"
```

Expected:
- Per-segment alignment strategy logged (`exact` / `stretch` / `pad` / `truncate`).
- `debug_output/m3_test_run/aligned_audio/aligned_NNNN.wav` files written.
- `debug_output/m3_test_run/merged_track.wav` written.
- `debug_output/m3_test_run/Demo-Module-5_vi.mp4` written.
- No FFmpeg or rubberband errors in console.

- [ ] **Step 3: Verify final video duration is within ±2s of source**

```powershell
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "debug_output/m3_test_run/Demo-Module-5_vi.mp4"
```
Expected: a value within `[58.4, 62.4]` (source is 60.415s).

- [ ] **Step 4: Commit**

```bash
git add scripts/test_m3_sync.py
git commit -m "feat(m3): extend validator with align_all and render stages"
```

---

## Task 6: Manual validation of M3 output

**Files:** none (subjective + duration checks).

- [ ] **Step 1: Listen to the merged audio**

Open `debug_output/m3_test_run/merged_track.wav` in any media player. Confirm:
- Vietnamese speech is present and clearly audible.
- The cloned voice resembles `speaker.wav` (timbre, accent).
- Speech segments roughly align with what would be expected at each subtitle's start time.

- [ ] **Step 2: Play the final video**

Open `debug_output/m3_test_run/Demo-Module-5_vi.mp4` in VLC or Windows Media Player. Confirm:
- Video plays normally.
- Audio is the dubbed Vietnamese track (NOT the original English).
- Video and audio do not desync catastrophically (small drift is acceptable for 2.A).

- [ ] **Step 3: Document any issues observed**

If the video/audio is unusable (e.g., audio cuts out, all silence, rubberband artifacts), STOP and investigate. Most likely culprits:
- Sample rate mismatch (verified passing 24000 in Task 5).
- Empty segments (segment had `audio_path: None` due to TTS failure — should have been logged in Task 3).
- adelay mono-vs-stereo issue: `merge_audio_segments` line 79 uses `adelay={start_ms}|{start_ms}` which is the stereo form. For mono audio FFmpeg accepts this without error but may produce stereo output. If output is acceptable, leave it; if not, change to `adelay={start_ms}` (single value for mono).

If issues are minor (small drift, occasional artifact), accept and proceed — Phase 2.A is the proof-of-concept run, not the polished final.

- [ ] **Step 4: Commit nothing — this is a validation gate.**

---

## Task 7: Fix `runner.py:175` chunk_offset bug

**Files:**
- Modify: `src/m4_pipeline/runner.py:175`

- [ ] **Step 1: Read the current line**

`src/m4_pipeline/runner.py` line 175 currently reads:
```python
                    srt_builder.add_entries(entries, chunk_offset=start)
```

- [ ] **Step 2: Apply the fix**

Change line 175 to:
```python
                    srt_builder.add_entries(entries, chunk_offset=0.0)
```

Rationale: prompts in `PromptChain` already instruct the VLM to emit absolute timestamps. Adding `chunk_offset=start` shifts every subtitle later than it should be. `scripts/run_vlm_extract.py` correctly uses `chunk_offset=0.0`; the runner must match.

- [ ] **Step 3: Commit**

```bash
git add src/m4_pipeline/runner.py
git commit -m "fix(m4): use chunk_offset=0.0 since VLM emits absolute timestamps"
```

---

## Task 8: Pass `sample_rate=24000` to renderer in runner

**Files:**
- Modify: `src/m4_pipeline/runner.py:233-236`

- [ ] **Step 1: Read the current call**

`src/m4_pipeline/runner.py` lines 233-236 currently read:
```python
            merged_audio_path = work_dir / "merged_audio.wav"
            renderer.merge_audio_segments(
                aligned_segments, duration, merged_audio_path
            )
```

- [ ] **Step 2: Apply the fix**

Change those lines to:
```python
            merged_audio_path = work_dir / "merged_audio.wav"
            renderer.merge_audio_segments(
                aligned_segments,
                duration,
                merged_audio_path,
                sample_rate=24000,
            )
```

Rationale: VieNeu-TTS produces 24kHz mono audio. `FFmpegRenderer.merge_audio_segments` defaults to 22050, which would resample and degrade quality.

- [ ] **Step 3: Commit**

```bash
git add src/m4_pipeline/runner.py
git commit -m "fix(m4): pass sample_rate=24000 to renderer to match TTS native rate"
```

---

## Task 9: End-to-end run via `scripts/run_pipeline.py`

**Files:** none (runs existing entry script).

- [ ] **Step 1: Confirm `scripts/run_pipeline.py` accepts the expected CLI args**

Read `scripts/run_pipeline.py`. Confirm it accepts `--video` and `--ref-audio` and instantiates `PipelineRunner.run()`. If it doesn't, document the actual interface and use that instead.

- [ ] **Step 2: Run the orchestrator end-to-end**

```powershell
python scripts/run_pipeline.py `
    --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
    --ref-audio "data/reference_audio/speaker.wav"
```

Expected:
- Console shows progress through all 8 steps: scene_detection → frame_extraction → ocr_extraction → vlm_translation → srt_generation → voice_cloning → audio_alignment → video_rendering.
- Final output: `output/Demo-Module-5/Demo-Module-5_dubbed.mp4` (or path reported by the script).
- No exceptions; pipeline completes.

- [ ] **Step 3: Verify the orchestrator output matches the standalone test**

Compare:
- Final video duration: should be within 2s of 60.415s.
- SRT (`output/Demo-Module-5/Demo-Module-5_vi.srt`) should have similar entry count and timestamps to the one from Task 2.
- Play the dubbed video — should be perceptually similar to the Task 5 output.

If the orchestrator output is significantly worse than the standalone test, investigate which step diverged.

- [ ] **Step 4: Commit nothing — orchestrator output is a generated artifact.**

---

## Task 10: Refresh `README.md`

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read the current README**

Read `README.md`. Note the existing structure and Vietnamese phrasing.

- [ ] **Step 2: Update Yêu cầu hệ thống section**

Find the section starting `## Yêu cầu hệ thống` and replace with:

```markdown
## Yêu cầu hệ thống

- **Python** ≥ 3.10
- **GPU** với CUDA 12.1+, ≥ 4 GB VRAM (cho Qwen3.5-2B local) hoặc Gemini API key
- **FFmpeg** (Windows: tải tại https://ffmpeg.org/, thêm `bin\` vào PATH; Linux: `apt install ffmpeg`)
- **Rubberband CLI** (Windows: tải tại https://breakfastquay.com/rubberband/ executable, thêm folder chứa `rubberband.exe` vào PATH; Linux: `apt install rubberband-cli`)
- **HuggingFace token** (cho VieNeu-TTS — lưu trong `.env` qua biến `HF_TOKEN`)
```

- [ ] **Step 3: Update Cài đặt section to include Windows PowerShell variant**

Replace the `## Cài đặt` section content with:

```markdown
## Cài đặt

### Linux / macOS

```bash
git clone <repo-url>
cd auto-video-narration-vlm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Chỉnh sửa với HF_TOKEN, GEMINI_API_KEY (nếu dùng API mode)
python scripts/download_models.py
```

### Windows (PowerShell)

```powershell
git clone <repo-url>
cd auto-video-narration-vlm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env  # Chỉnh sửa với HF_TOKEN, GEMINI_API_KEY (nếu dùng API mode)
python scripts/download_models.py
```

### Cài rubberband trên Windows

Tải `rubberband-X.X.X-gpl-executable-windows.zip` tại https://breakfastquay.com/rubberband/, giải nén vào ví dụ `C:\Tools\rubberband\`, rồi thêm vào user PATH:

```powershell
$dest = "C:\Tools\rubberband"
$current = [Environment]::GetEnvironmentVariable("Path", "User")
if ($current -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable("Path", "$current;$dest", "User")
}
```

Mở terminal mới rồi verify: `rubberband --version`.
```

- [ ] **Step 4: Update Sử dụng section with concrete example**

Replace the `## Sử dụng` section content with:

```markdown
## Sử dụng

### CLI — chạy pipeline đầy đủ

```bash
python scripts/run_pipeline.py \
    --video data/raw/sample.mp4 \
    --ref-audio data/reference_audio/speaker.wav
```

Output: `output/<video_stem>/<video_stem>_dubbed.mp4` + SRT trung gian.

### CLI — chỉ trích xuất SRT (không TTS)

```bash
python scripts/run_vlm_extract.py --video data/raw/sample.mp4 --mode local
```

`--mode` chấp nhận `local` (Qwen3.5-2B, mặc định) hoặc `api` (Gemini).

### Test M3 standalone (sau khi đã có SRT + audio chunks)

```bash
python scripts/test_m3_sync.py \
    --srt output/sample_vi.srt \
    --audio-dir debug_output/m1_m2_integration/audio_chunks \
    --video data/raw/sample.mp4 \
    --output-dir debug_output/m3_test_run
```

### Web UI

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Truy cập http://localhost:8000 (Gradio mounted tại /ui)
```
```

- [ ] **Step 5: Update Trạng thái module table at the bottom**

Find or add a section `## Trạng thái module` containing:

```markdown
## Trạng thái module

| Module | Trạng thái | Ghi chú |
|--------|-----------|---------|
| **M1: VLM Extraction** | ✅ Hoàn thành | 8 components, 8 pytest files. Default: Qwen3.5-2B local. |
| **M2: TTS Voice Cloning** | ✅ Hoàn thành | VieNeu-TTS v2 Turbo, 24 test case, bridge SRT→TTS hoạt động. |
| **M3: Audio Sync & Render** | ✅ Hoàn thành (Phase 2.A) | AudioAligner + TimeStretcher (rubberband) + FFmpegRenderer. Tested end-to-end trên Demo-Module-5. |
| **M4: Pipeline Orchestrator** | ✅ Hoàn thành | `PipelineRunner` chạy 8 step end-to-end. |
| **M5: Evaluation** | ⚠️ Một phần | BLEU, Speaker Similarity, Sync Accuracy hoạt động. MOS chưa implement. |
| **Web App** | ⚠️ Scaffolding | FastAPI + Gradio — chưa test thực tế. |
```

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: refresh README for Phase 2.A — Windows install, M3 status, end-to-end usage"
```

---

## Task 11: Create `docs/RUNNING.md` walkthrough

**Files:**
- Create: `docs/RUNNING.md`

- [ ] **Step 1: Write the file**

Create `docs/RUNNING.md`:

```markdown
# Hướng dẫn chạy lần đầu

Tài liệu này hướng dẫn chi tiết cách chạy pipeline `auto-video-narration-vlm` từ checkout sạch trên máy Windows. Linux/macOS tương tự — chỉ khác bước cài tools native.

## 1. Prerequisites

Cần cài sẵn trên máy:

| Tool | Version tối thiểu | Kiểm tra |
|------|-------------------|----------|
| Python | 3.10 | `python --version` |
| CUDA Toolkit | 12.1 | `nvidia-smi` |
| FFmpeg | 4.x | `ffmpeg -version` |
| Rubberband | 3.x trở lên (4.0 đã test OK) | `rubberband --version` |
| Git | 2.x | `git --version` |

GPU có ≥ 4 GB VRAM. Test thực tế trên RTX 3050 4GB chạy được Qwen3.5-2B local.

## 2. Clone và setup môi trường

```powershell
git clone <repo-url>
cd auto-video-narration-vlm

# Tạo venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Cài dependencies
pip install -r requirements.txt
```

## 3. Cấu hình `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Cần điền:
- `HF_TOKEN=hf_xxx` — bắt buộc cho VieNeu-TTS (lấy tại https://huggingface.co/settings/tokens)
- `GEMINI_API_KEY=...` — chỉ cần nếu dùng `--mode api`
- `VLM_MODE=local` — mặc định, dùng Qwen3.5-2B local
- `QWEN_MODEL_PATH=...` — đường dẫn weights Qwen3.5-2B (sẽ tải ở bước 4)

## 4. Tải model weights

```powershell
python scripts/download_models.py
```

Tải ~5 GB:
- Qwen3.5-2B (~4 GB)
- VieNeu-TTS v2 Turbo (~700 MB GGUF + ~200 MB ONNX)

## 5. Chạy thử pipeline đầy đủ

Chuẩn bị input:
- 1 video tiếng Anh ngắn (1–2 phút) đặt vào `data/raw/sample.mp4`
- 1 file audio reference (3–10 s, mono, WAV) đặt vào `data/reference_audio/speaker.wav`

Chạy:

```powershell
python scripts/run_pipeline.py `
    --video "data/raw/sample.mp4" `
    --ref-audio "data/reference_audio/speaker.wav"
```

Output mong đợi:
- `output/sample/sample_vi.srt` — phụ đề tiếng Việt
- `output/sample/sample_dubbed.mp4` — video lồng tiếng Việt
- `output/sample/work/audio_chunks/chunk_NNNN.wav` — audio TTS từng segment
- `output/sample/work/aligned_audio/aligned_NNNN.wav` — audio sau time-stretch
- `output/sample/work/merged_audio.wav` — track audio cuối trước khi mux

Thời gian chạy ước tính trên GTX 3050 4GB cho video 60s: ~6–10 phút (chủ yếu là Qwen3.5-2B inference).

## 6. Kiểm tra từng module riêng lẻ

Nếu pipeline đầy đủ fail, debug từng bước:

### Chỉ M1 (Video → SRT)

```powershell
python scripts/run_vlm_extract.py --video "data/raw/sample.mp4" --mode local
```

### M1 + M2 (SRT → audio chunks)

```powershell
python scripts/test_m1_m2_integration.py `
    --srt "output/sample_vi.srt" `
    --ref-audio "data/reference_audio/speaker.wav" `
    --output-dir "debug_output/m1_m2_integration"
```

### Chỉ M3 (chunks + SRT → dubbed video)

```powershell
python scripts/test_m3_sync.py `
    --srt "output/sample_vi.srt" `
    --audio-dir "debug_output/m1_m2_integration/audio_chunks" `
    --video "data/raw/sample.mp4" `
    --output-dir "debug_output/m3_test_run"
```

## 7. Troubleshooting

### `CUDA out of memory`
- Đóng tab browser, giảm batch size trong `.env` (nếu có), hoặc dùng `--mode api` (Gemini) để bỏ tải model local.

### `rubberband: command not found`
- Tải binary tại https://breakfastquay.com/rubberband/ → giải nén → thêm folder vào PATH → mở terminal mới.

### `FileNotFoundError: speaker.wav`
- Chạy `ls data/reference_audio/`. Phải có 1 file WAV. Nếu thiếu, thay đường dẫn `--ref-audio`.

### Model download stall
- HuggingFace rate limit hoặc mạng yếu. Chạy lại `python scripts/download_models.py` (sẽ resume).

### Video output không phát được / im lặng
- Kiểm tra `output/<stem>/work/merged_audio.wav` mở được không.
- Nếu merged_audio im lặng → M2 fail, kiểm tra log TTS.
- Nếu merged_audio OK nhưng video im lặng → ffmpeg mux fail, kiểm tra log step `video_rendering`.

### Phụ đề lệch timestamp
- Đây là known issue của Phase 2.A (timestamp accuracy của VLM chưa hoàn hảo). Sẽ cải thiện ở Phase 2.C.

## 8. Output structure tham khảo

```
output/
└── sample/
    ├── sample_vi.srt              # Phụ đề tiếng Việt
    ├── sample_dubbed.mp4          # Video lồng tiếng cuối cùng
    └── work/                       # Intermediate artifacts
        ├── audio_chunks/
        │   ├── chunk_0000.wav     # TTS output từng segment
        │   └── ...
        ├── aligned_audio/
        │   ├── aligned_0000.wav   # Sau time-stretch
        │   └── ...
        └── merged_audio.wav       # Track audio cuối
```
```

- [ ] **Step 2: Verify the file renders**

Open `docs/RUNNING.md` in any markdown viewer (VS Code preview is fine). Confirm code blocks render correctly and tables are well-formatted.

- [ ] **Step 3: Commit**

```bash
git add docs/RUNNING.md
git commit -m "docs: add RUNNING.md first-run walkthrough for Phase 2.A"
```

---

## Self-review

**Spec coverage:**
- §1 Goal — Tasks 2-9 produce the dubbed video.
- §2 Non-goals — Plan does not touch timestamp accuracy, MOS, GLM-OCR, Web UI, Pass 0, M3 pytest. ✓
- §3 Inputs — Tasks 2/3/9 use them.
- §4 Approach steps 1-7 — Task 1 verifies rubberband; Tasks 2-3 prep artifacts; Tasks 4-6 build standalone validator; Tasks 7-8 fix runner; Task 9 runs orchestrator; Tasks 10-11 docs. ✓
- §5.1 Data flow — Tasks 4-5 implement it.
- §5.2 Component contracts — audited above; method names corrected (`calculate_deltas`, `align_all`, `merge_audio_segments`, `render_final_video`).
- §5.3 New script — Tasks 4-5.
- §5.4 Runner patches — Tasks 7-8 (chunk_offset + sample_rate).
- §6 Validation criteria — Task 6 (subjective listen + duration check) + Task 9 (orchestrator parity).
- §7 Error handling — built into existing components; Task 6 documents the adelay mono concern.
- §8.1 README — Task 10.
- §8.2 RUNNING.md — Task 11.
- §9 Deliverables — all covered.
- §10 Open risks — Task 1 mitigates rubberband flag risk; Task 8 mitigates sample_rate risk; Task 6 documents adelay mono risk.

**Placeholder scan:** No "TBD" / "TODO" / "implement later". Each step has concrete code or commands. ✓

**Type consistency:** Method names verified against actual source: `calculate_deltas`, `align_all`, `stretch_to_fit`, `merge_audio_segments`, `render_final_video`, `get_video_info`, `load_srt`. ✓

---

**Next step:** Choose execution mode (subagent-driven vs inline).
