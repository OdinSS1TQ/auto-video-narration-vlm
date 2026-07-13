# Phase 2.A — End-to-end Vietnamese Dubbing (Design)

**Date:** 2026-05-05
**Branch:** `dev-05052026`
**Author:** Ngo Nguyen Tan Quan
**Status:** Approved (pending user review of this spec)

---

## 1. Goal

Produce the **first complete English → Vietnamese dubbed video output** by validating Module 3 (audio sync + render) against real M2 output, then wiring M3 into the pipeline orchestrator. Refresh the user-facing documentation so a reviewer can reproduce the run from a clean checkout on Windows.

## 2. Non-goals (YAGNI)

The following are explicitly out of scope for this spec; they are tracked as separate Phase 2 sub-projects:

- Improving VLM timestamp accuracy (Phase 2.C)
- `MOSEstimator` implementation (Phase 2.D evaluation)
- GLM-OCR integration into the pipeline
- Web UI (FastAPI + Gradio) testing/hardening
- Enabling Pass 0 global summary inside `PipelineRunner`
- Formal `pytest` coverage for Module 3 (script-driven validation suffices for 2.A; pytest comes in 2.B)
- Modifying historical artifacts: `Phase1_Progress_Report.md`, `BaoCao_TienDo_Phase1.md`

## 3. Confirmed inputs and environment

| Input | Value |
|---|---|
| Test video | `H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4` (60.4 s, 1920×1080, 30 fps, h264 + aac) |
| Reference audio | `data/reference_audio/speaker.wav` (10.0 s, 16 kHz, mono) |
| VLM backend | Qwen3.5-2B local (`AutoModelForImageTextToText`) |
| `rubberband` | 4.0.0 GPL, installed at `C:\Tools\rubberband\rubberband.exe`, on user PATH |
| `ffmpeg` / `ffprobe` | Installed at `C:\ffmpeg\bin\` |
| OS / Shell | Windows 11 Pro, PowerShell 5.1 |

## 4. Approach

Hybrid (Option C from brainstorming): validate each layer **standalone** before wiring into the orchestrator. Rationale: M3 has never run on real data; failures inside `PipelineRunner` are harder to diagnose than failures in a focused script.

Order of execution:

1. **Prep artifacts** — run two existing scripts in sequence to feed M3:
   1. `python scripts/run_vlm_extract.py --video <path> --mode local` → produces `output/Demo-Module-5_vi.srt`.
   2. `python scripts/test_m1_m2_integration.py --srt output/Demo-Module-5_vi.srt --ref-audio data/reference_audio/speaker.wav --output-dir debug_output/m1_m2_integration` → produces `debug_output/m1_m2_integration/audio_chunks/chunk_NNNN.wav` (one per SRT entry).
2. Build `scripts/test_m3_sync.py` to read those artifacts and exercise the three M3 components end-to-end.
3. Audit and patch `src/m3_sync/*.py` based on test findings.
4. Fix the `runner.py:175` `chunk_offset` bug.
5. Wire M3 into `PipelineRunner` (currently the runner stops after M2 voice cloning).
6. Run `scripts/run_pipeline.py` end-to-end and validate output.
7. Refresh `README.md` and create `docs/RUNNING.md`.

## 5. Architecture

### 5.1 Module 3 data flow

```
SRT entries (M1 output) ─┐
                         ├─► AudioAligner.compute_deltas
M2 audio chunks ─────────┘     │
(WAV per segment)              ▼
                         [(seg_idx, srt_start, srt_end, audio_dur, delta), …]
                                       │
                                       ▼
                         TimeStretcher.stretch_to_fit
                         (rubberband; clamp ratio ∈ [0.5, 2.0];
                          fallback: pad with silence or truncate)
                                       │
                                       ▼
                         FFmpegRenderer.merge_audio_segments
                         (concat segments at SRT timestamps; pad gaps with silence)
                                       │
              + original video ────────►  ffmpeg mux  ──►  output/<stem>_vi.mp4
```

### 5.2 Component contracts

These are the **expected** signatures based on `m3_sync/*.py` scaffolding. Step 3 of the approach (audit) will confirm or amend.

| Component | Input | Output |
|---|---|---|
| `AudioAligner.compute_deltas` | list of `{srt_start, srt_end, audio_path}` | list of dicts adding `audio_duration`, `delta`, `target_duration` |
| `TimeStretcher.stretch_to_fit` | `(audio_path, target_duration, tolerance)` | `(stretched_path, strategy: 'exact' \| 'stretch' \| 'pad' \| 'truncate')` |
| `FFmpegRenderer.merge_audio_segments` | list of `(start_time, audio_path)` + `total_duration` | path to single merged WAV |
| `FFmpegRenderer.render_video` | `(video_path, audio_path, output_path)` | path to dubbed MP4 |

### 5.3 New script: `scripts/test_m3_sync.py`

CLI:

```
python scripts/test_m3_sync.py `
    --srt output/Demo-Module-5_vi.srt `
    --audio-dir debug_output/m1_m2_integration/audio_chunks `
    --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
    --output-dir debug_output/m3_test_run
```

Naming contract: audio chunks are matched to SRT entries by filename ordinal (`chunk_0001.wav` → SRT entry index 1). If `test_m1_m2_integration.py` uses different padding, the test script normalizes via sorted listing.

Responsibilities:

1. Load SRT via `SRTBuilder.load_srt()`; pair entries to audio files by index.
2. Run `AudioAligner` → print per-segment deltas.
3. Run `TimeStretcher.stretch_to_fit` per segment → log strategy used (`exact` / `stretch` / `pad` / `truncate`) and ratio.
4. Run `FFmpegRenderer.merge_audio_segments` → produce a single track.
5. Run `FFmpegRenderer.render_video` to mux with the original.
6. Print a summary report: total stretch ratios, any clamped, final output duration vs source video duration, all intermediate file paths.

The script is intentionally **verbose** so the failure point is obvious if any component breaks.

### 5.4 Pipeline runner changes

Two patches to `src/m4_pipeline/runner.py`:

1. **Line 175** — change `chunk_offset=start` to `chunk_offset=0.0`. Rationale: prompts already request absolute timestamps; passing chunk start as offset double-counts and shifts every subtitle later than it should be. `scripts/run_vlm_extract.py` already uses `0.0` and produces correct output; runner must match.
2. **After voice_cloning step** — add two real steps (currently stubs or absent):
   - `audio_alignment`: instantiate `AudioAligner` + `TimeStretcher`, process each TTS output to fit its SRT slot.
   - `video_rendering`: instantiate `FFmpegRenderer`, merge segments, mux with original video.

Both new steps must respect the existing `_update_progress` callback contract for UI integration.

## 6. Validation criteria

The Phase 2.A run is considered successful when **all** of these hold:

- `scripts/test_m3_sync.py` exits 0 with no rubberband / ffmpeg errors.
- Output `Demo-Module-5_vi.mp4` is playable in a standard player (VLC / Windows Media Player).
- Final audio track duration is within ±2.0 s of the original video's 60.4 s.
- No segment was hard-clamped at ratio 0.5 or 2.0 unless flagged in the report (i.e., user is informed).
- Subjective listening test on segment 1 confirms Vietnamese speech is intelligible and matches the cloned reference voice.
- `python scripts/run_pipeline.py --video <path> --ref-audio data/reference_audio/speaker.wav` produces an equivalent file via the orchestrator.

## 7. Error handling

| Failure | Behavior |
|---|---|
| Rubberband fails on a single segment | Log warning, fall back to original (untranstreched) audio, continue. |
| Ratio out of [0.5, 2.0] | `stretch_to_fit` already routes to `pad` (silence) or `truncate`. Log strategy. |
| Audio segment > 2× SRT slot | Truncate to slot, log warning with segment index. |
| FFmpeg mux failure | Surface stderr verbatim, abort. (No silent fallback — bad output is worse than no output.) |
| Missing rubberband on PATH | `TimeStretcher.__init__` already logs warning; the script aborts on first stretch attempt with `FileNotFoundError`. |
| SRT and audio segment counts mismatch | Abort with clear error; this is a contract violation between M1 and M2. |

## 8. Documentation deliverables

### 8.1 `README.md` (Vietnamese, refresh)

Sections to update:

- **Tổng quan** — reflect that pipeline now produces full dubbed video (Phase 2.A done).
- **Yêu cầu hệ thống** — explicit Windows install path for rubberband; ffmpeg path note.
- **Cài đặt** — Windows-specific PowerShell snippets (replacing or supplementing the bash ones).
- **Sử dụng** — concrete CLI example using `Demo-Module-5.mp4` + `speaker.wav`.
- **Trạng thái module** — refresh table: M1 ✅, M2 ✅, M3 ✅ (Phase 2.A), M4 partial, M5 partial, Web UI scaffolding.

### 8.2 `docs/RUNNING.md` (Vietnamese, new)

Detailed first-run walkthrough for thesis reviewers / new developers:

1. Prerequisites checklist (Python 3.10+, CUDA 12.1+ + GPU ≥4 GB VRAM, ffmpeg, rubberband).
2. Cloning + venv creation (PowerShell + bash variants).
3. `pip install -r requirements.txt`.
4. `python scripts/download_models.py` (Qwen3.5-2B + VieNeu-TTS).
5. `.env` setup (which keys are required for local-only mode vs API mode).
6. Test run on `Demo-Module-5.mp4` with `speaker.wav`.
7. Expected output structure (`output/`, `debug_output/`).
8. Troubleshooting: common errors (CUDA OOM, rubberband not on PATH, model download stall, HF token rate limit).

## 9. Deliverables checklist

- [ ] `scripts/test_m3_sync.py` (new)
- [ ] Patches to `src/m3_sync/audio_aligner.py`, `time_stretcher.py`, `ffmpeg_renderer.py` as needed
- [ ] Patch to `src/m4_pipeline/runner.py` (line 175 fix + M3 wiring)
- [ ] `output/Demo-Module-5_vi.mp4` (proof of end-to-end)
- [ ] Updated `README.md`
- [ ] New `docs/RUNNING.md`

## 10. Open risks

| Risk | Mitigation |
|---|---|
| M3 scaffolding signatures don't match this spec's contracts | Step 3 audits and amends before any wiring. |
| Rubberband 4.0.0 CLI flag syntax differs from what `time_stretcher.py` assumes (`--time`, `--pitch`) | Verify with `rubberband --help` during step 2; patch if needed. |
| TTS audio is 24 kHz mono; ffmpeg mux of original video (44.1 / 48 kHz aac) needs resample | `FFmpegRenderer` must declare a unified output rate; verify in audit. |
| 60-s test video is too short to surface chunk-boundary issues | Acceptable for 2.A. Longer-video stress test deferred. |

## 11. Next step after this spec is approved

Invoke the `superpowers:writing-plans` skill to produce a detailed, ordered implementation plan with checkpoints.
