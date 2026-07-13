# Phase 2 Progress Report

**Automatic Vietnamese Video Dubbing Pipeline**
Using Vision-Language Model (VLM) and Zero-shot Voice Cloning

| | |
|---|---|
| **Student** | Ngo Nguyen Tan Quan |
| **Date** | May 27, 2026 |
| **Phase** | 2 — End-to-End Wiring, Timestamp Accuracy, OCR Mode |
| **Period** | April 18 – May 27, 2026 |

---

## 1. Summary

Phase 2 focused on closing the three biggest gaps from Phase 1: (1) modules were scaffolded but never wired into a single runnable pipeline, (2) the pipeline had a timestamp offset bug, and (3) VLM-generated timestamps were inaccurate. By the end of Phase 2, the system can produce a complete English-to-Vietnamese dubbed video through two distinct pipelines: `--mode vlm` (VLM-driven) and `--mode ocr` (OCR-driven for videos with burned-in captions).

---

## 2. Tasks Completed

### 2.1 End-to-End Pipeline (Phase 2.A)

- **Validated Module 3 (Sync/Render) standalone** — Created a standalone test script to verify audio alignment and video rendering work correctly with real TTS output, before wiring them into the orchestrator.
- **Wired all 8 pipeline steps into PipelineRunner** — Scene detection, frame extraction, OCR extraction, VLM translation, SRT generation, voice cloning, audio alignment, and video rendering now run end-to-end in sequence.
- **Fixed the chunk_offset bug** — The Phase 1 known bug at `runner.py:175` caused double-offset timestamps. Fixed to use `chunk_offset=0.0` since prompts emit absolute timestamps.
- **Fixed Windows compatibility issues** — ASCII-sanitized video filenames with special characters (e.g., `【VNEXT】`) for subprocess compatibility; added UTF-8 encoding for subprocess output; supported custom `RUBBERBAND_PATH` env var for the time stretcher.
- **Added VLM memory management** — Pipeline now unloads VLM weights after SRT generation so TTS can fit on 8 GB GPUs.
- **Aligned pipeline behavior with standalone scripts** — Pipeline now uses adaptive frame sampling, SSIM deduplication, lenient VLM JSON parsing, and always dumps raw VLM responses for debugging.
- **Activated Pass 0 global summary** — The video-level context summary (implemented in Phase 1 but never triggered) is now invoked by the pipeline runner in both modes.
- **Produced the first full dubbed video output** — Successfully generated a complete Vietnamese-dubbed video from the test asset `Demo-Module-5.mp4` (60.4 s).
- **Updated documentation** — Refreshed `README.md` and added `docs/RUNNING.md` walkthrough.

### 2.2 SRT Timing Sync (Phase 2.C)

- **Built EntryRetimer** — A post-VLM retiming module that discards the VLM's invented timestamps and reconstructs realistic timing based on chunk duration, frame anchors, and Vietnamese text length (characters-per-second model). This eliminates the pathology where all subtitles were crammed into the first 16 seconds of a 45-second chunk.
- **Rewrote AudioAligner with budgeted stretch** — Instead of compressing every segment to fit unconditionally, the aligner now decides per-segment between four strategies: leave as-is, compress to fit, compress to max speedup with slip, or pad with silence.
- **Implemented slip cascade** — When a TTS segment is too long even after maximum compression, the overflow pushes subsequent segment starts forward instead of forcing unnatural speed-up on every segment.
- **Added capped time stretching** — New `stretch_capped` method in TimeStretcher limits rubberband compression to a configurable maximum speedup (default 1.25x) to preserve speech naturalness.
- **Status: quality still open** — The retimer removes the worst timing problems, but slot widths are still heuristic-based. The VLM mode timestamps remain unsatisfactory for high-quality dubbing, which motivated the OCR approach below.

### 2.3 OCR-Driven Timestamp Pipeline (Phase 2.OCR)

- **Built CaptionTimeline** — A new module that samples video frames at 2 fps, crops the bottom caption band, runs GLM-OCR to read burned-in English captions, and groups consecutive identical readings into timed caption segments. Includes hallucination detection (repetition loops) and normalization of common OCR artifacts.
- **Added narration classifier** — A VLM-driven classifier that labels each OCR segment as `narration` or `screen` (UI labels, code snippets, slide titles), dropping non-narration rows before translation. This prevents the pipeline from dubbing on-screen text that isn't spoken narration.
- **Implemented conservative segment merging** — Handles two patterns: (1) typewriter/animated captions where one segment is a prefix of the next, and (2) tight fragmentation where adjacent short segments are concatenated within configurable thresholds.
- **Added end-time extension** — Each segment's end time is extended by up to 0.5 seconds (capped by the next segment's start) to give TTS breathing room for Vietnamese narration, which runs ~30% longer than English.
- **Integrated OCR mode into PipelineRunner** — Added `--mode ocr` flag that dispatches to a dedicated 8-step OCR pipeline while preserving `--mode vlm` as the unchanged default.
- **Added error handling for OCR-specific failures** — Zero captions detected, too few narration segments, and too few translated segments all raise descriptive errors with actionable hints.

### 2.4 Evaluation Module (Phase 2.E)

- **Fixed SyncAccuracy bug** — `evaluate_segments` was using hardcoded `0.0` instead of the SRT entry's actual `srt_start_sec`. Fixed and added TDD test with a synthetic 440 Hz onset fixture.
- **Added full-mix sync evaluation** — New `evaluate_against_merged_audio` method evaluates a single merged audio file against the full SRT timeline using windowed onset detection, reporting mean delay, p50, p95, and missed onsets.
- **Built translation evaluation wrapper** — `evaluate_translation_from_srts()` computes corpus-level BLEU-4 and chrF++ between a generated SRT and a hand-curated reference SRT, plus sentence-level chrF++ with worst-5 diagnosis.
- **Built voice clone evaluation wrapper** — `evaluate_voice_clone()` computes speaker similarity (resemblyzer cosine) across three pipeline stages: pre-alignment chunks, post-alignment chunks, and merged audio, with per-stage mean/std/p05 statistics.
- **Built caption drift evaluator** — `compare_caption_tracks()` matches predicted vs ground-truth caption entries by fuzzy text similarity within a time window, reporting start/end drift (mean/p50/p95), Jaccard overlap, and missed/extra counts.
- **Extended ReportGenerator** — Added `generate_markdown()` for formatted Markdown reports with Summary table, Translation, Voice Clone, Sync, Caption Drift, and Config Snapshot sections (skips empty sections). Extended `generate_summary_csv()` with new columns for pipeline mode, merged speaker similarity, sync p95, and caption drift p95.
- **Created evaluation CLI** — `scripts/run_evaluation.py` orchestrates all evaluation wrappers via command-line arguments, generating JSON + Markdown + CSV reports with a full PipelineConfig snapshot (20 knobs).
- **13 evaluation tests** — All passing: sync accuracy fix (2), caption drift (6), translation eval (1), report markdown (2), CLI smoke test (2).

### 2.5 Audio & OCR Pipeline Fixes (May 26–27)

- **Fixed audio volume normalization** — FFmpegRenderer's `amix` filter was dividing volume by the number of input segments (e.g., 20 segments → 1/20th volume). Added `normalize=0` since time-separated segments don't actually overlap, restoring full output volume.
- **Fixed frame capture accuracy** — Replaced seek-based frame sampling (`CAP_PROP_POS_MSEC`) in `iter_video_samples` with sequential decode. The seek approach snapped to the nearest keyframe, causing multiple timestamps to return the same frame and missing real caption transitions during animations.
- **Increased OCR sample rate** — Default `OCR_SAMPLE_FPS` bumped from 2.0 to 3.0 (one frame every 0.33s instead of 0.5s) to capture shorter-lived captions and animation transitions.
- **Added identical-text dedup to segment merging** — New strategy in `merge_short_segments` detects when adjacent segments have identical or near-identical English text (SequenceMatcher ratio ≥ 0.85) across gaps up to ~7.5s, collapsing them into one segment before VLM translation. Prevents duplicate translations caused by animation cycles where the same caption appears, fades, and reappears.

### 2.6 Testing

- **Added 7 new test files** covering:
  - EntryRetimer (distribution, weighted slots, snap to frame timestamps)
  - AudioAligner budgeted stretch (all four strategies and slip cascade)
  - TimeStretcher capped compression (decision matrix)
  - CaptionTimeline (sampling, cropping, grouping, dedup, minimum duration)
  - Translation-only prompt (format validation)
  - Runner mode branching (VLM vs OCR dispatch)
  - OCR-mode config properties (env var defaults)
- **Created manual smoke test script** for CaptionTimeline that dumps segments and SRT for visual inspection.

---

## 3. Module Status After Phase 2

| Module | Status | Change from Phase 1 |
|---|---|---|
| **M1 — VLM Extraction** | Functional (both modes) | EntryRetimer added; CaptionTimeline + narration classifier + merge/extend added; GLM-OCR actively used in OCR mode; Pass 0 wired into runner; sequential frame decode fix; sample FPS 2→3; identical-text dedup in merge |
| **M2 — TTS** | Unchanged | Reused by both modes via BatchInference |
| **M3 — Sync/Render** | Functional | AudioAligner rewritten with budgeted stretch + slip cascade; capped time stretching added; Windows fixes; amix volume normalization fix |
| **M4 — Pipeline** | Functional | Mode dispatcher added; chunk_offset bug fixed; VLM memory management; ASCII path sanitization |
| **M5 — Evaluation** | Functional | SyncAccuracy bug fixed + full-mix variant; 3 new wrappers (translation, voice, caption-drift); ReportGenerator extended with Markdown emitter; `scripts/run_evaluation.py` CLI; 13 tests |
| **Web App** | Unchanged | Not touched in Phase 2 |

---

## 4. Known Issues

| Issue | Severity |
|---|---|
| `--mode vlm` timestamps still inaccurate (heuristic-based); OCR mode is more reliable for videos with burned-in captions | Medium |
| ~~OCR-mode quality not quantitatively measured~~ — **Resolved:** caption-drift evaluator + translation metrics now available | ~~High~~ Done |
| Hardcoded caption-band ratio (0.10) only works for one video layout | Medium |
| ~~M5 evaluation module never invoked end-to-end~~ — **Resolved:** `scripts/run_evaluation.py` CLI orchestrates all M5 wrappers | ~~Medium~~ Done |
| Only tested on one video (`Demo-Module-5.mp4`, 60.4 s) | Medium |
| Videos with few burned-in captions produce too few OCR segments (e.g., Demo-Module-1: 2 segments, 1 narration) | Medium |
| Web UI not exercised with real pipeline | Low |

---

## 5. Phase 3 — Remaining Tasks (May 26 – June 20, 2026)

### 5.1 Evaluation & Metrics (Priority: High)

- [x] Wire Module 5 evaluation into the pipeline runner for automated quality reporting — `scripts/run_evaluation.py`
- [x] Run BLEU-4 and chrF++ scoring against a hand-curated Vietnamese reference translation — `evaluate_translation_from_srts()`
- [x] Measure speaker similarity (resemblyzer cosine) between reference audio and dubbed output — `evaluate_voice_clone()`
- [x] Measure sync accuracy (librosa onset detection) between subtitle timestamps and dubbed audio — `SyncAccuracy.evaluate_against_merged_audio()`
- [x] Measure OCR-mode timestamp drift against ground-truth captions (mean/median/p95) — `compare_caption_tracks()`
- [ ] Measure narration classifier precision/recall

### 5.2 Multi-Video Validation (Priority: High)

- [ ] Test on at least 2–3 additional non-audio narrated videos of varying length
- [ ] Test on a longer captioned video (stresses OCR latency and segment merging)
- [ ] Document failures and edge cases found

### 5.3 Quality Improvements (Priority: Medium)

- [ ] Calibrate `VI_CHARS_PER_SEC` per-voice by measuring TTS output speaking rate, replacing the hardcoded 15.0
- [ ] Auto-detect caption band position by clustering OCR bounding-box Y-positions, replacing the hardcoded `CAPTION_BAND_RATIO`
- [x] Tune OCR-mode parameters (sampling FPS 2→3, sequential frame decode, identical-text dedup in merge) based on evaluation results

### 5.4 Web UI Integration (Priority: Medium)

- [ ] Surface `--mode {vlm, ocr}` selection in the Gradio interface
- [ ] Wire pipeline progress callbacks to the UI for real-time status
- [ ] Test full upload → process → download flow through the web interface

### 5.5 Documentation & Thesis Writing (Priority: High)

- [ ] Generate evaluation report with all metrics for thesis appendix
- [ ] Document system architecture and design decisions
- [ ] Prepare demo video(s) showing the full pipeline in action
- [ ] Complete thesis write-up

### 5.6 Nice-to-Have (if time permits)

- [ ] Profile OCR mode performance on 10+ minute videos
