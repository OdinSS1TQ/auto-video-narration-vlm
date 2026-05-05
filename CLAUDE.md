# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Automatic Vietnamese video dubbing pipeline (graduation/DATN project). Takes English video input, produces Vietnamese-dubbed output via VLM translation + zero-shot voice cloning. Written in Vietnamese-accented documentation; code identifiers and comments are in English.

## Commands

```bash
# Install
pip install -r requirements.txt

# Run full pipeline (CLI)
python scripts/run_pipeline.py --video data/raw/sample.mp4 --ref-audio data/reference_audio/speaker.wav

# Run standalone VLM extraction
python scripts/run_vlm_extract.py --video <path> --mode api|local

# Download model weights
python scripts/download_models.py

# Start web UI
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Tests
pytest                          # all tests
pytest tests/test_scene_detector.py  # single file
pytest tests/m1_vlm/            # module-specific tests
pytest -k "test_prompt"         # by name pattern

# Lint
ruff check src/ app/ tests/
black --check src/ app/ tests/
```

## Architecture

5 numbered modules under `src/`, each prefixed `m1_` through `m5_`:

- **`src/m1_vlm/`** — Video → SRT pipeline. Scene detection (PySceneDetect) → adaptive frame extraction → SSIM dedup → VLM inference (Gemini API or local Qwen3.5-2B/Qwen2.5-VL-3B) → SRT generation. Uses a 2-layer context system (global video summary + sliding window) for coherent translation. GLM-OCR 0.9B for on-screen text.

- **`src/m2_tts/`** — Zero-shot voice cloning via VieNeu-TTS v2 Turbo. `TTSClient` handles reference audio encoding → synthesis. `BatchInference` processes segments sequentially (SDK is not thread-safe). `SpeakerEncoder` wraps both vieneu (for synthesis) and resemblyzer (for evaluation embeddings).

- **`src/m3_sync/`** — Audio alignment and video rendering. `AudioAligner` computes timing deltas between generated audio and subtitle timestamps. `TimeStretcher` wraps rubberband CLI for pitch-preserving stretch (ratio clamped 0.5–2.0). `FFmpegRenderer` merges audio segments and renders final video.

- **`src/m4_pipeline/`** — Pipeline orchestrator. `PipelineConfig` loads settings from `.env` + YAML configs. `PipelineRunner` executes 8 async steps: scene_detection → frame_extraction → ocr_extraction → vlm_translation → srt_generation → voice_cloning → audio_alignment → video_rendering. Custom exception hierarchy rooted at `PipelineError`.

- **`src/m5_evaluation/`** — Metrics: BLEU-4/chrF++ (sacrebleu), speaker similarity (resemblyzer cosine), sync accuracy (librosa onset detection), report generation.

- **`app/`** — FastAPI + Gradio web UI. Routes: upload, process (background task), status, download. Gradio mounted at `/ui`.

## Key Patterns

- **Lazy model loading**: All heavy models (VLM, TTS, OCR) initialize on first inference call, not at import time.
- **Dual config**: `.env` for secrets and runtime params, `configs/*.yaml` for structured defaults (default.yaml, gemini.yaml, qwen_local.yaml, tts.yaml).
- **Multi-backend VLM**: API mode (Gemini via google-genai) or local mode (Qwen3.5-2B default, Qwen2.5-VL-3B legacy). Controlled by `VLM_MODE` env var.
- **Sequential TTS**: Batch inference is intentionally sequential — the VieNeu SDK is not thread-safe.
- **Progress callbacks**: Pipeline runner accepts a progress callback for UI integration.

## External Dependencies

System tools required: **FFmpeg** and **rubberband-cli** (CLI, not just Python package). The time stretcher shells out to the `rubberband` command.

## Config Reference

- `.env` / `.env.example` — all runtime settings (API keys, model paths, chunk params, thresholds)
- `configs/default.yaml` — scene detection, chunking, SRT validation defaults
- `configs/tts.yaml` — TTS engine config, voice cloning settings
- `configs/gemini.yaml` / `configs/qwen_local.yaml` — VLM backend-specific settings
