# Phase 1 Progress Report

**Automatic Vietnamese Video Dubbing Pipeline**
Using Vision-Language Model (VLM) and Zero-shot Voice Cloning

| | |
|---|---|
| **Student** | Ngo Nguyen Tan Quan |
| **Date** | April 17, 2026 |
| **Phase** | 1 — VLM Extraction & TTS Voice Cloning |
| **Period** | March 31 – April 17, 2026 |

---

## 1. Executive Summary

Phase 1 focused on two core modules: **Module 1 (VLM extraction)** — building the pipeline to extract Vietnamese subtitles from English video using Vision-Language Models, and **Module 2 (TTS voice cloning)** — implementing zero-shot Vietnamese speech synthesis. These two modules have been developed and tested independently; they have not yet been connected end-to-end.

**What was accomplished:**

- Built a complete video-to-SRT pipeline with scene detection, adaptive frame extraction, SSIM deduplication, multi-backend VLM inference, and structured prompt engineering
- Evaluated and iterated on multiple VLM models (Gemini, Qwen2.5-VL-3B, Qwen3.5-2B) before selecting Qwen3.5-2B as the default local model
- Designed a 2-layer context system and anti-hallucination prompting to produce coherent cross-chunk translations
- Integrated and tested VieNeu-TTS v2 Turbo for zero-shot Vietnamese voice cloning with comprehensive test coverage (24 tests)
- Identified and partially addressed a timestamp alignment issue in chunk-based extraction

**What remains incomplete:**

- Timestamp accuracy in generated SRT (timespans not fully resolved)
- Connecting M1 output (SRT) to M2 input (TTS segments)
- Modules 3–5 (sync, pipeline orchestrator, evaluation) exist as scaffolding but have not been tested
- End-to-end pipeline has not been run

---

## 2. Technical Decisions

### 2.1 VLM Model Selection

The project evaluated three VLM backends before settling on the current architecture:

| Model | VRAM | Context | Pros | Cons | Role |
|-------|------|---------|------|------|------|
| **Gemini 1.5 Flash** | Cloud | 1M tokens | Best quality, handles video natively | Rate limits, requires API key, latency | API mode (development/benchmarking) |
| **Qwen2.5-VL-3B-Instruct** | ~6 GB | 8K tokens | First local model tested, decent quality | High VRAM, requires `qwen_vl_utils`, 8K context too small for long videos | Legacy (deprecated) |
| **Qwen3.5-2B** | ~4 GB | 262K tokens | Low VRAM, simple API, massive context window | Smaller model, may hallucinate more than Gemini | **Default local model** |

**Why Qwen3.5-2B was chosen over Qwen2.5-VL-3B:**

1. **Lower VRAM** — 4GB vs 6GB, accessible on consumer GPUs (GTX 1660, RTX 3050, etc.)
2. **Simpler API** — Uses `AutoModelForImageTextToText` (standard Transformers), no `qwen_vl_utils` dependency needed
3. **Larger context window** — 262K tokens vs 8K tokens, critical for processing longer video chunks with full frame context
4. **Forward-compatible** — Uses the universal chat template API, making it easier to swap to other Qwen3.5 variants (e.g., 0.8B for ultra-lite)

**Model auto-detection:** `VLMClient` detects the model family from the path string. If the path contains "qwen2.5" or "qwen2-vl", it loads via the legacy `Qwen2_5_VLForConditionalGeneration` path. Otherwise, it uses the generic `AutoModelForImageTextToText` path (Qwen3.5).

**Generation parameters (from Qwen3.5 model card):**

```
top_p=0.8, top_k=20, repetition_penalty=1.1, max_new_tokens=4096
```

### 2.2 Structured Prompt Design

The prompting architecture (`PromptChain`) went through several iterations. The final design uses **structured JSON output** with embedded constraints and anti-hallucination techniques.

#### Prompt Modes

**Mode 1 — Single-prompt (default):** All-in-one extraction + translation + timestamping in a single VLM call. Used for Gemini (large context) and Qwen3.5 (262K context). More efficient, reduces error accumulation across steps.

**Mode 2 — 3-step chain:** Separate calls for extract → translate → format. Available via `--mode 3step`. Better for models with smaller context windows where combining all tasks degrades quality.

#### Anti-Hallucination Techniques

The prompts incorporate several strategies to prevent the VLM from generating inaccurate content:

1. **Frame timestamp anchoring** — The prompt lists exact frame timestamps as anchor points. The VLM is instructed: *"Your subtitle timestamps MUST fall within [chunk_start, chunk_end]"* and *"Each subtitle's start_time should be near a frame timestamp."* This grounds the model's temporal reasoning to actual visual evidence.

2. **Absolute timestamps** — Prompts explicitly state: *"Generate timestamps in ABSOLUTE time (relative to video start, NOT relative to chunk start)."* This was a fix applied after discovering that relative timestamps caused double-offset errors when assembling chunks.

3. **Anti-repeat injection** — Previous chunk translations are injected with explicit `"ALREADY GENERATED — DO NOT REPEAT"` labels, preventing the VLM from re-generating content from overlapping chunks.

4. **"HIGH-LEVEL OVERVIEW ONLY" instruction** — The prompt tells the VLM to describe PURPOSE and TOPIC rather than specific UI actions (e.g., "clicking the File menu"). This prevents over-detailed narration.

5. **Structured output schema** — All prompts request a JSON array with a strict schema:

```json
{
  "index": 1,
  "start_time": "00:01:23,456",
  "end_time": "00:01:26,789",
  "original_text": "English narration",
  "translated_text": "Vietnamese narration"
}
```

#### SRT Constraints Embedded in Prompt

The prompts encode subtitle best practices directly into the VLM's instructions:

- Subtitle duration: 2–7 seconds per entry
- Maximum 2 lines per subtitle
- Maximum 42 characters per line
- Maximum 25 characters/second reading speed
- No overlapping timestamps

#### Translation Rules

Specific rules for translating English technical content to Vietnamese:

- **Code/commands:** Keep in English (e.g., `pip install`, `def function()`)
- **Technical terms:** Keep English with Vietnamese explanation on first use (e.g., "thuật toán Machine Learning (Học máy)")
- **UI/button text:** Translate explanation, keep label in English (e.g., "nhấn nút Save để lưu")
- **Style:** Conversational Vietnamese, professional spoken style suitable for narration

### 2.3 Context System Design

A 2-layer hierarchical context system prevents the "chunk isolation" problem where each chunk is translated without awareness of neighboring content.

**Layer 1 — Global Summary (Pass 0):**
- 15 frames sampled uniformly across the entire video
- VLM generates: topic, style, sections, key_terms, overall summary
- Set once, injected into every chunk's prompt
- Provides the VLM with video-level context regardless of which chunk it's processing

**Layer 2 — Sliding Window:**
- Maintains a window of the last 3 chunk results (~135 seconds of context)
- Shows ALL previous subtitle entries with "DO NOT REPEAT" markers
- Automatically extracts terminology glossary (original → translated pairs)
- Prevents duplicate translations and inconsistent terminology across chunks

### 2.4 TTS Engine Selection

**VieNeu-TTS v2 Turbo** was chosen for zero-shot Vietnamese voice cloning:

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Engine** | VieNeu-TTS v2 Turbo | Native Vietnamese support, zero-shot cloning |
| **Backend** | GGUF + ONNX | No PyTorch/Transformers needed at runtime, smaller footprint |
| **Reference audio** | 3–5 seconds | Balance between cloning quality and user convenience |
| **Processing** | Sequential batches | VieNeu SDK is not thread-safe; sequential avoids race conditions |
| **Caching** | SHA1-based in-memory ref_codes | Reference encoding is expensive; caching avoids re-encoding the same audio |
| **Output** | float32, 24kHz mono | Standard format compatible with downstream audio processing |

**Previously evaluated (rejected):**
- **F5-TTS:** Reserved as engine slot but not implemented. Would require PyTorch.
- **viXTTS:** Reserved as engine slot but not implemented. Would require PyTorch + XTTS model.

### 2.5 Video Processing Pipeline

**Scene detection:** PySceneDetect `ContentDetector` with threshold 27.0. Splits video into processing chunks (default 45s) with 5-second overlap to prevent narration gaps at chunk boundaries.

**Frame extraction:** Adaptive sampling based on scene structure:
- Short scenes (< 5s): onset frame + midpoint
- Long scenes (> 5s): onset + midpoint + intermediate frames
- Min/max constraints: 3–30 frames per chunk

**SSIM deduplication:** Custom fast SSIM (no scikit-image dependency). Downscales to 256x256 for speed. Compares against the LAST KEPT frame (not the previous frame) to prevent gradual drift. Typically removes 40–70% of frames.

---

## 3. Module Status

### 3.1 Module 1: VLM Extraction — Complete (timespan issue pending)

The VLM pipeline is functional and has been tested with real videos via `scripts/run_vlm_extract.py`.

| Component | Status | Description |
|-----------|--------|-------------|
| SceneDetector | ✅ Done | PySceneDetect, chunk splitting, adaptive timestamps |
| FrameExtractor | ✅ Done | Base64 encoding, uniform sampling, timestamp-based extraction |
| FrameDeduplicator | ✅ Done | Custom SSIM, 40–70% frame reduction |
| VLMClient | ✅ Done | Multi-backend: Gemini API + Qwen3.5-2B (default) + Qwen2.5-VL-3B (legacy) |
| PromptChain | ✅ Done | Single-prompt + 3-step chain, anti-hallucination, translation rules |
| ContextWindow | ✅ Done | 2-layer: global summary + sliding window |
| SRTBuilder | ✅ Done | Build, deduplicate (time + text), validate, save |
| SubtitleValidator | ✅ Done | Format, overlap, duration validation + auto-fix |
| GLMOCR | ⚠️ Implemented but unused | GLM-OCR 0.9B for on-screen text; not integrated into pipeline |

**Known issue — Timestamp accuracy:**

The generated SRT timestamps are not fully accurate. The initial approach generated timestamps relative to chunk start, causing double-offset errors when assembling chunks. This was partially addressed by:
1. Changing prompts to request absolute timestamps
2. Setting `chunk_offset=0.0` in the extraction script
3. Adding SRT deduplication for overlapping chunk entries

However, timestamp precision still needs improvement — the VLM sometimes generates inaccurate start/end times, especially for narration without clear visual cues.

**Test coverage:**
- 8 formal pytest tests in `tests/m1_vlm/`
- Comprehensive verification script: `scripts/verify_m1_vlm.py` (535 lines)
- Production extraction script: `scripts/run_vlm_extract.py` (630 lines)

### 3.2 Module 2: TTS Voice Cloning — Complete (standalone, not connected to M1)

The TTS module is fully implemented and tested independently using synthetic reference audio. It has NOT been connected to Module 1's SRT output.

| Component | Status | Description |
|-----------|--------|-------------|
| TTSClient | ✅ Done | VieNeu-TTS v2 Turbo, voice cloning, preset voices, ref_codes caching |
| SpeakerEncoder | ✅ Done | Dual-path: VieNeu (synthesis) + resemblyzer (evaluation embeddings) |
| BatchInference | ✅ Done | Sequential processing, one-time reference encoding, error tolerance |

**Test coverage (24 tests):**

`scripts/test_m2_tts.py` (936 lines):

| Category | Count | What's tested |
|----------|-------|---------------|
| Unit tests (no GPU) | 12 | Import, instantiation, text normalization (8 Vietnamese edge cases), cache key stability, engine validation, cosine similarity math, batch preparation (6 scenarios) |
| Integration tests (GPU) | 12 | Model load, preset voice synthesis, ref_codes cache verification (timing), voice-cloned synthesis to file + ndarray, context manager, SpeakerEncoder delegation, BatchInference.process_all (4 segments), sequential synthesis (4 rounds), reference duration variants (2s/3s/5s/8s) |

`scripts/test_vieneu_tts.py` (195 lines): Quick smoke test for import chain, lazy loading, preset synthesis, reference encoding, and voice cloning.

### 3.3 Modules 3–5 & Web App — Scaffolded, Not Tested

These modules have code structure in place but have not been tested or connected:

| Module | Status | What exists |
|--------|--------|-------------|
| **M3: Sync** | Scaffolded | AudioAligner, TimeStretcher (rubberband), FFmpegRenderer |
| **M4: Pipeline** | Scaffolded | PipelineRunner (8-step async), PipelineConfig, exceptions, logger |
| **M5: Evaluation** | Partial | BLEUScorer ✅, SpeakerSimilarity ✅, SyncAccuracy ✅, MOSEstimator ❌ (skeleton) |
| **Web App** | Scaffolded | FastAPI routers (upload, process, status, download) + Gradio UI |

---

## 4. Configuration Architecture

The project uses a dual configuration system:

**`.env` file** — Secrets and runtime parameters:
- API keys (Gemini)
- Model paths (Qwen, GLM-OCR, TTS)
- VLM mode selection (api/local)
- Processing parameters (chunk duration, thresholds, concurrency)

**`configs/*.yaml`** — Structured defaults:
- `default.yaml` — Scene detection (threshold 27.0), chunking (45s, 5s overlap), SRT bounds
- `gemini.yaml` — Gemini API model, generation params, rate limits, safety settings
- `qwen_local.yaml` — Qwen3.5-2B defaults (~4GB VRAM), inference params, memory optimization
- `tts.yaml` — VieNeu-TTS v2 Turbo config, voice cloning settings, audio post-processing

---

## 5. Development Timeline

| Date | Commit | What Changed |
|------|--------|--------------|
| Mar 31 | `f5f70a0` | Project setup + Module 1 scaffolding. Gemini API mode + Qwen2.5-VL-3B as local option. |
| Apr 13 | `69fc1b3` | Qwen2.5-VL-3B active development. Script + timespan generation from video. First real VLM inference. |
| Apr 14 | `5705a3a` | Switched default to **Qwen3.5-2B** (~4GB VRAM). Updated prompts for absolute timestamps. Simplified model loading API. |
| Apr 17 | *(unstaged)* | Module 2 rewrite: F5-TTS/viXTTS scaffolding → VieNeu-TTS v2 Turbo integration. 24 TTS tests. Pipeline runner updated but not tested end-to-end. |

---

## 6. Known Issues & Open Questions

| Issue | Severity | Description |
|-------|----------|-------------|
| **Timestamp accuracy** | High | SRT timestamps generated by VLM are not precise. Absolute time instructions help but the model still struggles with exact timing, especially when narration has no clear visual anchor. |
| **M1 → M2 not connected** | High | SRT output from M1 has not been fed into M2 for TTS synthesis. The integration point needs mapping from SRT entries to TTS segments. |
| **Pipeline runner uses chunk_offset** | Medium | `runner.py` line 175 still uses `chunk_offset=start`, while `run_vlm_extract.py` correctly uses `chunk_offset=0.0`. Running the full pipeline through the orchestrator would produce double-offset timestamps. |
| **Pass 0 not used in pipeline** | Medium | Global summary (Pass 0) is implemented in ContextWindow but not triggered by the pipeline runner. |
| **GLM-OCR skipped** | Low | OCR module exists but is not integrated. Current VLM-only approach works without it. |
| **MOS estimation missing** | Low | Speech quality assessment is a skeleton. BLEU/chrF++ and speaker similarity are functional. |

---

## 7. Next Steps

### Immediate (Phase 2)

1. **Fix timestamp accuracy** — Evaluate whether absolute timestamps are sufficient or if a post-processing alignment step is needed
2. **Connect M1 → M2** — Build the integration layer that maps SRT entries to TTS segments and runs voice cloning on each subtitle
3. **End-to-end test** — Run a full video through M1 → M2 → M3 (sync/render) and validate output quality
4. **Enable Pass 0 global summary** — Integrate the global video summary into the pipeline for better context

### Short-term

5. **Real reference audio testing** — Test TTS with actual human voice recordings instead of synthetic WAV
6. **Module 3 integration** — Test audio alignment and time stretching with real TTS output
7. **Module 5 evaluation** — Run metrics on actual dubbed output; implement MOSEstimator

### Longer-term

8. **Web UI deployment** — Test the FastAPI + Gradio interface with real usage
9. **Performance optimization** — Profile and optimize VLM inference, TTS batch processing
10. **Production readiness** — Persistent storage, error handling, Docker containerization
