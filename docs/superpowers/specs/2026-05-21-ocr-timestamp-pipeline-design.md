# OCR-Driven Timestamp Pipeline — Design Spec

**Status:** Draft for approval
**Date:** 2026-05-21
**Author:** Quân (with Claude)
**Related code:** `src/m1_vlm/glm_ocr.py`, `src/m4_pipeline/runner.py`, `scripts/run_pipeline.py`

---

## 1. Problem Statement

The current pipeline relies on the **VLM** to assign timestamps to translated narration. The VLM does three things per chunk:

1. *Pass 0 — global summary* (topic, style, key terms across the full video)
2. *Per-chunk* — extract English narration from frames
3. *Per-chunk* — translate to Vietnamese **and** invent timestamps

Step 3 is fragile. Frame timestamps are passed in as "anchor points," but the VLM still hallucinates timing, and `EntryRetimer` post-processes its output with a chars-per-second heuristic. This is the root of most observed SRT drift.

**Observation:** the source tutorial videos already carry **burned-in English captions in the bottom bar**. Those captions are ground truth for both **what was said** and **exactly when**. If we read them with OCR, we get accurate timing for free.

---

## 2. Goals & Non-Goals

### Goals
- Read burned-in English captions with **GLM-OCR** to recover exact `(start, end, text)` for each subtitle.
- Use VLM **only for translation**, with the corresponding mid-segment frame attached so it can disambiguate UI/code references.
- Ship the new path as a **`--mode ocr`** option on the existing `PipelineRunner`. **`--mode vlm` (default) must remain byte-identical to today's behavior.**
- Reuse M2 (TTS), M3 (sync/render) unchanged.

### Non-Goals
- No change to the VLM pipeline behavior at default settings.
- No automatic caption-band detection in this iteration — fixed bottom-crop ratio is enough.
- No multi-line subtitle layout heuristics — emit one SRT entry per detected caption segment.
- No audio ASR fallback — if no captions are detected, we fail loud, not silently fall back to VLM.

---

## 3. High-Level Architecture

### Mode `vlm` (unchanged)
```
scene_detection → frame_extraction → ocr_extraction(noop)
  → vlm_translation (3 tasks: summary + extract + translate+timestamp)
  → srt_generation → voice_cloning → audio_alignment → video_rendering
```

### Mode `ocr` (new)
```
scene_detection                        ← reused
  → dense_frame_sampling (2 fps, full video)
  → caption_crop (bottom band)
  → glm_ocr_per_frame                  ← GLMOCR
  → caption_segmentation               ← group consecutive frames w/ same text
  → vlm_translation_per_segment        ← VLM gets {en_text + mid-frame image}
  → srt_generation                     ← exact OCR timestamps
  → voice_cloning                      ← reused (M2)
  → audio_alignment                    ← reused (M3)
  → video_rendering                    ← reused (M3)
```

Step labels shown above are added to `PipelineRunner.STEPS_OCR` so progress callbacks still work.

---

## 4. Components

### 4.1 `src/m1_vlm/caption_ocr.py` — `CaptionTimeline` (NEW)

Single class encapsulating: sample → crop → OCR → segment.

```python
@dataclass
class CaptionSegment:
    start_sec: float      # exact, from first frame where caption appeared
    end_sec: float        # exact, from last frame before caption changed
    en_text: str          # OCR text (normalized)
    mid_frame: np.ndarray # frame nearest (start+end)/2, for VLM context

class CaptionTimeline:
    def __init__(
        self,
        ocr: GLMOCR,
        sample_fps: float = 2.0,
        caption_band_ratio: float = 0.22,   # crop bottom 22%
        dedup_ratio: float = 0.85,          # Levenshtein ratio threshold
        min_duration_sec: float = 0.3,      # drop blips shorter than this
        ocr_prompt: str = "Read the subtitle text only. Return only the text.",
    ): ...

    def build(self, video_path: Path) -> list[CaptionSegment]:
        """Sample → OCR → segment. Returns ordered list spanning the video."""
```

**Algorithm (build):**
1. Open video, read FPS + duration with OpenCV.
2. Compute target timestamps: `t = 0, 1/fps, 2/fps, ... < duration`.
3. For each timestamp: seek, grab frame, crop bottom `caption_band_ratio` of frame height.
4. Run `GLMOCR.extract_text` on each crop. Normalize: lowercase, collapse whitespace, strip punctuation at edges.
5. Walk the (timestamp, normalized_text) stream. Group consecutive samples where `SequenceMatcher.ratio(prev, curr) >= dedup_ratio` AND both are non-empty.
6. A group becomes a `CaptionSegment`:
   - `start_sec` = first timestamp of group
   - `end_sec`   = next-group's first timestamp (or video end)
   - `en_text`   = OCR text of the **middle** sample of the group (median == most stable)
   - `mid_frame` = stored separately (extracted lazily later to save RAM)
7. Drop segments whose duration < `min_duration_sec`.
8. Return list.

**RAM note:** Frames are released after OCR; we only persist the chosen `mid_frame` per segment.

### 4.2 `src/m1_vlm/prompt_chain.py` — new method

```python
def build_translation_only_prompt(
    self,
    en_text: str,
    chunk_info: str | None = None,
    global_context: str | None = None,
    previous_translations: str | None = None,
) -> str:
    """Translate one EN caption to natural VI voiceover.
    Prompt assumes one frame image will be attached as visual grounding."""
```

Output schema: `{"translated_text": "..."}` — small, easy to parse, no timing field.

### 4.3 `src/m4_pipeline/runner.py` — branch on mode

- Add `STEPS_OCR` list (parallel to `STEPS`) so progress reporting stays correct.
- In `run()`, after loading config, branch:
  ```python
  if self.config.pipeline_mode == "ocr":
      await self._run_ocr_mode(...)
  else:
      await self._run_vlm_mode(...)   # current run() body, factored out unchanged
  ```
- `_run_vlm_mode` is the **verbatim** current body (extracted, not modified). `_run_ocr_mode` is new.

**`_run_ocr_mode` outline:**
```
1. Scene detection                       (reused)
2. CaptionTimeline.build → segments      (NEW)
3. Optional Pass 0 global summary        (reused PromptChain.build_global_summary_prompt)
   - sampled from a handful of segments' mid_frames
4. For each segment:
     prompt = build_translation_only_prompt(seg.en_text, global_context=...)
     vi = await vlm.generate(prompt, images_base64=[encode(seg.mid_frame)])
     parse JSON → vi_text
     entries.append({start_time, end_time, original_text: seg.en_text, translated_text: vi_text})
5. SRTBuilder.add_entries (chunk_offset=0)  → save SRT
6. Free VLM, run TTS / sync / render        (reused as-is)
```

### 4.4 `src/m4_pipeline/config.py` — new properties

```python
@property
def pipeline_mode(self) -> str:
    return os.getenv("PIPELINE_MODE", "vlm")   # "vlm" | "ocr"

@property
def ocr_sample_fps(self) -> float:
    return float(os.getenv("OCR_SAMPLE_FPS", "2.0"))

@property
def caption_band_ratio(self) -> float:
    return float(os.getenv("CAPTION_BAND_RATIO", "0.22"))

@property
def caption_dedup_ratio(self) -> float:
    return float(os.getenv("CAPTION_DEDUP_RATIO", "0.85"))

@property
def caption_min_duration_sec(self) -> float:
    return float(os.getenv("CAPTION_MIN_DURATION_SEC", "0.3"))
```

### 4.5 `scripts/run_pipeline.py` — new flag

```
--mode {vlm,ocr}   (default: vlm)
```

Flag overrides env var if both set. Default = `vlm` preserves all existing invocations.

### 4.6 `.env.example` — document the new vars.

---

## 5. Data Flow

```
video.mp4
   │
   ├─► SceneDetector → chunks (unused in OCR mode for timing, kept for progress UX)
   │
   ├─► CaptionTimeline
   │       │
   │       ├─ sample @ 2 fps   ─► [(t0, frame0), (t1, frame1), ...]
   │       ├─ crop bottom 22%  ─► [(t0, crop0), ...]
   │       ├─ GLM-OCR          ─► [(t0, "hello world"), (t1, "hello world"), ...]
   │       └─ segment+dedup    ─► [CaptionSegment(0.0, 2.5, "hello world", frame@1.25), ...]
   │
   ├─► VLM Pass 0 (optional)   ─► global summary str
   │
   ├─► For each segment:
   │     VLM.translate(en_text + mid_frame, global_ctx) ─► vi_text
   │
   ├─► SRTBuilder ─► video_vi.srt
   │
   └─► TTS ─► AudioAligner ─► FFmpegRenderer ─► video_dubbed.mp4
```

---

## 6. Error Handling

| Failure | Behavior |
|---|---|
| Video unreadable / no frames | `PipelineError("Video has no decodable frames")` — fail fast. |
| GLM-OCR import / load failure | `PipelineError` with hint to set `GLM_OCR_MODEL_PATH` or install transformers. No fallback. |
| OCR returns empty for **all** samples | `PipelineError("No captions detected — video may not have burned-in subtitles; use --mode vlm instead")`. |
| Single OCR call fails | Logged as warning; that sample treated as empty string; segmentation continues. |
| Segment text contains only punctuation/numbers after normalization | Dropped before VLM translation. |
| VLM translation returns malformed JSON | Per-segment raw response dumped to `work/segments_raw/seg_NN.txt`; that segment skipped (logged); pipeline continues. |
| Less than 3 valid translated segments | `PipelineError("Too few segments translated; aborting before TTS")` to avoid wasting compute. |

All errors that abort the pipeline are subclasses of `PipelineError` so existing UI/CLI handling works.

---

## 7. Testing Strategy

Tests live under `tests/m1_vlm/` and `tests/m4_pipeline/`. We use real fixtures where cheap, mocks for GLM-OCR/VLM (the heavy stuff).

### Unit (fast, no models)
- `tests/m1_vlm/test_caption_timeline.py`
  - `test_sampling_timestamps_respect_fps` — synthetic 10s video, fps=2 → expect 20 samples.
  - `test_caption_band_crop_geometry` — feed a known-size frame, assert crop shape == `(H*ratio, W, 3)`.
  - `test_segmentation_groups_identical_text` — mock OCR returns `["A","A","A","B","B"]` → 2 segments.
  - `test_segmentation_dedup_ratio_handles_jitter` — `["hello world","hello wrld","hello world"]` → 1 segment when ratio=0.85.
  - `test_segments_below_min_duration_dropped` — flicker blip filtered.
  - `test_empty_ocr_throughout_returns_empty_list` — runner can then raise.

- `tests/m1_vlm/test_translation_only_prompt.py`
  - Snapshot test: prompt contains EN text, mentions Vietnamese, asks for `{"translated_text": ...}`.

### Integration (cheap, real OpenCV / mocked models)
- `tests/m4_pipeline/test_runner_mode_branch.py`
  - With `pipeline_mode="vlm"`, monkeypatch `_run_vlm_mode` to assert it's the one called.
  - With `pipeline_mode="ocr"`, monkeypatch `_run_ocr_mode` similarly.
  - Confirms zero behavior change for default mode.

### Manual / smoke
- `scripts/test_caption_ocr.py` (new) — point at a known video with burned-in captions, dump segments to stdout, write `captions.srt` for eyeball check. Run inside venv per project convention.

---

## 8. Config Reference (summary)

| Env var | Default | Meaning |
|---|---|---|
| `PIPELINE_MODE` | `vlm` | `vlm` (today) or `ocr` (new). |
| `OCR_SAMPLE_FPS` | `2.0` | Caption sampling rate. |
| `CAPTION_BAND_RATIO` | `0.22` | Fraction of frame height (from bottom) sent to OCR. |
| `CAPTION_DEDUP_RATIO` | `0.85` | SequenceMatcher threshold for "same caption." |
| `CAPTION_MIN_DURATION_SEC` | `0.3` | Drop caption segments shorter than this. |
| `GLM_OCR_MODEL_PATH` | (existing) | HF repo id or local path; reused as-is. |

CLI: `--mode {vlm,ocr}` on `scripts/run_pipeline.py`. Overrides `PIPELINE_MODE` when present.

---

## 9. Open Decisions Locked In

- **VLM role:** translator + frame-aware (gets mid-frame per segment). ✅
- **Sampling:** fixed 2 fps. ✅
- **Caption region:** fixed bottom crop, configurable ratio. ✅
- **Packaging:** mode flag on existing runner; `vlm` default unchanged. ✅

---

## 10. Risks

1. **Captions may not always cover the entire spoken narration** (e.g., intros without subtitles). Acceptable for v1 — segments only exist where captions exist; remainder of audio is silence. Document this.
2. **OCR latency** at 2 fps over a 10-minute video = ~1200 GLM-OCR calls. Need to confirm acceptable wall-clock on the user's hardware. If too slow, the adaptive sampling option (proposed earlier, declined) is the natural follow-up.
3. **Mid-frame for VLM translation** doubles the per-segment payload vs. text-only translation. Mitigation: VLM is called per segment, not per chunk; payload is one image, not eight.

---

## 11. Out of Scope (deferred)

- Auto-detect caption band by clustering OCR bounding-box Y-positions.
- ASR fallback when no captions exist.
- Multi-language captions other than English source.
- Sub-segment splitting (a single very long caption stays as one SRT entry).
