# Phase 2 Closeout — Quantitative Evaluation + Second-Video Validation

**Date:** 2026-05-25
**Status:** Draft for approval
**Author:** Quân (with Claude)
**Related code:** `src/m5_evaluation/*`, `src/m1_vlm/caption_ocr.py`, `scripts/run_pipeline.py`
**Supersedes (in scope):** items 1–3 + 6 of `Phase1_Progress_Report.md` §7 "Next Steps"; §9 "Next Steps → Immediate" of `Phase2_Progress_Report.md`.

---

## 1. Problem Statement

Phase 2 shipped three design specs (2.A end-to-end, 2.C timing-sync, OCR mode) but the project still cannot answer the basic thesis-defense questions:

- *How good is the Vietnamese translation?* — no BLEU / chrF++ numbers exist for any output.
- *Does the cloned voice actually sound like the reference?* — `SpeakerSimilarity` is implemented but never invoked.
- *Is the audio in sync with the picture?* — `SyncAccuracy.evaluate_segments` has a bug (calls `compute_delay(0.0, audio_path)` — `expected_start` is never used) and has never been run on real output.
- *Does the OCR-mode timeline match the burned-in captions?* — `--mode ocr` is the project's preferred path for timing, but no drift / precision / recall numbers exist.
- *Does the pipeline work on more than one video?* — every Phase 2 run has been against the same 60-second `Demo-Module-5.mp4`.

The VLM-mode timing accuracy gap is **out of scope** here. The user has decided OCR mode is the operational answer for now and that the residual VLM-mode error can be tuned later.

---

## 2. Goals & Non-Goals

### Goals

- Produce a single JSON + Markdown evaluation report per dubbed video that contains: **BLEU-4 + chrF++** (translation), **mean / p50 / p95 speaker cosine similarity** (voice clone), **mean / p95 onset offset** (sync accuracy), and **caption-drift mean / p95** (OCR-mode timing only).
- Fix the broken `SyncAccuracy.evaluate_segments` so its result reflects actual delay between SRT timestamps and detected speech onset.
- Add an **OCR drift metric** that compares `CaptionTimeline` segments against a hand-curated ground-truth caption-track for the test video.
- Wire all metrics into a single CLI: `python scripts/run_evaluation.py --video … --srt … --dubbed … [--mode-ocr-segments …] [--ground-truth …]`.
- Run the harness end-to-end on `Demo-Module-5.mp4` (already dubbed) **and one new test video** with burned-in captions. Persist both reports under `docs/eval_results/`.
- Replace `MOSEstimator` skeleton with a documented **deferred status** — leave the file but raise `NotImplementedError` with a clear comment and a follow-up tracked in `Phase3_Plan.md` (do **not** ship a half-real MOS).

### Non-goals

- Improving VLM-mode timestamp accuracy. Deferred per user direction.
- Implementing UTMOS / DNSMOS / NISQA. Deferred to Phase 3.
- Web UI integration of evaluation results. Deferred.
- Production hardening (Docker, persistent job state).
- Translation reference larger than the test videos. We hand-curate ≤ 60 lines per video; corpus-scale eval is Phase 3.

---

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  scripts/run_evaluation.py                       │
│                                                                  │
│   --video       (source mp4)                                     │
│   --srt         (generated Vietnamese SRT — vlm or ocr mode)     │
│   --dubbed      (final dubbed audio or video)                    │
│   --ref-audio   (reference voice WAV used for cloning)           │
│   --ground-truth-srt  (optional: hand-curated VI ref for BLEU)   │
│   --ground-truth-captions  (optional: hand-curated EN captions   │
│                              for OCR drift measurement)          │
│   --ocr-segments-json (optional: work_ocr/segments_final.json    │
│                          to evaluate OCR-mode timing directly)   │
│                                                                  │
│   ── orchestrates ─►                                             │
│      ├─ TranslationEval  (BLEUScorer)                            │
│      ├─ VoiceEval        (SpeakerSimilarity)                     │
│      ├─ SyncEval         (SyncAccuracy — fixed)                  │
│      └─ CaptionDriftEval (NEW)                                   │
│                                                                  │
│   ── emits ─►  docs/eval_results/<video_stem>_report.json        │
│                docs/eval_results/<video_stem>_report.md          │
│                docs/eval_results/evaluation_summary.csv          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Components

### 4.1 `src/m5_evaluation/sync_accuracy.py` — fix the bug

The current `evaluate_segments` calls `self.compute_delay(0.0, audio_path)` — the `expected_start` arg is hardcoded to `0.0`, which means the function measures **"where does speech start inside this chunk WAV?"** rather than **"how far is the actual audio onset from the SRT-declared start?"**. Fix the call:

```python
expected = float(segment.get("srt_start_sec", segment.get("start_sec", 0.0)))
delay = self.compute_delay(expected, audio_path)
```

…and update `compute_delay` to operate on the dubbed video timeline rather than the chunk-local timeline. Two modes:

| Mode | Input | What's measured |
|---|---|---|
| **Per-chunk** (existing, fixed) | `(srt_start_sec, chunk_audio_path)` | Chunk-internal onset relative to SRT start. Useful for diagnosing batch-TTS gaps. |
| **Full-mix** (NEW) | `(srt_start_sec, merged_audio_path)` | Walk SRT entries; detect onset within `[srt_start − 1s, srt_end + 1s]` window of the *merged* dubbed audio; report signed delay. Useful for diagnosing slip-cascade behavior. |

The full-mix variant is added as a new method `evaluate_against_merged_audio(srt_entries, merged_audio_path)` returning per-entry delay + `{mean, p50, p95, n_missed}` aggregates. `n_missed` counts entries with no onset detected in the window.

### 4.2 `src/m5_evaluation/caption_drift.py` — NEW

Compares two `[(start_sec, end_sec, text)]` sequences (predicted vs ground truth) for OCR-mode timing evaluation.

```python
@dataclass
class DriftStats:
    n_matched: int
    n_missed_in_pred: int      # GT entry has no pred match
    n_extra_in_pred: int       # pred entry not in GT
    start_drift_mean: float
    start_drift_p50: float
    start_drift_p95: float
    end_drift_mean: float
    duration_jaccard_mean: float  # |overlap| / |union| in time

def compare_caption_tracks(
    pred: list[CaptionEntry],
    truth: list[CaptionEntry],
    text_match_ratio: float = 0.7,
    time_match_window_sec: float = 1.5,
) -> DriftStats: ...
```

Matching algorithm: greedy from left-to-right. For each GT entry, find the predicted entry whose start is within `time_match_window_sec` and whose text passes `SequenceMatcher.ratio(...) >= text_match_ratio`. First match wins; matched predicted entries are not re-considered. Records unmatched GT entries as `n_missed_in_pred`. Remaining unmatched predicted entries are `n_extra_in_pred`.

Input formats:
- Predicted: either an SRT file (parse via `SRTBuilder.load_srt`) or `work_ocr/segments_final.json`.
- Ground truth: an SRT file (the user produces ≤ 60 hand-curated lines from VTT/closed captions).

### 4.3 `src/m5_evaluation/translation_eval.py` — NEW (thin wrapper)

Wraps the existing `BLEUScorer` with SRT input/output and sentence-level breakdowns.

```python
def evaluate_translation_from_srts(
    hypothesis_srt: Path,
    reference_srt: Path,
) -> dict:
    """Match hypothesis entries to reference entries by index, compute
    corpus BLEU-4 + chrF++ over translated_text fields, plus the worst-5
    sentences by sentence-level chrF++."""
```

Matching by index assumes both SRTs have the same entry order (true when both come from the same source via OCR mode or hand-curation against the same source). If lengths differ, log a warning and truncate to `min(len(hyp), len(ref))`.

### 4.4 `src/m5_evaluation/voice_eval.py` — NEW (thin wrapper)

Wraps `SpeakerSimilarity.evaluate_batch` to operate on the pipeline's `work/audio_chunks/` (raw TTS output) and `work/aligned_audio/` (post-stretch) directories, plus a separate similarity on the final merged audio. Returns:

```python
{
    "n_chunks": int,
    "pre_align": {"mean": float, "std": float, "p05": float},   # raw TTS
    "post_align": {"mean": float, "std": float, "p05": float},  # after rubberband
    "merged":     {"similarity": float},                         # full dubbed mix
}
```

Tracking pre vs post catches cases where the rubberband stretch degrades speaker identity (a known artifact at `MAX_SPEEDUP = 1.25`).

### 4.5 `src/m5_evaluation/report_generator.py` — extend

`generate_report` already accepts `translation_metrics`, `mos_metrics`, `similarity_metrics`, `sync_metrics`. Add a `caption_drift_metrics` slot and a Markdown emitter:

```python
def generate_report(
    self, video_name, *,
    translation_metrics=None, similarity_metrics=None, sync_metrics=None,
    caption_drift_metrics=None, mos_metrics=None,
    pipeline_mode: str = "unknown",      # "vlm" | "ocr"
    pipeline_elapsed_sec: float = 0.0,
    pipeline_config_snapshot: dict | None = None,
) -> dict: ...

def generate_markdown(self, report: dict) -> Path:
    """Write a human-readable Markdown next to the JSON."""
```

Markdown layout: one section per metric family, a top-line summary table, and the pipeline config snapshot at the bottom so a reviewer can see exactly which `M3_MAX_SPEEDUP`, `CAPTION_BAND_RATIO`, etc. produced the numbers.

`generate_summary_csv` adds new columns: `pipeline_mode`, `caption_drift_p95`, `speaker_sim_merged`, `sync_delay_p95`.

### 4.6 `scripts/run_evaluation.py` — NEW CLI

```bash
python scripts/run_evaluation.py \
  --video data/raw/Demo-Module-5.mp4 \
  --srt data/outputs/Demo-Module-5/Demo-Module-5_vi.srt \
  --dubbed data/outputs/Demo-Module-5/Demo-Module-5_dubbed.mp4 \
  --ref-audio data/reference_audio/speaker.wav \
  --audio-chunks data/outputs/Demo-Module-5/work_ocr/audio_chunks \
  --aligned-audio data/outputs/Demo-Module-5/work_ocr/aligned_audio \
  --merged-audio data/outputs/Demo-Module-5/work_ocr/merged_audio.wav \
  --ground-truth-srt data/references/Demo-Module-5_vi_truth.srt \
  --ground-truth-captions data/references/Demo-Module-5_en_truth.srt \
  --ocr-segments-json data/outputs/Demo-Module-5/work_ocr/segments_final.json \
  --pipeline-mode ocr \
  --output-name Demo-Module-5
```

Behavior:
- Any `--ground-truth-*` flag omitted ⇒ skip that metric family with a logged INFO message (not an error).
- `--ocr-segments-json` + `--ground-truth-captions` together ⇒ run `CaptionDriftEval`.
- `--ground-truth-srt` ⇒ run `TranslationEval`.
- `--audio-chunks` + `--ref-audio` ⇒ run `VoiceEval` (always available).
- `--merged-audio` + `--srt` ⇒ run `SyncEval` full-mix variant.

Exits 0 if at least one metric family produced a result; exits 1 if every family was skipped or errored.

### 4.7 Hand-curated reference data

Two files committed under `data/references/`:

| File | Source | Size |
|---|---|---|
| `Demo-Module-5_en_truth.srt` | Burned-in captions transcribed manually | ≤ 60 entries |
| `Demo-Module-5_vi_truth.srt` | Hand-translated by the user using `_en_truth.srt` as source | same N |

Both should be checked into the repo. They are the only labels we ship — for the second test video, the same pair must be produced before evaluation.

### 4.8 Second test video

User-supplied second video (with burned-in English captions, ≤ 5 minutes). It must be:

- Different domain from Demo-Module-5 (e.g., a software tutorial of a different product, or a different style of presenter).
- Have visible burned-in captions in the bottom band (so OCR mode is exercised).
- Short enough that a full dub fits in 30 min wall-clock on dev hardware.

Selection happens at plan-execution time; the spec only requires "one new video, same pipeline + evaluation harness, results persisted to `docs/eval_results/`."

---

## 5. Data Flow (new)

```
dubbed video + SRT + ref WAV + ground truths
   │
   ├─► TranslationEval (hyp_srt, ref_srt)
   │       └─► BLEU-4 + chrF++ + worst-5 sentences
   │
   ├─► VoiceEval (ref_wav, audio_chunks/, aligned_audio/, merged_audio)
   │       └─► pre_align / post_align / merged similarity stats
   │
   ├─► SyncEval (merged_audio, srt_entries)
   │       └─► onset delay per entry + {mean, p50, p95, n_missed}
   │
   ├─► CaptionDriftEval (ocr_segments OR ocr_srt, en_truth_srt)
   │       └─► matched/missed/extra counts + start/end drift + jaccard
   │
   └─► ReportGenerator
          ├─► docs/eval_results/<stem>_report.json
          ├─► docs/eval_results/<stem>_report.md
          └─► docs/eval_results/evaluation_summary.csv  (append-row)
```

---

## 6. Configuration Additions

None new in `.env` / `config.py`. Evaluation is invoked offline and takes its inputs from the CLI; no runtime flags affect pipeline behavior.

`docs/eval_results/` is the canonical output directory (matches existing `ReportGenerator` default). Add the directory to `.gitignore` for `*.json` and `*.csv` but **commit** the `*_report.md` so the thesis can reference them.

---

## 7. Error Handling

| Failure | Behavior |
|---|---|
| Hypothesis SRT and reference SRT have different entry counts | Log WARNING, truncate to `min(N)`, continue. |
| Audio chunk directory empty | Skip VoiceEval; log INFO. |
| Onset detection fails on a window | Count toward `n_missed`; continue. |
| Caption-truth file missing | Skip CaptionDriftEval; log INFO. |
| `librosa.load` fails | Skip that segment; log WARNING. Hard-fail only if every segment fails. |
| All four metric families skipped | Exit 1 with message *"No evaluation produced. Provide at least one ground-truth or audio input."* |

---

## 8. Testing Strategy

### Unit (fast, no models)

- `tests/m5_evaluation/test_caption_drift.py` — synthetic pred/truth pairs covering: perfect match, GT-only entries, pred-only entries, jitter within `time_match_window`, text mismatch beyond ratio, monotonic ordering preservation.
- `tests/m5_evaluation/test_translation_eval.py` — feed two tiny SRTs (3 entries each), assert BLEU > 0 for identical inputs and `chrf_plus_plus == 100.0` for identical inputs.
- `tests/m5_evaluation/test_sync_accuracy_fix.py` — synthesize a 5 s WAV with a known onset, call `compute_delay(expected_start=1.0, ...)`, assert returned delay matches the synthetic onset.
- `tests/m5_evaluation/test_report_markdown.py` — feed a full report dict, assert generated Markdown contains all four metric-family section headers and the config snapshot block.

### Integration (cheap, real files)

- `tests/m5_evaluation/test_run_evaluation_cli.py` — invoke the CLI in-process on a fixture mini-pipeline output (already-dubbed 5 s clip checked into `tests/fixtures/m5/`), assert exit 0 and that the report JSON contains expected keys.

### Manual / smoke

- After implementation, run the CLI against the existing `Demo-Module-5` outputs (no rerun of pipeline). Confirm a single JSON + Markdown report lands under `docs/eval_results/`.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Hand-curated VI reference is noisy / one-person-judgment | Spec only requires **one** reference per video; thesis defense can frame this as a baseline single-rater eval. Multi-rater corpus eval is explicitly Phase 3. |
| `librosa.onset.onset_detect` over-fires on Vietnamese voice with breaks | Switch detection windows: use `[srt_start − 0.5s, srt_start + 0.5s]` rather than the full ±1 s default if `n_missed > 30%`. Tunable via CLI. |
| Resemblyzer at 24 kHz vs 16 kHz mismatch on VieNeu output | `SpeakerEncoder.extract_embedding` already resamples; verify in the test_run_evaluation_cli fixture. |
| Second test video introduces an OCR-band-layout that breaks the fixed `CAPTION_BAND_RATIO` | Spec acknowledges this — if it triggers, raise a follow-up under "auto-detect caption band" (deferred per the OCR spec). Don't block Phase 2 closeout on it. |
| MOS skeleton remains as `NotImplementedError` — questions at thesis defense | Document explicitly in `Phase2_Progress_Report.md` and `Phase3_Plan.md` that MOS is out-of-scope for Phase 2 and that NISQA/DNSMOS is tracked for Phase 3. |
| `evaluation_summary.csv` row-append on every run accumulates stale entries | The CSV is regenerated from `*_report.json` files on each run (existing `load_reports` path); single source of truth is the JSON files. |

---

## 10. Out of Scope (deferred to Phase 3)

- Real MOS estimation (UTMOS / NISQA / DNSMOS).
- Auto-detect caption band by clustering OCR bbox Y-positions.
- Web UI surfacing for evaluation reports.
- Inter-rater agreement / multi-reference BLEU.
- Sync evaluation against the source video's audio (lip-sync proxy).
- Per-voice automatic `VI_CHARS_PER_SEC` calibration.

---

## 11. Files Changed (summary)

### New

- `src/m5_evaluation/caption_drift.py` — `compare_caption_tracks` + `DriftStats`.
- `src/m5_evaluation/translation_eval.py` — SRT-aware BLEU/chrF wrapper.
- `src/m5_evaluation/voice_eval.py` — directory-aware speaker similarity wrapper.
- `scripts/run_evaluation.py` — CLI orchestrator.
- `data/references/Demo-Module-5_en_truth.srt` — hand-curated EN reference (≤ 60 entries).
- `data/references/Demo-Module-5_vi_truth.srt` — hand-curated VI reference (same N).
- `tests/m5_evaluation/__init__.py` (if missing).
- `tests/m5_evaluation/test_caption_drift.py`.
- `tests/m5_evaluation/test_translation_eval.py`.
- `tests/m5_evaluation/test_sync_accuracy_fix.py`.
- `tests/m5_evaluation/test_report_markdown.py`.
- `tests/m5_evaluation/test_run_evaluation_cli.py`.
- `tests/fixtures/m5/` — tiny dubbed-output fixture for the CLI smoke test.
- `docs/eval_results/Demo-Module-5_report.md` (committed; JSON optional).
- `docs/eval_results/<second-video>_report.md` (committed; JSON optional).
- `docs/Phase3_Plan.md` — outline of what was punted (MOS, auto-band, web UI, multi-rater eval).

### Modified

- `src/m5_evaluation/sync_accuracy.py` — fix `evaluate_segments` bug; add `evaluate_against_merged_audio`.
- `src/m5_evaluation/report_generator.py` — `caption_drift_metrics` slot; `generate_markdown`; richer summary CSV columns.
- `src/m5_evaluation/__init__.py` — re-export new classes.
- `src/m5_evaluation/mos_estimator.py` — keep `NotImplementedError` but make the docstring + raise message point at `docs/Phase3_Plan.md`.
- `docs/Phase2_Progress_Report.md` — add §11 "Evaluation Results" with the actual numbers from both videos.
- `.gitignore` — `docs/eval_results/*.json` and `docs/eval_results/*.csv`.

---

## 12. Open Decisions Locked In

- **MOS:** explicitly deferred. Phase 2 closeout does **not** implement UTMOS/NISQA. ✅
- **Reference data:** hand-curated ≤ 60 entries per video, one rater (the user). Multi-rater is Phase 3. ✅
- **Sync eval:** primary signal is full-mix onset detection against SRT timestamps. Per-chunk variant kept for diagnostics only. ✅
- **Second video:** required for Phase 2 closeout but choice deferred to plan execution. ✅
- **Eval output location:** `docs/eval_results/`; Markdown checked in, JSON/CSV gitignored. ✅
