# Phase 2 Closeout — Implementation Plan

**Date:** 2026-05-25
**Spec:** `docs/superpowers/specs/2026-05-25-phase2-closeout-design.md`
**Branch:** `dev-21052026` (or a new `dev-25052026-eval` if isolation is desired)
**Owner:** Quân
**Status:** Ready for execution

---

## 0. Preconditions

1. Phase 2 dubbed outputs already exist under `data/outputs/Demo-Module-5/` (commit `02601cd`'s OCR-mode run).
2. `.venv` is activated for every Python invocation:
   ```powershell
   .venv\Scripts\Activate.ps1
   $env:PYTHONIOENCODING="utf-8"
   ```
3. `sacrebleu`, `librosa`, `resemblyzer` already in `requirements.txt` (verify with `pip list | findstr /R "sacrebleu librosa resemblyzer"`).
4. The spec is approved.

---

## 1. Stages

The work is six stages, ordered so the next stage can verify the previous one with real data.

| # | Stage | Code only? | Wall-clock estimate |
|---|---|---|---|
| 1 | Sync-accuracy bug fix + full-mix variant | Yes | 1 h |
| 2 | Three new M5 wrappers (translation, voice, caption-drift) | Yes | 2 h |
| 3 | `ReportGenerator` extension + Markdown emitter | Yes | 1 h |
| 4 | `scripts/run_evaluation.py` CLI | Yes | 1.5 h |
| 5 | Hand-curate ground-truth SRTs + run baseline on Demo-Module-5 | No (user task) | 2 h |
| 6 | Second-video pipeline run + evaluation + report updates | Mixed | 3 h |

Total: ~10.5 h focused work spread across 2–3 days.

---

## Stage 1 — Fix `SyncAccuracy` + add full-mix variant

**Goal:** Existing `evaluate_segments` actually uses SRT start time. New method evaluates a single merged audio against the SRT timeline.

### 1.1 Read current state

```powershell
.venv\Scripts\Activate.ps1
pytest tests -k "sync" --collect-only
```

Expectation: no existing tests target `SyncAccuracy`. Confirm.

### 1.2 Add fixture (in `tests/fixtures/m5/`)

Create `tests/fixtures/m5/synthetic_onset_5s.wav`:
- 5 s, 16 kHz mono.
- 1.0 s silence, then a 0.5 s burst of 440 Hz sine wave at 0.5 amplitude, then 3.5 s silence.

Generate the file via a one-off script (do NOT check the script in — only the WAV):

```python
import numpy as np, soundfile as sf
sr = 16000
t = np.arange(5*sr) / sr
y = np.zeros_like(t)
mask = (t >= 1.0) & (t < 1.5)
y[mask] = 0.5 * np.sin(2*np.pi*440*t[mask])
sf.write("tests/fixtures/m5/synthetic_onset_5s.wav", y.astype(np.float32), sr)
```

### 1.3 Write the test first (TDD)

`tests/m5_evaluation/__init__.py` — empty file if it doesn't exist.

`tests/m5_evaluation/test_sync_accuracy_fix.py`:

```python
from pathlib import Path
from src.m5_evaluation.sync_accuracy import SyncAccuracy

FIX = Path(__file__).parent.parent / "fixtures" / "m5" / "synthetic_onset_5s.wav"

def test_compute_delay_against_known_onset():
    sa = SyncAccuracy()
    onset = sa.detect_onset(FIX)
    assert 0.9 <= onset <= 1.15  # librosa precision

    delay_at_zero = sa.compute_delay(expected_start=0.0, audio_path=FIX)
    delay_at_one  = sa.compute_delay(expected_start=1.0, audio_path=FIX)
    assert delay_at_zero == sa.detect_onset(FIX)
    assert delay_at_one < delay_at_zero  # closer to expected → smaller delay

def test_evaluate_segments_uses_srt_start():
    sa = SyncAccuracy()
    segments = [
        {"srt_start_sec": 1.0, "aligned_audio_path": str(FIX)},
        {"srt_start_sec": 0.0, "aligned_audio_path": str(FIX)},
    ]
    result = sa.evaluate_segments(segments)
    # Both share one audio but use different expected starts;
    # they cannot produce identical delays unless the bug is back.
    assert result["count"] == 2
    assert result["mean_delay"] > 0  # at least one mismatch
```

Run: `pytest tests/m5_evaluation/test_sync_accuracy_fix.py -v`. Expect **failures** — that's the point.

### 1.4 Apply the fix

In `src/m5_evaluation/sync_accuracy.py`:

- In `evaluate_segments`, replace:
  ```python
  delay = self.compute_delay(0.0, audio_path)
  ```
  with:
  ```python
  expected = float(segment.get("srt_start_sec", segment.get("start_sec", 0.0)))
  delay = self.compute_delay(expected, audio_path)
  ```
- Add an `evaluate_against_merged_audio` method (see spec §4.1). Implementation:

```python
def evaluate_against_merged_audio(
    self,
    srt_entries: list[dict],
    merged_audio_path: str | Path,
    window_sec: float = 1.0,
) -> dict:
    """For each SRT entry, detect onset within [srt_start - window, srt_end + window]
    in the merged audio and report signed delay (positive = audio late)."""
    import librosa, numpy as np

    y, sr = librosa.load(str(merged_audio_path), sr=None)
    delays = []
    missed = 0
    for entry in srt_entries:
        start = self._timestamp_to_seconds(entry["start_time"])
        end = self._timestamp_to_seconds(entry["end_time"])
        win_start = max(0, start - window_sec)
        win_end = min(len(y)/sr, end + window_sec)
        seg = y[int(win_start*sr):int(win_end*sr)]
        onsets = librosa.onset.onset_detect(y=seg, sr=sr, units="time")
        if len(onsets) == 0:
            missed += 1
            continue
        actual = float(onsets[0] + win_start)
        delays.append(actual - start)  # signed
    if not delays:
        return {"mean_delay": 0.0, "p50": 0.0, "p95": 0.0, "n_missed": missed, "count": 0}
    arr = np.array(delays)
    return {
        "mean_delay": float(arr.mean()),
        "p50": float(np.percentile(np.abs(arr), 50)),
        "p95": float(np.percentile(np.abs(arr), 95)),
        "n_missed": missed,
        "count": len(delays),
    }
```

Also add a static `_timestamp_to_seconds` helper (mirror `AudioAligner._timestamp_to_seconds`).

### 1.5 Verify

```powershell
pytest tests/m5_evaluation/test_sync_accuracy_fix.py -v
```

All green. Commit:

```
fix(m5): SyncAccuracy.evaluate_segments uses SRT start; add full-mix variant
```

**Checkpoint:** Do not proceed to Stage 2 unless Stage 1 tests pass.

---

## Stage 2 — Three new M5 wrappers

### 2.1 `src/m5_evaluation/caption_drift.py`

```python
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Iterable, List
import numpy as np

@dataclass
class CaptionEntry:
    start_sec: float
    end_sec: float
    text: str

@dataclass
class DriftStats:
    n_matched: int
    n_missed_in_pred: int
    n_extra_in_pred: int
    start_drift_mean: float
    start_drift_p50: float
    start_drift_p95: float
    end_drift_mean: float
    duration_jaccard_mean: float

def compare_caption_tracks(
    pred: List[CaptionEntry],
    truth: List[CaptionEntry],
    text_match_ratio: float = 0.7,
    time_match_window_sec: float = 1.5,
) -> DriftStats:
    used = set()
    starts, ends, jaccards = [], [], []
    n_missed = 0
    for gt in truth:
        match_idx = None
        for j, p in enumerate(pred):
            if j in used: continue
            if abs(p.start_sec - gt.start_sec) > time_match_window_sec: continue
            ratio = SequenceMatcher(None, p.text.lower(), gt.text.lower()).ratio()
            if ratio >= text_match_ratio:
                match_idx = j
                break
        if match_idx is None:
            n_missed += 1
            continue
        used.add(match_idx)
        p = pred[match_idx]
        starts.append(abs(p.start_sec - gt.start_sec))
        ends.append(abs(p.end_sec - gt.end_sec))
        inter = max(0.0, min(p.end_sec, gt.end_sec) - max(p.start_sec, gt.start_sec))
        union = max(p.end_sec, gt.end_sec) - min(p.start_sec, gt.start_sec)
        jaccards.append(inter / union if union > 0 else 0.0)

    n_extra = len(pred) - len(used)
    if starts:
        sd = np.array(starts); ed = np.array(ends); jc = np.array(jaccards)
        return DriftStats(
            n_matched=len(starts),
            n_missed_in_pred=n_missed,
            n_extra_in_pred=n_extra,
            start_drift_mean=float(sd.mean()),
            start_drift_p50=float(np.percentile(sd, 50)),
            start_drift_p95=float(np.percentile(sd, 95)),
            end_drift_mean=float(ed.mean()),
            duration_jaccard_mean=float(jc.mean()),
        )
    return DriftStats(0, n_missed, n_extra, 0, 0, 0, 0, 0)

def load_caption_entries_from_srt(path) -> List[CaptionEntry]: ...   # use SRTBuilder.load_srt
def load_caption_entries_from_ocr_json(path) -> List[CaptionEntry]: ...  # work_ocr/segments_final.json
```

### 2.2 `src/m5_evaluation/translation_eval.py`

```python
from pathlib import Path
from src.m5_evaluation.bleu_scorer import BLEUScorer
from src.m1_vlm.srt_builder import SRTBuilder

def evaluate_translation_from_srts(hyp_srt: Path, ref_srt: Path) -> dict:
    hyp = SRTBuilder.load_srt(hyp_srt)
    ref = SRTBuilder.load_srt(ref_srt)
    if len(hyp) != len(ref):
        from loguru import logger
        logger.warning(f"Length mismatch: hyp={len(hyp)} ref={len(ref)}; truncating")
    n = min(len(hyp), len(ref))
    hyp_texts = [h["translated_text"] for h in hyp[:n]]
    ref_texts = [[r["translated_text"] for r in ref[:n]]]  # one ref per hyp
    scorer = BLEUScorer()
    metrics = scorer.evaluate(hyp_texts, ref_texts)
    # Sentence-level chrF++ for worst-5 sentences.
    import sacrebleu
    sent_chrfs = [
        sacrebleu.sentence_chrf(h, [r], word_order=2).score
        for h, r in zip(hyp_texts, ref_texts[0])
    ]
    worst = sorted(range(n), key=lambda i: sent_chrfs[i])[:5]
    metrics["worst5"] = [
        {"index": i+1, "chrf": sent_chrfs[i], "hyp": hyp_texts[i], "ref": ref_texts[0][i]}
        for i in worst
    ]
    metrics["n_entries"] = n
    return metrics
```

### 2.3 `src/m5_evaluation/voice_eval.py`

```python
from pathlib import Path
import numpy as np
from src.m5_evaluation.speaker_similarity import SpeakerSimilarity

def evaluate_voice_clone(
    reference_audio: Path,
    raw_chunks_dir: Path | None,
    aligned_chunks_dir: Path | None,
    merged_audio: Path | None,
) -> dict:
    sim = SpeakerSimilarity()
    result: dict = {}

    def _stats(dir_):
        if dir_ is None or not Path(dir_).exists():
            return None
        out = sim.evaluate_batch(reference_audio, dir_)
        if out.get("count", 0) == 0:
            return None
        return {
            "mean": out["mean_similarity"],
            "std": out["std_similarity"],
            "p05": float(np.percentile(
                _per_file_similarities(reference_audio, dir_, sim), 5
            )),
            "n_chunks": out["count"],
        }

    result["pre_align"]  = _stats(raw_chunks_dir)
    result["post_align"] = _stats(aligned_chunks_dir)
    if merged_audio and Path(merged_audio).exists():
        result["merged"] = {"similarity": sim.compute_similarity(reference_audio, merged_audio)}
    return result

def _per_file_similarities(ref, dir_, sim):
    return [
        sim.compute_similarity(ref, p) for p in sorted(Path(dir_).glob("*.wav"))
    ]
```

### 2.4 Update `src/m5_evaluation/__init__.py`

```python
from src.m5_evaluation.caption_drift import compare_caption_tracks, DriftStats, CaptionEntry
from src.m5_evaluation.translation_eval import evaluate_translation_from_srts
from src.m5_evaluation.voice_eval import evaluate_voice_clone
__all__ += ["compare_caption_tracks", "DriftStats", "CaptionEntry",
            "evaluate_translation_from_srts", "evaluate_voice_clone"]
```

### 2.5 Write unit tests

- `tests/m5_evaluation/test_caption_drift.py` — see spec §8 cases.
- `tests/m5_evaluation/test_translation_eval.py` — feed two identical 3-entry SRTs (use `SRTBuilder` to construct in-memory and `save()` to a tmp_path), assert `chrf_plus_plus == 100.0`.

Run: `pytest tests/m5_evaluation -v`. Expect green.

Commit:

```
feat(m5): caption-drift, translation-eval, voice-eval wrappers
```

---

## Stage 3 — Extend `ReportGenerator`

### 3.1 Edit `report_generator.py`

- Add kwargs to `generate_report`: `caption_drift_metrics`, `pipeline_mode`, `pipeline_elapsed_sec`, `pipeline_config_snapshot`.
- Persist all kwargs into the JSON.
- Add `generate_markdown(self, report)` returning a `Path`. Output layout:

```markdown
# Evaluation Report — <video_name>

**Generated:** 2026-05-25T14:32:11
**Pipeline mode:** ocr
**Pipeline duration:** 215.7 s

## Summary

| Metric | Value |
|---|---|
| BLEU-4 | 32.4 |
| chrF++ | 53.1 |
| Speaker similarity (merged) | 0.78 |
| Sync onset p95 | 0.46 s |
| Caption drift p95 | 0.31 s |

## Translation
...

## Voice clone
...

## Sync
...

## Caption drift (OCR-mode only)
...

## Config snapshot
```yaml
M3_MAX_SPEEDUP: 1.25
CAPTION_BAND_RATIO: 0.10
...
```
```

Skip sections whose metric is missing (don't emit empty headers).

### 3.2 Update `generate_summary_csv`

Add columns: `pipeline_mode`, `caption_drift_p95`, `speaker_sim_merged`, `sync_delay_p95`. Existing columns preserved for back-compat.

### 3.3 Test

`tests/m5_evaluation/test_report_markdown.py` — feed a fully-populated report dict, assert the Markdown contains every expected section header and the config snapshot block.

Commit:

```
feat(m5): ReportGenerator emits Markdown + caption-drift slot
```

---

## Stage 4 — `scripts/run_evaluation.py`

### 4.1 Skeleton

```python
"""CLI: Evaluate a dubbed video against ground truth + reference audio."""
import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m5_evaluation import (
    evaluate_translation_from_srts, evaluate_voice_clone,
    compare_caption_tracks, ReportGenerator,
)
from src.m5_evaluation.sync_accuracy import SyncAccuracy
from src.m5_evaluation.caption_drift import (
    load_caption_entries_from_srt, load_caption_entries_from_ocr_json,
)
from src.m1_vlm.srt_builder import SRTBuilder
from src.m4_pipeline.config import PipelineConfig

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--srt", required=True)
    p.add_argument("--dubbed", required=True)
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--audio-chunks")
    p.add_argument("--aligned-audio")
    p.add_argument("--merged-audio")
    p.add_argument("--ground-truth-srt")
    p.add_argument("--ground-truth-captions")
    p.add_argument("--ocr-segments-json")
    p.add_argument("--pipeline-mode", default="unknown", choices=["vlm","ocr","unknown"])
    p.add_argument("--output-name", required=True)
    args = p.parse_args()

    cfg = PipelineConfig()
    config_snap = {
        "M3_MAX_SPEEDUP": cfg.m3_max_speedup,
        "M3_MIN_GAP_SEC": cfg.m3_min_gap_sec,
        "M1_VI_CHARS_PER_SEC": cfg.vi_chars_per_sec,
        "CAPTION_BAND_RATIO": cfg.caption_band_ratio,
        "OCR_SAMPLE_FPS": cfg.ocr_sample_fps,
        # ... add all relevant Phase 2 knobs
    }

    translation = sync = voice = drift = None
    ran_any = False

    if args.ground_truth_srt:
        translation = evaluate_translation_from_srts(
            Path(args.srt), Path(args.ground_truth_srt))
        ran_any = True

    if args.audio_chunks or args.aligned_audio or args.merged_audio:
        voice = evaluate_voice_clone(
            Path(args.ref_audio),
            Path(args.audio_chunks) if args.audio_chunks else None,
            Path(args.aligned_audio) if args.aligned_audio else None,
            Path(args.merged_audio) if args.merged_audio else None,
        )
        ran_any = True

    if args.merged_audio:
        sa = SyncAccuracy()
        srt_entries = SRTBuilder.load_srt(Path(args.srt))
        sync = sa.evaluate_against_merged_audio(srt_entries, Path(args.merged_audio))
        ran_any = True

    if args.ground_truth_captions:
        truth = load_caption_entries_from_srt(Path(args.ground_truth_captions))
        if args.ocr_segments_json:
            pred = load_caption_entries_from_ocr_json(Path(args.ocr_segments_json))
        else:
            pred = load_caption_entries_from_srt(Path(args.srt))
        drift = compare_caption_tracks(pred, truth).__dict__
        ran_any = True

    if not ran_any:
        print("ERROR: No evaluation produced. Provide at least one ground-truth or audio input.", file=sys.stderr)
        sys.exit(1)

    rep = ReportGenerator()
    report = rep.generate_report(
        video_name=args.output_name,
        translation_metrics=translation,
        similarity_metrics=voice,
        sync_metrics=sync,
        caption_drift_metrics=drift,
        pipeline_mode=args.pipeline_mode,
        pipeline_config_snapshot=config_snap,
    )
    md_path = rep.generate_markdown(report)
    rep.generate_summary_csv(rep.load_reports())
    print(f"\n✅ JSON: {Path(rep.output_dir) / (args.output_name + '_report.json')}")
    print(f"✅ Markdown: {md_path}")

if __name__ == "__main__":
    main()
```

### 4.2 Add smoke test

`tests/m5_evaluation/test_run_evaluation_cli.py` — invoke `main()` in-process with a fixture mini-pipeline output (fixture under `tests/fixtures/m5/mini_run/`). Patch sys.argv. Assert exit 0 (no SystemExit raised) and that `docs/eval_results/test_smoke_report.json` exists.

Build the fixture as a tiny synthetic dub (3 entries, 5 s WAVs) — script lives in `tests/fixtures/m5/make_fixture.py` (one-off, not run in CI).

Commit:

```
feat(scripts): run_evaluation.py CLI orchestrator + smoke test
```

---

## Stage 5 — Hand-curate references + baseline run

### 5.1 Transcribe burned-in captions

Open `H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4` in a player that shows frame-accurate timestamps (VLC: `View → Advanced Controls` + `Tools → Track Synchronization`). Transcribe each visible English caption with its `(start_sec, end_sec)` into:

`data/references/Demo-Module-5_en_truth.srt` — 30–60 entries, SRT format, manually verified.

### 5.2 Translate to Vietnamese

Hand-translate every entry in `_en_truth.srt` keeping the **same index and timestamps**:

`data/references/Demo-Module-5_vi_truth.srt`.

Quality bar: natural conversational Vietnamese suitable for narration. Don't over-localize technical terms; keep `pip install`, code, and UI labels in English (per the project's translation rules).

### 5.3 Run the baseline evaluation

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
python scripts/run_evaluation.py `
  --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
  --srt "data\outputs\_VNEXT__ToyoBeauty_Demo-Module-5\_VNEXT__ToyoBeauty_Demo-Module-5_vi.srt" `
  --dubbed "data\outputs\_VNEXT__ToyoBeauty_Demo-Module-5\_VNEXT__ToyoBeauty_Demo-Module-5_dubbed.mp4" `
  --ref-audio "data\reference_audio\speaker.wav" `
  --audio-chunks "data\outputs\_VNEXT__ToyoBeauty_Demo-Module-5\work_ocr\audio_chunks" `
  --aligned-audio "data\outputs\_VNEXT__ToyoBeauty_Demo-Module-5\work_ocr\aligned_audio" `
  --merged-audio "data\outputs\_VNEXT__ToyoBeauty_Demo-Module-5\work_ocr\merged_audio.wav" `
  --ground-truth-srt "data\references\Demo-Module-5_vi_truth.srt" `
  --ground-truth-captions "data\references\Demo-Module-5_en_truth.srt" `
  --ocr-segments-json "data\outputs\_VNEXT__ToyoBeauty_Demo-Module-5\work_ocr\segments_final.json" `
  --pipeline-mode ocr `
  --output-name Demo-Module-5
```

Sanity-check the output:
- BLEU-4 ≥ 20 (any thesis-grade NMT clears this on short tutorial videos with a single hand reference).
- Speaker similarity (merged) ≥ 0.7 (VieNeu Turbo + 10 s reference).
- Sync onset p95 < 1.0 s.
- Caption drift p95 < 0.5 s (OCR mode against burned-in truth).

If any number is dramatically off (e.g., BLEU < 5, similarity < 0.4), debug **before** committing the report.

### 5.4 Commit

```
docs(m5): baseline evaluation results for Demo-Module-5

- data/references/Demo-Module-5_{en,vi}_truth.srt (hand-curated)
- docs/eval_results/Demo-Module-5_report.md
```

(`_report.json` and `evaluation_summary.csv` are gitignored.)

---

## Stage 6 — Second-video run + Phase 2 closeout

### 6.1 Pick the second video

Criteria (spec §4.8):
- ≤ 5 minutes, different domain from Demo-Module-5, burned-in English captions in bottom band.
- Suggested candidates: another `VNext` demo module of a different product, or a public software tutorial recording the user has rights to use.

Save it to `data/raw/<second-video>.mp4`.

### 6.2 Run the pipeline

```powershell
python scripts/run_pipeline.py `
  --video "data\raw\<second-video>.mp4" `
  --ref-audio "data\reference_audio\speaker.wav" `
  --mode ocr `
  --vlm-mode local
```

Watch for:
- `CaptionTimeline` produces a reasonable number of segments (expect 1–3 per 10 s of video).
- Narration classifier doesn't drop > 50% of segments (if it does, the captions are mostly screen text and the test video is a poor choice).
- AudioAligner log shows mostly `exact` / `compress_fit`, few `compress_max + slip`.

### 6.3 Hand-curate references and evaluate

Same procedure as Stage 5 but for the new video. Output goes to `docs/eval_results/<second-video>_report.md`.

### 6.4 Update `Phase2_Progress_Report.md`

Add a new section `§11 Evaluation Results`:

```markdown
## 11. Evaluation Results

| Video | Mode | BLEU-4 | chrF++ | Speaker sim (merged) | Sync p95 (s) | Caption drift p95 (s) |
|---|---|---|---|---|---|---|
| Demo-Module-5 (60 s) | ocr | XX.X | XX.X | 0.XX | 0.XX | 0.XX |
| <second-video> (NN s) | ocr | XX.X | XX.X | 0.XX | 0.XX | — |

Full reports: docs/eval_results/Demo-Module-5_report.md,
              docs/eval_results/<second-video>_report.md
```

Numbers are filled in from the actual `_report.md` files (don't hand-edit; copy from the generated tables).

Update §7 "Known Issues After Phase 2" to remove the rows that are now resolved:
- "OCR-mode quality unmeasured" → resolved.
- "One test video" → resolved.
- "M5 evaluation absent" → resolved (excluding MOS).

Add a row:
- "MOS estimation still skeleton" → severity Medium; deferred to Phase 3 per `docs/Phase3_Plan.md`.

### 6.5 Create `docs/Phase3_Plan.md`

Single page, outlining what was punted (use the spec §10 list as a starting point). Frame it as the thesis "future work" section.

### 6.6 Commit

```
docs: Phase 2 closeout — evaluation results + Phase 3 plan

- docs/eval_results/<second-video>_report.md
- docs/Phase2_Progress_Report.md §11 Evaluation Results
- docs/Phase3_Plan.md (MOS, auto-band, web UI, multi-rater)
```

---

## 2. Verification at every commit boundary

Before each commit:

```powershell
.venv\Scripts\Activate.ps1
pytest tests/m5_evaluation -v
ruff check src/m5_evaluation scripts/run_evaluation.py
```

Both must be green.

After the Demo-Module-5 baseline (end of Stage 5):

```powershell
python scripts/run_evaluation.py --help
# sanity-check the JSON
type docs\eval_results\Demo-Module-5_report.json | python -m json.tool | findstr "bleu4 mean_similarity p95"
```

After Phase 2 closeout (end of Stage 6):

```powershell
# regenerate summary CSV from all reports
python -c "from src.m5_evaluation.report_generator import ReportGenerator; r=ReportGenerator(); r.generate_summary_csv(r.load_reports())"
type docs\eval_results\evaluation_summary.csv
```

Confirm both video rows are present.

---

## 3. Rollback strategy

Every stage produces an independent commit. If Stage 5's baseline numbers are catastrophic, roll back at Stage 4's commit and debug the metrics code before re-running. Do not commit ground-truth SRTs separately from the report that uses them — they should land together so reviewers can re-run.

---

## 4. Risks (execution-time)

| Risk | Trigger | Response |
|---|---|---|
| Onset detection over-fires on Vietnamese sibilants | Sync p95 > 1.0 s on Demo-Module-5 | Tighten window in `evaluate_against_merged_audio` from 1.0 s to 0.5 s; rerun. |
| Hand-curated truth has typos | BLEU dramatically low | Eyeball worst-5 sentences in the report; fix truth SRT; rerun. Do not "fix" by relaxing metrics. |
| Second video uses a different caption-band ratio | `n_segments_raw == 0` | Override `CAPTION_BAND_RATIO` via `.env` for that run; document in the second-video report. |
| GLM-OCR hallucination loops on the second video | `n_segments_after_classify` < 3 → `PipelineError` | Check `work_ocr/segments_raw.json` for the artifact strings; tune `is_ocr_repetition_artifact` parameters if needed (don't broaden them blindly — keep the existing min_length / window_chars semantics). |
| Resemblyzer embedding model download fails offline | `SpeakerSimilarity` init crashes | Pre-download via `python scripts/download_models.py` before running the eval. |

---

## 5. Done criteria

Phase 2 is **closed** when:

- [ ] Stage 1–4 commits landed; `pytest tests/m5_evaluation -v` green.
- [ ] `docs/eval_results/Demo-Module-5_report.md` committed with non-degenerate numbers.
- [ ] `docs/eval_results/<second-video>_report.md` committed with non-degenerate numbers.
- [ ] `docs/Phase2_Progress_Report.md` §11 added; §7 updated.
- [ ] `docs/Phase3_Plan.md` created.
- [ ] `data/references/*_truth.srt` committed for both videos.

When all checkboxes are ticked, declare Phase 2 done in the next commit message.
