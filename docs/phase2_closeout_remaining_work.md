# Phase 2 Closeout — Remaining Work

**Last session:** 2026-05-25
**Plan:** `docs/superpowers/plans/2026-05-25-phase2-closeout-plan.md`

## What was completed (Stages 1–3)

### Stage 1 — SyncAccuracy bug fix + full-mix variant
- Created test fixture: `tests/fixtures/m5/synthetic_onset_5s.wav` (440 Hz burst at 1.0s)
- Fixed `evaluate_segments` to use `srt_start_sec` instead of hardcoded `0.0`
- Added `evaluate_against_merged_audio` method for evaluating a single merged audio against an SRT timeline
- Added `_timestamp_to_seconds` helper
- Tests: `tests/m5_evaluation/test_sync_accuracy_fix.py` (2 tests, green)

### Stage 2 — Three new M5 wrappers
- `src/m5_evaluation/caption_drift.py` — `compare_caption_tracks()`, `load_caption_entries_from_srt()`, `load_caption_entries_from_ocr_json()`
- `src/m5_evaluation/translation_eval.py` — `evaluate_translation_from_srts()`
- `src/m5_evaluation/voice_eval.py` — `evaluate_voice_clone()`
- Updated `src/m5_evaluation/__init__.py` with new exports
- Tests: `test_caption_drift.py` (6 tests), `test_translation_eval.py` (1 test), all green

### Stage 3 — ReportGenerator extension + Markdown emitter
- Extended `generate_report()` with: `caption_drift_metrics`, `pipeline_mode`, `pipeline_elapsed_sec`, `pipeline_config_snapshot`
- Added `generate_markdown(report)` method — outputs formatted Markdown with Summary table, Translation, Voice Clone, Sync, Caption Drift, Config Snapshot sections (skips empty sections)
- Updated `generate_summary_csv()` with new columns: `pipeline_mode`, `speaker_sim_merged`, `sync_delay_p95`, `caption_drift_p95`
- Tests: `test_report_markdown.py` (2 tests, green)

**All 11 tests pass:** `pytest tests/m5_evaluation/ -v`

---

## What remains to do

### Stage 4 — `scripts/run_evaluation.py` CLI (code-only, ~1.5h)

Create the CLI orchestrator script that ties all evaluation wrappers together. The plan has the full skeleton at Stage 4 (section 4.1). Key points:

- Accepts: `--video`, `--srt`, `--dubbed`, `--ref-audio`, `--ground-truth-srt`, `--ground-truth-captions`, `--ocr-segments-json`, `--merged-audio`, `--audio-chunks`, `--aligned-audio`, `--pipeline-mode`, `--output-name`
- Calls `evaluate_translation_from_srts`, `evaluate_voice_clone`, `SyncAccuracy.evaluate_against_merged_audio`, `compare_caption_tracks` based on which args are provided
- Generates JSON + Markdown report via `ReportGenerator`
- Optional: smoke test in `tests/m5_evaluation/test_run_evaluation_cli.py`

### Stage 5 — Hand-curate ground-truth SRTs + baseline run (manual, ~2h)

1. Open `Demo-Module-5.mp4` in VLC with frame-accurate timestamps
2. Transcribe burned-in English captions → `data/references/Demo-Module-5_en_truth.srt`
3. Hand-translate to Vietnamese → `data/references/Demo-Module-5_vi_truth.srt`
4. Run `scripts/run_evaluation.py` with all paths (see plan Stage 5.3 for full command)
5. Sanity-check: BLEU-4 ≥ 20, speaker sim ≥ 0.7, sync p95 < 1.0s, caption drift p95 < 0.5s

### Stage 6 — Second-video run + Phase 2 closeout (mixed, ~3h)

1. Pick a second video (≤ 5 min, burned-in English captions, different domain)
2. Run `scripts/run_pipeline.py --mode ocr` on it
3. Hand-curate references and run evaluation
4. Update `Phase2_Progress_Report.md` with §11 Evaluation Results
5. Create `docs/Phase3_Plan.md`

---

## Files modified/created (not yet committed)

### New files:
- `src/m5_evaluation/caption_drift.py`
- `src/m5_evaluation/translation_eval.py`
- `src/m5_evaluation/voice_eval.py`
- `tests/m5_evaluation/__init__.py`
- `tests/m5_evaluation/test_sync_accuracy_fix.py`
- `tests/m5_evaluation/test_caption_drift.py`
- `tests/m5_evaluation/test_translation_eval.py`
- `tests/m5_evaluation/test_report_markdown.py`
- `tests/fixtures/m5/synthetic_onset_5s.wav`

### Modified files:
- `src/m5_evaluation/__init__.py` — added new exports
- `src/m5_evaluation/sync_accuracy.py` — bug fix + new method
- `src/m5_evaluation/report_generator.py` — extended with Markdown + new kwargs
