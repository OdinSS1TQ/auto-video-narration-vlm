# SRT Timestamp Anchoring + Budgeted-Stretch Alignment

Date: 2026-05-19
Status: Approved
Scope: M1 (post-VLM re-timing) + M3 (audio alignment); applies to both the standalone
`scripts/run_vlm_extract.py` flow and the end-to-end `src/m4_pipeline/runner.py` flow.

## Problem

Two distinct timing defects compound in the current pipeline output:

1. **SRT timestamps don't match the video.** Qwen3.5-2B emits subtitle entries packed
   into consecutive 3-second slots starting at `00:00:01` regardless of chunk length.
   On the test video, five entries occupy `1s–16s` even though the chunk is much
   longer; subtitles do not correspond to the moments those actions happen on screen.
2. **TTS audio overflows the SRT slot.** Vietnamese narration of the same content takes
   ~30% longer than the English source. With 3s slots, every generated segment needs
   to be time-compressed to fit, producing rushed, unnatural speech.

The current M3 `AudioAligner` only handles defect #2 — and handles it by compressing
unconditionally. Fixing M3 alone leaves the speaker sounding natural but still talking
about content that has long since left the screen.

## Goals

- Subtitle start times land near actual visual transitions in the chunk.
- Subtitle durations match how long the Vietnamese narration actually takes to speak.
- TTS audio plays at near-natural speed (≤1.25x speed-up) in the final video.
- The fix applies uniformly to standalone (`run_vlm_extract.py`) and end-to-end
  (`runner.py`) flows by living in shared modules.

## Non-goals

- Improving the VLM's content (translation quality, hallucination). The Pass 0 global
  summary work already landed.
- Frame-perfect lip sync. The source video is screen-recording / demo content; we are
  aligning narration to *visual sections*, not to a speaking mouth.
- Changing VieNeu's voice cloning. The reference encoding path is untouched.
- Backwards compatibility with existing SRT files in `data/outputs/` — they will be
  regenerated.

## Design

### Layer 1 — M1 post-processing: SRT Entry Re-Timing

After the VLM returns N parsed entries for a chunk spanning `[chunk_start, chunk_end]`,
**discard the VLM-supplied `start_time` / `end_time`** and reconstruct them.

Algorithm (`EntryRetimer.retime`):

1. **Estimate speech duration** per entry from translated text length:

   ```
   expected_dur_i = max(MIN_DUR, len(translated_text_i) / VI_CHARS_PER_SEC)
   ```

   `VI_CHARS_PER_SEC = 15.0` (measured ≈ 16.8 char/s on VieNeu Turbo for natural
   Vietnamese; 15 leaves headroom). `MIN_DUR = 1.5s` enforces a readable floor.

2. **Normalize** so the durations sum to `(chunk_end - chunk_start) * FILL_RATIO`,
   where `FILL_RATIO = 0.95` reserves 5% for inter-entry gaps and chunk-edge padding.
   Each entry's duration is scaled by the same factor:

   ```
   scale = (chunk_end - chunk_start) * FILL_RATIO / sum(expected_dur_i)
   normalized_dur_i = expected_dur_i * scale
   ```

3. **Place sequentially** starting at `chunk_start`:

   ```
   start_i = chunk_start + sum(normalized_dur_j for j < i) + i * MIN_GAP
   end_i   = start_i + normalized_dur_i
   ```

   `MIN_GAP = 0.1s` (shared with M3).

4. **Snap starts to frame timestamps** the VLM saw. For each computed `start_i`,
   replace it with the nearest value from the chunk's `frame_timestamps` list,
   breaking ties toward the earlier timestamp. Snapping is applied to all starts
   based on the sequential placement from step 3; later starts do not re-shift
   in response to earlier ones snapping. After snapping, enforce monotonicity:
   if `snapped_start_i ≤ snapped_start_{i-1}`, set
   `snapped_start_i = snapped_start_{i-1} + MIN_GAP` (unsnapped fallback for that
   entry only, logged at DEBUG).

5. **Recompute end times** from snapped starts: for `i < N-1`,
   `end_i = start_{i+1} - MIN_GAP`. For the last entry, `end_{N-1} = chunk_end`.

The retimer mutates `start_time` / `end_time` only. `index`, `original_text`,
`translated_text` pass through untouched.

#### Module placement and interface

New file: `src/m1_vlm/entry_retimer.py`.

```python
class EntryRetimer:
    def __init__(
        self,
        chars_per_sec: float = 15.0,
        fill_ratio: float = 0.95,
        min_dur_sec: float = 1.5,
        min_gap_sec: float = 0.1,
    ): ...

    def retime(
        self,
        entries: list[dict],
        chunk_start: float,
        chunk_end: float,
        frame_timestamps: list[float],
    ) -> list[dict]:
        """Return new list of entries with rewritten start_time/end_time."""
```

Call sites:

- `src/m4_pipeline/runner.py` — after the JSON parse in the per-chunk loop, before
  `srt_builder.add_entries(entries, chunk_offset=0.0)`.
- `scripts/run_vlm_extract.py` — same insertion point, after `parsed = json.loads(...)`
  and before `cw.add_chunk_result` / `builder.add_entries`.

The retimer is pure (no I/O, no side effects) and is unit-testable in isolation.

### Layer 2 — M3 budgeted-stretch + slip cascade

Replace the current "compress to fit unconditionally" logic in `AudioAligner` with a
budgeted policy. Per segment, given `audio_dur` from the generated WAV and
`target_dur = end_sec - start_sec`:

| Condition | Strategy | Effect on this segment | Effect downstream |
|---|---|---|---|
| `audio ≤ target + TOL` | `exact` | leave as-is | none |
| `target < audio ≤ target * MAX_SPEEDUP` | `compress_fit` | rubberband stretch to `target` | none |
| `audio > target * MAX_SPEEDUP` | `compress_max + slip` | rubberband stretch to `audio / MAX_SPEEDUP` | shift next segment's effective start by overflow |
| `audio < target - TOL` (rare) | `pad` | pad with silence to `target` (existing path) | none |

`MAX_SPEEDUP = 1.25`. `TOL = 0.2s` (existing).

#### Slip cascade

After per-segment alignment, walk the list left-to-right and compute effective starts:

```
effective_start[0] = srt_start[0]
for i in 1..N-1:
    effective_start[i] = max(srt_start[i],
                             effective_start[i-1] + aligned_dur[i-1] + MIN_GAP)
```

If `effective_start[i] > srt_start[i]`, the segment has slipped. Slips only push
forward; they never pull a segment earlier.

The last segment in a chunk can run past its `srt_end`. If it would run past
`chunk_end`, truncate the audio to `chunk_end - effective_start[last]` (existing
`_truncate_audio` path).

#### Module changes

`src/m3_sync/time_stretcher.py` — add:

```python
def stretch_capped(
    self,
    audio_path: Path,
    target_duration: float,
    max_ratio: float,
    output_dir: Path,
) -> tuple[Path, float, str]:
    """
    Stretch toward target_duration but never compress more than max_ratio (audio/target).
    Returns (output_path, achieved_duration, strategy: 'exact'|'compress_fit'|'compress_max').
    """
```

`src/m3_sync/audio_aligner.py` — rewrite `calculate_deltas` and `align_all`:

- `calculate_deltas` adds the new strategy values above and computes
  `max_compressed_dur = audio_dur / MAX_SPEEDUP`.
- `align_all` calls `stretch_capped(...)` instead of `stretch_to_fit(...)`, then
  walks the list to assign `effective_start_sec` and `aligned_duration` and marks
  `slip_applied: bool`. Returns segments enriched with these fields.

`FFmpegRenderer.merge_audio_segments` already reads `segment["start_sec"]`. The
aligner will overwrite `start_sec` with `effective_start_sec` on the returned
segments so the renderer needs no change. (Original SRT start is preserved as
`srt_start_sec` for diagnostics.)

### Config additions

In `src/m4_pipeline/config.py` (and `.env.example`):

| Property | Env var | Default | Used by |
|---|---|---|---|
| `vi_chars_per_sec` | `M1_VI_CHARS_PER_SEC` | `15.0` | EntryRetimer |
| `chunk_fill_ratio` | `M1_CHUNK_FILL_RATIO` | `0.95` | EntryRetimer |
| `m3_max_speedup` | `M3_MAX_SPEEDUP` | `1.25` | AudioAligner |
| `m3_min_gap_sec` | `M3_MIN_GAP_SEC` | `0.1` | EntryRetimer + AudioAligner |

`AudioAligner.__init__` already takes constructor args; pass these from
`runner.py`. `EntryRetimer.__init__` accepts them likewise.

### Standalone validator

`scripts/test_m3_sync.py` already dumps `stage1_deltas.json` per segment. Extend
that JSON to include `effective_start_sec`, `aligned_duration`, `slip_applied`,
`strategy` so M3 behavior is inspectable without re-running the pipeline.

## Data flow (new vs old)

**Old:**
```
VLM → parsed entries (bad timestamps) → SRT → TTS (audio too long) →
AudioAligner (compress all) → rushed dubbed video
```

**New:**
```
VLM → parsed entries → EntryRetimer (timestamps re-anchored to chunk + frames) →
SRT → TTS → AudioAligner (budget-stretch + slip cascade) → natural-paced dubbed video
```

## Testing

### Unit tests

New file: `tests/m1_vlm/test_entry_retimer.py`.

- 5 entries of equal length → expect even distribution across chunk.
- 5 entries with one 3x longer translated_text → expect that entry to get
  proportionally longer slot.
- N=1 entry → spans `[chunk_start, chunk_end]`.
- Frame timestamps `[2.0, 5.0, 10.0]`, computed start `4.7` → snaps to `5.0`.
- Sum of normalized durations ≤ `(chunk_end - chunk_start) * fill_ratio`.

New file: `tests/m3_sync/test_audio_aligner_budget.py` (or extend existing).

- Segment with `audio = 1.1 * target` → `compress_fit`, no slip.
- Segment with `audio = 2.0 * target` → `compress_max` (ratio 1.25), slip by
  `target * 0.6`.
- Two segments, first slips by 0.4s and second has slack → second's
  `effective_start = srt_start_2` (slack absorbed).
- Two segments, first slips by 0.4s and second has no slack → second's
  `effective_start = effective_start_1 + aligned_dur_1 + min_gap`.

### Integration verification

After implementation, re-run the existing test video end-to-end:

```
.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
python scripts/run_pipeline.py `
  --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
  --ref-audio "data\reference_audio\speaker.wav" --vlm-mode local
```

Expected:
- SRT entries span the full chunk duration (not just first 16s).
- `AudioAligner` log shows mostly `exact` / `compress_fit`, few or zero
  `compress_max + slip`.
- Subjective listen: narration sounds natural-paced, lines up with visible
  sections of the demo.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `VI_CHARS_PER_SEC` is wrong for other speakers / styles | Config-driven; measure and tune per voice if needed. Default chosen from VieNeu Turbo measurement. |
| Snapping to frame timestamps clusters all subtitles at the few extracted frames | Frame extractor produces ≥3 timestamps per chunk via adaptive sampling; with FILL_RATIO=0.95 and MIN_GAP, snap targets stay distinct. If clustering occurs, fall back to unsnapped sequential placement (log a warning). |
| Long Vietnamese text in a short chunk forces FILL_RATIO > 1 | If `sum(expected_dur) > chunk_duration`, `scale < 1` already compresses; M3 budget+slip catches residual. If even compressed durations exceed `chunk_duration`, the last entry truncates at `chunk_end`. |
| Slipping past `chunk_end` truncates content | Acceptable per goals; logged. Future work: pre-flight check that flags chunks where TTS will inevitably overflow. |
| Existing dubbed outputs under `data/outputs/` are stale | Document re-run in commit message; no migration script. |

## Out of scope (deferred)

- Per-voice automatic measurement of `VI_CHARS_PER_SEC`.
- Adjusting VLM prompts to produce better timestamps. The retimer makes this moot
  for now; if a future VLM produces good timestamps, the retimer becomes a no-op
  if `respect_vlm_timestamps=True` (not implemented now).
- Lip-sync alignment for talking-head videos.

## Files changed (summary)

New:
- `src/m1_vlm/entry_retimer.py`
- `tests/m1_vlm/test_entry_retimer.py`
- `tests/m3_sync/test_audio_aligner_budget.py`

Modified:
- `src/m4_pipeline/runner.py` — invoke EntryRetimer; pass config to AudioAligner.
- `scripts/run_vlm_extract.py` — invoke EntryRetimer.
- `src/m3_sync/audio_aligner.py` — budget+slip logic; new returned fields.
- `src/m3_sync/time_stretcher.py` — add `stretch_capped`.
- `src/m4_pipeline/config.py` — four new properties.
- `.env.example` — four new env vars with comments.
- `scripts/test_m3_sync.py` — extend `stage1_deltas.json` with new fields.
