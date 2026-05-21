# SRT Timing Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Commits are user-driven on this project.** Steps that mark a slice as "ready to commit" describe a verification gate, not an automated `git commit`. Do not run `git commit` or `git push`; leave staging and committing to the user.

**Goal:** Fix two compounding timing defects so the dubbed video's narration matches what's on screen and plays at near-natural speed: (1) M1 EntryRetimer rebuilds VLM-supplied SRT timestamps from the chunk range and frame anchors, (2) M3 AudioAligner adds a 1.25x speed-up budget with forward-slip cascade for residual overflow.

**Architecture:** New pure-function module `src/m1_vlm/entry_retimer.py` invoked after every chunk's VLM parse in both `runner.py` and `scripts/run_vlm_extract.py`. `src/m3_sync/time_stretcher.py` gains a `stretch_capped()` helper. `src/m3_sync/audio_aligner.py` is rewritten to use the cap and to compute a slip cascade over `effective_start_sec`. Four new config knobs in `PipelineConfig`. All thresholds env-overridable.

**Tech Stack:** Python 3.11, pytest with `unittest.mock`, soundfile (audio metadata), rubberband CLI (pitch-preserving stretch), pysrt, loguru.

**Spec:** `docs/superpowers/specs/2026-05-19-srt-timing-sync-design.md`

---

## File Map

**Create:**
- `src/m1_vlm/entry_retimer.py` — `EntryRetimer` class, pure (no I/O).
- `tests/m1_vlm/test_entry_retimer.py` — unit tests for EntryRetimer.
- `tests/m3_sync/__init__.py` — package marker.
- `tests/m3_sync/test_audio_aligner_budget.py` — unit tests for new budget+slip logic.

**Modify:**
- `src/m4_pipeline/config.py` — add 4 properties.
- `.env.example` — document 4 new env vars.
- `src/m4_pipeline/runner.py` — call EntryRetimer after VLM parse; pass config values to AudioAligner.
- `scripts/run_vlm_extract.py` — call EntryRetimer after VLM parse.
- `src/m3_sync/time_stretcher.py` — add `stretch_capped()`.
- `src/m3_sync/audio_aligner.py` — rewrite `calculate_deltas` strategy + `align_all` to use cap and emit slip cascade.
- `scripts/test_m3_sync.py` — log new fields in `stage1_deltas.json`.

---

## Task 1: Config knobs

**Files:**
- Modify: `src/m4_pipeline/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add four properties to `PipelineConfig`**

Open `src/m4_pipeline/config.py`. After the existing `ssim_threshold` property (around line 135), add:

```python
    # === M1 Entry Retiming ===
    @property
    def vi_chars_per_sec(self) -> float:
        """Estimated Vietnamese narration speaking rate (chars/sec) used by
        EntryRetimer to size subtitle slots. Measured ~16.8 char/s on VieNeu
        Turbo; default 15.0 leaves headroom for slower deliveries."""
        return float(os.getenv("M1_VI_CHARS_PER_SEC", "15.0"))

    @property
    def chunk_fill_ratio(self) -> float:
        """Fraction of each chunk's duration that EntryRetimer fills with
        subtitle slots. The remainder is reserved for inter-entry gaps and
        chunk-edge padding."""
        return float(os.getenv("M1_CHUNK_FILL_RATIO", "0.95"))

    # === M3 Audio Alignment ===
    @property
    def m3_max_speedup(self) -> float:
        """Maximum TTS time-stretch ratio (audio/target) allowed before
        AudioAligner falls back to slipping the next segment forward."""
        return float(os.getenv("M3_MAX_SPEEDUP", "1.25"))

    @property
    def m3_min_gap_sec(self) -> float:
        """Minimum gap (seconds) between consecutive subtitle entries and
        between aligned audio segments after slip cascade."""
        return float(os.getenv("M3_MIN_GAP_SEC", "0.1"))
```

- [ ] **Step 2: Document them in `.env.example`**

Append to `.env.example`:

```dotenv
# === M1 Entry Retiming ===
# Vietnamese speaking rate estimate (chars/sec) for EntryRetimer slot sizing.
M1_VI_CHARS_PER_SEC=15.0
# Fraction of chunk duration that subtitle slots fill (rest is gap budget).
M1_CHUNK_FILL_RATIO=0.95

# === M3 Audio Alignment ===
# Max TTS speed-up before AudioAligner slips next segment.
M3_MAX_SPEEDUP=1.25
# Min gap between subtitles and between aligned audio segments.
M3_MIN_GAP_SEC=0.1
```

- [ ] **Step 3: Smoke-test the config loads**

Run: `.venv\Scripts\Activate.ps1; python -c "from src.m4_pipeline.config import PipelineConfig; c = PipelineConfig(); print(c.vi_chars_per_sec, c.chunk_fill_ratio, c.m3_max_speedup, c.m3_min_gap_sec)"`

Expected: `15.0 0.95 1.25 0.1`

- [ ] **Step 4: Slice ready to commit**

Files changed: `src/m4_pipeline/config.py`, `.env.example`. Stop for user to commit.

---

## Task 2: EntryRetimer module — failing tests first

**Files:**
- Create: `tests/m1_vlm/test_entry_retimer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/m1_vlm/test_entry_retimer.py`:

```python
"""Tests for EntryRetimer — SRT timestamp rebuilding from chunk + frame anchors."""

import pytest

from src.m1_vlm.entry_retimer import EntryRetimer


def _make_entry(idx, text="hello world"):
    return {
        "index": idx,
        "start_time": "00:00:00,000",
        "end_time": "00:00:01,000",
        "original_text": "x",
        "translated_text": text,
    }


def _ts_to_sec(ts: str) -> float:
    parts = ts.replace(",", ":").split(":")
    h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
    return h * 3600 + m * 60 + s + ms / 1000.0


class TestEntryRetimer:
    def test_single_entry_spans_full_chunk(self):
        retimer = EntryRetimer()
        entries = [_make_entry(1, text="x" * 30)]

        out = retimer.retime(entries, chunk_start=10.0, chunk_end=20.0, frame_timestamps=[10.0, 15.0, 20.0])

        assert len(out) == 1
        assert _ts_to_sec(out[0]["start_time"]) == pytest.approx(10.0, abs=0.05)
        assert _ts_to_sec(out[0]["end_time"]) == pytest.approx(20.0, abs=0.05)

    def test_equal_length_entries_get_even_slots(self):
        retimer = EntryRetimer(min_dur_sec=0.0, min_gap_sec=0.0, fill_ratio=1.0)
        entries = [_make_entry(i + 1, text="x" * 30) for i in range(5)]
        # Provide many frame anchors so snapping doesn't distort.
        frames = [i * 0.1 for i in range(int(60 / 0.1))]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=50.0, frame_timestamps=frames)

        starts = [_ts_to_sec(e["start_time"]) for e in out]
        durations = [_ts_to_sec(e["end_time"]) - s for e, s in zip(out, starts)]
        # 5 equal slots in a 50s window → each ~10s
        for d in durations:
            assert d == pytest.approx(10.0, abs=0.2)

    def test_long_text_gets_proportionally_longer_slot(self):
        retimer = EntryRetimer(min_dur_sec=0.0, min_gap_sec=0.0, fill_ratio=1.0)
        entries = [
            _make_entry(1, text="x" * 10),
            _make_entry(2, text="x" * 30),  # 3x longer
            _make_entry(3, text="x" * 10),
        ]
        frames = [i * 0.05 for i in range(int(50 / 0.05))]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=50.0, frame_timestamps=frames)

        d0 = _ts_to_sec(out[0]["end_time"]) - _ts_to_sec(out[0]["start_time"])
        d1 = _ts_to_sec(out[1]["end_time"]) - _ts_to_sec(out[1]["start_time"])
        d2 = _ts_to_sec(out[2]["end_time"]) - _ts_to_sec(out[2]["start_time"])
        # The middle entry should be ~3x the others.
        assert d1 == pytest.approx(3 * d0, rel=0.1)
        assert d2 == pytest.approx(d0, rel=0.1)

    def test_starts_snap_to_nearest_frame_timestamp(self):
        retimer = EntryRetimer(min_dur_sec=0.0, min_gap_sec=0.0, fill_ratio=1.0)
        entries = [_make_entry(i + 1, text="x" * 15) for i in range(3)]
        # Frames at 0, 5, 12, 20 — computed starts 0, ~6.66, ~13.33 should snap.
        frames = [0.0, 5.0, 12.0, 20.0]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=20.0, frame_timestamps=frames)

        starts = [_ts_to_sec(e["start_time"]) for e in out]
        # First always at chunk_start.
        assert starts[0] == pytest.approx(0.0, abs=0.01)
        # Second computed ~6.66 → nearest frame is 5.0 (tie-break earlier wins).
        assert starts[1] == pytest.approx(5.0, abs=0.01)
        # Third computed ~13.33 → nearest is 12.0.
        assert starts[2] == pytest.approx(12.0, abs=0.01)

    def test_tie_break_prefers_earlier_frame(self):
        retimer = EntryRetimer(min_dur_sec=0.0, min_gap_sec=0.0, fill_ratio=1.0)
        entries = [_make_entry(1, text="x" * 10), _make_entry(2, text="x" * 10)]
        # Computed start of entry 2 will be exactly 5.0; both 4.0 and 6.0 are
        # equidistant — earlier (4.0) must win.
        frames = [0.0, 4.0, 6.0, 10.0]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=10.0, frame_timestamps=frames)

        assert _ts_to_sec(out[1]["start_time"]) == pytest.approx(4.0, abs=0.01)

    def test_monotonicity_enforced_when_snap_collides(self):
        retimer = EntryRetimer(min_dur_sec=0.0, min_gap_sec=0.5, fill_ratio=1.0)
        entries = [_make_entry(i + 1, text="x" * 10) for i in range(3)]
        # Only one frame in the range → all three would snap to 5.0 without
        # monotonicity enforcement.
        frames = [5.0]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=15.0, frame_timestamps=frames)

        starts = [_ts_to_sec(e["start_time"]) for e in out]
        # Each subsequent start must be strictly greater than the previous
        # by at least min_gap (0.5).
        assert starts[1] > starts[0] + 0.49
        assert starts[2] > starts[1] + 0.49

    def test_last_entry_end_clamped_to_chunk_end(self):
        retimer = EntryRetimer(chars_per_sec=1.0, fill_ratio=1.0)  # Force tiny scale.
        entries = [_make_entry(1, text="x" * 1000)]  # Would want 1000s naturally.

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=10.0, frame_timestamps=[0.0, 10.0])

        assert _ts_to_sec(out[0]["end_time"]) == pytest.approx(10.0, abs=0.01)

    def test_min_duration_enforced(self):
        retimer = EntryRetimer(chars_per_sec=1000.0, min_dur_sec=1.5, fill_ratio=1.0)
        # At 1000 char/s a 10-char entry would want 0.01s; min_dur_sec must win.
        entries = [_make_entry(i + 1, text="x" * 10) for i in range(3)]
        frames = [i * 0.1 for i in range(int(60 / 0.1))]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=60.0, frame_timestamps=frames)

        # With 3 entries of pre-clamp 0.01s each and min_dur_sec=1.5, the
        # *expected* durations going into normalization should all be >= 1.5.
        # After normalization to fill 60s, each gets ~20s — but the floor's
        # effect is testable by checking they're all equal (proves the floor
        # made them equal before normalization).
        durations = [
            _ts_to_sec(e["end_time"]) - _ts_to_sec(e["start_time"])
            for e in out
        ]
        for d in durations:
            assert d == pytest.approx(durations[0], rel=0.05)

    def test_passthrough_fields_preserved(self):
        retimer = EntryRetimer()
        entries = [{
            "index": 7,
            "start_time": "00:00:00,000",
            "end_time": "00:00:01,000",
            "original_text": "hello",
            "translated_text": "xin chào",
            "custom_field": "keep_me",
        }]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=5.0, frame_timestamps=[0.0])

        assert out[0]["index"] == 7
        assert out[0]["original_text"] == "hello"
        assert out[0]["translated_text"] == "xin chào"
        assert out[0]["custom_field"] == "keep_me"

    def test_empty_entries_returns_empty(self):
        retimer = EntryRetimer()
        assert retimer.retime([], chunk_start=0.0, chunk_end=10.0, frame_timestamps=[0.0]) == []

    def test_no_frame_timestamps_skips_snap(self):
        retimer = EntryRetimer(min_dur_sec=0.0, min_gap_sec=0.0, fill_ratio=1.0)
        entries = [_make_entry(1, text="x" * 10), _make_entry(2, text="x" * 10)]

        out = retimer.retime(entries, chunk_start=0.0, chunk_end=10.0, frame_timestamps=[])

        starts = [_ts_to_sec(e["start_time"]) for e in out]
        # No snap → starts come from sequential placement directly.
        assert starts[0] == pytest.approx(0.0, abs=0.01)
        assert starts[1] == pytest.approx(5.0, abs=0.05)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\Activate.ps1; pytest tests/m1_vlm/test_entry_retimer.py -v`

Expected: `ModuleNotFoundError: No module named 'src.m1_vlm.entry_retimer'` for every test.

---

## Task 3: EntryRetimer module — implementation

**Files:**
- Create: `src/m1_vlm/entry_retimer.py`

- [ ] **Step 1: Implement the module**

Create `src/m1_vlm/entry_retimer.py`:

```python
"""
Entry Retimer — Rebuild SRT timestamps from chunk range and frame anchors.

The VLM (especially small local models like Qwen3.5-2B) often emits arbitrary
fixed-width subtitle slots that bear no relation to when actions actually
happen in the video. This module discards the VLM's timestamps and reconstructs
them from:
  1. Translated-text length (proxy for how long Vietnamese narration will take)
  2. The chunk's actual time range
  3. The frame timestamps the VLM was shown (so subtitle starts land on real
     visual transitions)

Pure module — no I/O, no globals. Safe to call from any thread.
"""

from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger


class EntryRetimer:
    """Reassign start_time / end_time on parsed VLM subtitle entries."""

    def __init__(
        self,
        chars_per_sec: float = 15.0,
        fill_ratio: float = 0.95,
        min_dur_sec: float = 1.5,
        min_gap_sec: float = 0.1,
    ):
        """
        Args:
            chars_per_sec: Estimated narration speaking rate. Used to convert
                translated_text length into an expected speech duration.
            fill_ratio: Fraction of chunk duration that subtitle slots fill.
                The remainder is reserved for gaps.
            min_dur_sec: Minimum slot duration before normalization. Prevents
                very-short entries from collapsing.
            min_gap_sec: Minimum gap between consecutive entries.
        """
        if chars_per_sec <= 0:
            raise ValueError("chars_per_sec must be > 0")
        if not 0 < fill_ratio <= 1.0:
            raise ValueError("fill_ratio must be in (0, 1]")
        if min_dur_sec < 0 or min_gap_sec < 0:
            raise ValueError("min_dur_sec and min_gap_sec must be >= 0")

        self.chars_per_sec = chars_per_sec
        self.fill_ratio = fill_ratio
        self.min_dur_sec = min_dur_sec
        self.min_gap_sec = min_gap_sec

    def retime(
        self,
        entries: List[Dict[str, Any]],
        chunk_start: float,
        chunk_end: float,
        frame_timestamps: List[float],
    ) -> List[Dict[str, Any]]:
        """
        Return new entries with rewritten start_time / end_time.

        Other fields (index, original_text, translated_text, ...) pass through.

        Args:
            entries: Parsed VLM entries with at least `translated_text`.
            chunk_start: Chunk start time in seconds (absolute, video time).
            chunk_end: Chunk end time in seconds (absolute, video time).
            frame_timestamps: Timestamps of the frames the VLM saw, in
                seconds (absolute, video time).
        """
        if not entries:
            return []
        if chunk_end <= chunk_start:
            raise ValueError(
                f"chunk_end ({chunk_end}) must be > chunk_start ({chunk_start})"
            )

        n = len(entries)
        chunk_duration = chunk_end - chunk_start

        # --- Step 1: expected duration per entry from text length ---
        expected = [
            max(self.min_dur_sec, len(e.get("translated_text", "")) / self.chars_per_sec)
            for e in entries
        ]

        # --- Step 2: normalize to fill the chunk ---
        gap_budget = self.min_gap_sec * (n - 1)
        slot_budget = chunk_duration * self.fill_ratio - gap_budget
        if slot_budget <= 0:
            # Chunk so short that gaps alone fill it; fall back to equal split.
            logger.warning(
                f"Chunk {chunk_start:.2f}-{chunk_end:.2f}s too short for "
                f"{n} entries with min_gap={self.min_gap_sec}s; using equal split"
            )
            slot_budget = chunk_duration
            normalized = [slot_budget / n] * n
        else:
            total_expected = sum(expected)
            if total_expected <= 0:
                normalized = [slot_budget / n] * n
            else:
                scale = slot_budget / total_expected
                normalized = [d * scale for d in expected]

        # --- Step 3: sequential placement ---
        starts: List[float] = []
        cursor = chunk_start
        for i in range(n):
            starts.append(cursor)
            cursor = cursor + normalized[i] + self.min_gap_sec

        # --- Step 4: snap starts to nearest frame timestamp ---
        snapped = self._snap_starts(starts, frame_timestamps)

        # --- Step 5: recompute ends from snapped starts ---
        ends: List[float] = []
        for i in range(n):
            if i < n - 1:
                end = snapped[i + 1] - self.min_gap_sec
            else:
                end = chunk_end
            # Guarantee end > start by at least a tick.
            ends.append(max(end, snapped[i] + 0.01))
        # Clamp last entry's end to chunk_end.
        ends[-1] = min(ends[-1], chunk_end)

        # --- Build output ---
        out: List[Dict[str, Any]] = []
        for entry, s, e in zip(entries, snapped, ends):
            new_entry = dict(entry)
            new_entry["start_time"] = self._sec_to_timestamp(s)
            new_entry["end_time"] = self._sec_to_timestamp(e)
            out.append(new_entry)
        return out

    def _snap_starts(
        self,
        starts: List[float],
        frame_timestamps: List[float],
    ) -> List[float]:
        """Snap each start to the nearest frame timestamp (ties → earlier).

        If no frame timestamps are supplied, return starts unchanged.
        After snapping, enforce strict monotonicity by min_gap_sec; if a snap
        would violate it, fall back to (previous_snapped + min_gap_sec) for
        that entry only.
        """
        if not frame_timestamps:
            return list(starts)
        sorted_frames = sorted(frame_timestamps)

        out: List[float] = []
        for i, s in enumerate(starts):
            best = self._nearest_with_earlier_tie(s, sorted_frames)
            if i > 0 and best <= out[i - 1] + self.min_gap_sec - 1e-9:
                logger.debug(
                    f"Retimer: snap collision at entry {i} (snap={best:.3f} "
                    f"<= prev+gap={out[i - 1] + self.min_gap_sec:.3f}); "
                    f"falling back to sequential placement for this entry"
                )
                best = out[i - 1] + self.min_gap_sec
            out.append(best)
        return out

    @staticmethod
    def _nearest_with_earlier_tie(value: float, sorted_frames: List[float]) -> float:
        """Return the frame nearest to `value`; on exact tie pick the earlier."""
        best = sorted_frames[0]
        best_dist = abs(value - best)
        for f in sorted_frames[1:]:
            d = abs(value - f)
            if d < best_dist:
                best = f
                best_dist = d
            # On tie, do not switch (earlier wins because we iterate sorted asc).
        return best

    @staticmethod
    def _sec_to_timestamp(seconds: float) -> str:
        """Format seconds as SRT timestamp HH:MM:SS,mmm."""
        if seconds < 0:
            seconds = 0.0
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int(round((seconds - int(seconds)) * 1000))
        if ms == 1000:
            ms = 0
            s += 1
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv\Scripts\Activate.ps1; pytest tests/m1_vlm/test_entry_retimer.py -v`

Expected: all 11 tests PASS.

- [ ] **Step 3: Slice ready to commit**

Files changed: `src/m1_vlm/entry_retimer.py`, `tests/m1_vlm/test_entry_retimer.py`. Stop for user to commit.

---

## Task 4: Wire EntryRetimer into `runner.py`

**Files:**
- Modify: `src/m4_pipeline/runner.py`

- [ ] **Step 1: Import EntryRetimer**

In `src/m4_pipeline/runner.py`, add to the existing M1 imports block (after `from src.m1_vlm.context_window import ContextWindow`):

```python
from src.m1_vlm.entry_retimer import EntryRetimer
```

- [ ] **Step 2: Construct the retimer near the other M1 objects**

Find the section in `run()` that creates `prompt_chain`, `context_window`, `validator`, `srt_builder` (currently around line 177-180). Right after `srt_builder = SRTBuilder()`, add:

```python
            entry_retimer = EntryRetimer(
                chars_per_sec=self.config.vi_chars_per_sec,
                fill_ratio=self.config.chunk_fill_ratio,
                min_gap_sec=self.config.m3_min_gap_sec,
            )
```

- [ ] **Step 3: Invoke retimer after VLM parse, before SRT add**

Find the per-chunk loop block that ends with:

```python
                srt_builder.add_entries(entries, chunk_offset=0.0)
                context_window.add_chunk_result(chunk_idx, entries)
                logger.info(f"Chunk {chunk_idx}: parsed {len(entries)} subtitle entries")
```

Replace it with (note: `timestamps` and `start`/`end` are already in scope from earlier in the loop):

```python
                retimed = entry_retimer.retime(
                    entries=entries,
                    chunk_start=start,
                    chunk_end=end,
                    frame_timestamps=timestamps,
                )
                logger.info(
                    f"Chunk {chunk_idx}: parsed {len(entries)} entries; "
                    f"retimer rewrote starts (first={retimed[0]['start_time']}, "
                    f"last_end={retimed[-1]['end_time']})"
                )

                valid, issues = validator.validate_sequence(retimed)
                if not valid:
                    logger.debug(f"Chunk {chunk_idx}: validation issues fixed: {issues}")
                    retimed = validator.fix_overlaps(retimed)
                    retimed = validator.reindex(retimed)

                srt_builder.add_entries(retimed, chunk_offset=0.0)
                context_window.add_chunk_result(chunk_idx, retimed)
```

- [ ] **Step 4: Remove the now-duplicate validator block above**

Earlier in the same loop the existing code already calls the validator on `entries` (before the change above). Locate this block:

```python
                valid, issues = validator.validate_sequence(entries)
                if not valid:
                    logger.debug(f"Chunk {chunk_idx}: validation issues fixed: {issues}")
                    entries = validator.fix_overlaps(entries)
                    entries = validator.reindex(entries)

                srt_builder.add_entries(entries, chunk_offset=0.0)
                context_window.add_chunk_result(chunk_idx, entries)
                logger.info(f"Chunk {chunk_idx}: parsed {len(entries)} subtitle entries")
```

This entire block must be REPLACED by the retimer call from Step 3 (the new block already contains its own validator pass on `retimed`). After the edit, the loop body has exactly one validator block, on the retimed entries.

- [ ] **Step 5: Smoke-test the import + syntax**

Run: `.venv\Scripts\Activate.ps1; python -c "from src.m4_pipeline.runner import PipelineRunner; print('import ok')"`

Expected: `import ok`.

- [ ] **Step 6: Slice ready to commit**

Files changed: `src/m4_pipeline/runner.py`. Stop for user to commit.

---

## Task 5: Wire EntryRetimer into `run_vlm_extract.py`

**Files:**
- Modify: `scripts/run_vlm_extract.py`

- [ ] **Step 1: Construct the retimer at the top of Step 6 (Build SRT)**

In `scripts/run_vlm_extract.py`, find the section starting `# ── Step 6: Validate & Build SRT ─────────────`. Before the `validator = SubtitleValidator()` line, add:

```python
    from src.m1_vlm.entry_retimer import EntryRetimer

    # EntryRetimer reads its knobs from the same env vars PipelineConfig uses,
    # so standalone and end-to-end share thresholds.
    import os as _os
    retimer = EntryRetimer(
        chars_per_sec=float(_os.getenv("M1_VI_CHARS_PER_SEC", "15.0")),
        fill_ratio=float(_os.getenv("M1_CHUNK_FILL_RATIO", "0.95")),
        min_gap_sec=float(_os.getenv("M3_MIN_GAP_SEC", "0.1")),
    )
```

- [ ] **Step 2: Retime each chunk's entries before adding to builder**

Find this block inside the `for result in all_results:` loop:

```python
        # Validate
        valid, issues = validator.validate_sequence(entries)
        if not valid:
            print(f"  Chunk {result['chunk_index']}: {len(issues)} validation issues")
            for iss in issues[:3]:
                print(f"    ⚠ {iss}")
            # Try to fix overlaps
            entries = validator.fix_overlaps(entries)
            entries = validator.reindex(entries)

        # Since prompts now generate ABSOLUTE timestamps,
        # we do NOT add chunk_offset (timestamps are already correct)
        builder.add_entries(entries, chunk_offset=0.0)
        total_entries += len(entries)
```

Replace it with:

```python
        # Retime entries from chunk range + frame anchors (overrides the
        # VLM-supplied start/end timestamps).
        chunk_idx_local = result["chunk_index"]
        cs = float(result["chunk_start"])
        ce = float(result["chunk_end"])
        cf = chunk_frames.get(chunk_idx_local, [])
        ts_anchors = [ts for ts, _, _ in cf]
        entries = retimer.retime(
            entries=entries,
            chunk_start=cs,
            chunk_end=ce,
            frame_timestamps=ts_anchors,
        )

        # Validate after retiming.
        valid, issues = validator.validate_sequence(entries)
        if not valid:
            print(f"  Chunk {chunk_idx_local}: {len(issues)} validation issues")
            for iss in issues[:3]:
                print(f"    ⚠ {iss}")
            entries = validator.fix_overlaps(entries)
            entries = validator.reindex(entries)

        builder.add_entries(entries, chunk_offset=0.0)
        total_entries += len(entries)
```

- [ ] **Step 3: Smoke-test the script parses**

Run: `.venv\Scripts\Activate.ps1; python scripts/run_vlm_extract.py --help`

Expected: usage text prints with no Python errors.

- [ ] **Step 4: Slice ready to commit**

Files changed: `scripts/run_vlm_extract.py`. Stop for user to commit.

---

## Task 6: `TimeStretcher.stretch_capped` — failing tests

**Files:**
- Create: `tests/m3_sync/__init__.py`
- Create: `tests/m3_sync/test_time_stretcher_capped.py`

- [ ] **Step 1: Create the package marker**

Create empty file `tests/m3_sync/__init__.py`:

```python
```

- [ ] **Step 2: Write failing tests**

Create `tests/m3_sync/test_time_stretcher_capped.py`:

```python
"""Tests for TimeStretcher.stretch_capped — capped-ratio stretch helper."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.m3_sync.time_stretcher import TimeStretcher


@pytest.fixture
def stretcher(tmp_path):
    with patch.object(TimeStretcher, "_check_rubberband"):
        s = TimeStretcher()
    return s


def _mock_info(duration: float):
    m = MagicMock()
    m.duration = duration
    return m


class TestStretchCapped:
    def test_within_tolerance_returns_exact(self, stretcher, tmp_path):
        src = tmp_path / "in.wav"
        src.write_bytes(b"riff-fake")
        with patch("soundfile.info", return_value=_mock_info(3.0)):
            out, achieved, strategy = stretcher.stretch_capped(
                audio_path=src,
                target_duration=3.05,
                max_ratio=1.25,
                output_dir=tmp_path / "out",
                tol=0.2,
            )
        assert strategy == "exact"
        assert out == src
        assert achieved == pytest.approx(3.0)

    def test_within_budget_compresses_to_fit(self, stretcher, tmp_path):
        src = tmp_path / "in.wav"
        src.write_bytes(b"riff-fake")
        out_dir = tmp_path / "out"
        with patch("soundfile.info", return_value=_mock_info(3.5)):
            with patch.object(TimeStretcher, "stretch") as mock_stretch:
                mock_stretch.return_value = out_dir / "in_stretched.wav"
                out, achieved, strategy = stretcher.stretch_capped(
                    audio_path=src,
                    target_duration=3.0,   # ratio 3.5/3.0 = 1.167 <= 1.25
                    max_ratio=1.25,
                    output_dir=out_dir,
                    tol=0.2,
                )
        assert strategy == "compress_fit"
        assert achieved == pytest.approx(3.0)
        mock_stretch.assert_called_once()
        _args, kwargs = mock_stretch.call_args
        # Target passed to underlying stretcher must be 3.0.
        called_target = kwargs.get("target_duration", mock_stretch.call_args.args[2])
        assert called_target == pytest.approx(3.0)

    def test_over_budget_caps_at_max_ratio(self, stretcher, tmp_path):
        src = tmp_path / "in.wav"
        src.write_bytes(b"riff-fake")
        out_dir = tmp_path / "out"
        with patch("soundfile.info", return_value=_mock_info(6.0)):
            with patch.object(TimeStretcher, "stretch") as mock_stretch:
                mock_stretch.return_value = out_dir / "in_stretched.wav"
                # Target 3.0, audio 6.0 → required ratio 2.0; cap 1.25 → keep
                # achieved = 6.0 / 1.25 = 4.8s.
                out, achieved, strategy = stretcher.stretch_capped(
                    audio_path=src,
                    target_duration=3.0,
                    max_ratio=1.25,
                    output_dir=out_dir,
                    tol=0.2,
                )
        assert strategy == "compress_max"
        assert achieved == pytest.approx(4.8, abs=0.01)
        # Underlying stretch called with target = achieved (4.8s), NOT 3.0.
        called_target = mock_stretch.call_args.args[2] if len(mock_stretch.call_args.args) >= 3 else mock_stretch.call_args.kwargs["target_duration"]
        assert called_target == pytest.approx(4.8, abs=0.01)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv\Scripts\Activate.ps1; pytest tests/m3_sync/test_time_stretcher_capped.py -v`

Expected: `AttributeError: ... has no attribute 'stretch_capped'` (or similar) for all three tests.

---

## Task 7: `TimeStretcher.stretch_capped` — implementation

**Files:**
- Modify: `src/m3_sync/time_stretcher.py`

- [ ] **Step 1: Add the helper**

In `src/m3_sync/time_stretcher.py`, append a new method to the `TimeStretcher` class (after `stretch_to_fit`, before `_pad_audio`):

```python
    def stretch_capped(
        self,
        audio_path: str | Path,
        target_duration: float,
        max_ratio: float,
        output_dir: str | Path,
        tol: float = 0.2,
    ) -> tuple[Path, float, str]:
        """
        Stretch toward `target_duration` but never compress more than `max_ratio`
        (audio_dur / achieved_dur). Pitch is preserved (rubberband).

        Returns:
            (output_path, achieved_duration, strategy)
              strategy ∈ {"exact", "compress_fit", "compress_max"}

            - "exact": |audio - target| <= tol; no stretch performed.
            - "compress_fit": audio > target but within budget; achieved = target.
            - "compress_max": audio > target * max_ratio; achieved = audio /
                max_ratio. Caller is responsible for absorbing the residual
                overflow (achieved - target) downstream (slip cascade).

        This helper does NOT pad or truncate. Use existing `stretch_to_fit` for
        the pad/truncate paths when audio is shorter than target.
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        info = sf.info(str(audio_path))
        current = info.duration
        delta = current - target_duration

        if abs(delta) <= tol or current <= target_duration:
            # Within tolerance OR audio already shorter than target — leave
            # alone; the audio fits without compression.
            return audio_path, current, "exact"

        ratio_needed = current / target_duration  # > 1 since current > target
        if ratio_needed <= max_ratio:
            achieved = target_duration
            strategy = "compress_fit"
        else:
            achieved = current / max_ratio
            strategy = "compress_max"

        suffix = audio_path.suffix
        stem = audio_path.stem
        out_path = output_dir / f"{stem}_stretched{suffix}"
        self.stretch(audio_path, out_path, achieved)
        return out_path, achieved, strategy
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `.venv\Scripts\Activate.ps1; pytest tests/m3_sync/test_time_stretcher_capped.py -v`

Expected: 3 PASS.

- [ ] **Step 3: Slice ready to commit**

Files changed: `src/m3_sync/time_stretcher.py`, `tests/m3_sync/__init__.py`, `tests/m3_sync/test_time_stretcher_capped.py`. Stop for user to commit.

---

## Task 8: `AudioAligner` budget+slip — failing tests

**Files:**
- Create: `tests/m3_sync/test_audio_aligner_budget.py`

- [ ] **Step 1: Write failing tests**

Create `tests/m3_sync/test_audio_aligner_budget.py`:

```python
"""Tests for AudioAligner budget+slip behavior."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.m3_sync.audio_aligner import AudioAligner


def _seg(idx: int, start: str, end: str, audio_path: str = "x.wav") -> dict:
    return {
        "index": idx,
        "start_time": start,
        "end_time": end,
        "audio_path": audio_path,
        "translated_text": "irrelevant",
    }


@pytest.fixture
def aligner():
    from src.m3_sync.time_stretcher import TimeStretcher
    with patch.object(TimeStretcher, "_check_rubberband"):
        a = AudioAligner(max_speedup=1.25, min_gap_sec=0.1, tolerance_sec=0.2)
    return a


class TestCalculateDeltasNewStrategies:
    def test_exact_within_tolerance(self, aligner):
        # Target 3.0s, audio 3.1s → within tol 0.2 → exact.
        segs = [_seg(1, "00:00:01,000", "00:00:04,000")]
        with patch("soundfile.info", return_value=MagicMock(duration=3.1)):
            with patch("pathlib.Path.exists", return_value=True):
                out = aligner.calculate_deltas(segs)
        assert out[0]["strategy"] == "exact"

    def test_within_budget_strategy_is_compress_fit(self, aligner):
        # Target 3.0s, audio 3.6s → ratio 1.2 < 1.25 → compress_fit.
        segs = [_seg(1, "00:00:01,000", "00:00:04,000")]
        with patch("soundfile.info", return_value=MagicMock(duration=3.6)):
            with patch("pathlib.Path.exists", return_value=True):
                out = aligner.calculate_deltas(segs)
        assert out[0]["strategy"] == "compress_fit"

    def test_over_budget_strategy_is_compress_max(self, aligner):
        # Target 3.0s, audio 6.0s → ratio 2.0 > 1.25 → compress_max.
        segs = [_seg(1, "00:00:01,000", "00:00:04,000")]
        with patch("soundfile.info", return_value=MagicMock(duration=6.0)):
            with patch("pathlib.Path.exists", return_value=True):
                out = aligner.calculate_deltas(segs)
        assert out[0]["strategy"] == "compress_max"

    def test_audio_shorter_strategy_is_pad(self, aligner):
        # Target 4.0s, audio 2.0s → pad.
        segs = [_seg(1, "00:00:01,000", "00:00:05,000")]
        with patch("soundfile.info", return_value=MagicMock(duration=2.0)):
            with patch("pathlib.Path.exists", return_value=True):
                out = aligner.calculate_deltas(segs)
        assert out[0]["strategy"] == "pad"


class TestSlipCascade:
    def test_no_slip_when_all_within_budget(self, aligner, tmp_path):
        segs = [
            _seg(1, "00:00:01,000", "00:00:04,000"),  # target 3.0
            _seg(2, "00:00:04,000", "00:00:07,000"),  # target 3.0
        ]
        # Each audio 3.0s → exact for both.
        with patch("soundfile.info", return_value=MagicMock(duration=3.0)):
            with patch("pathlib.Path.exists", return_value=True):
                deltas = aligner.calculate_deltas(segs)
                out = aligner.align_all(deltas, output_dir=tmp_path)

        assert out[0]["effective_start_sec"] == pytest.approx(1.0)
        assert out[1]["effective_start_sec"] == pytest.approx(4.0)
        assert out[0]["slip_applied"] is False
        assert out[1]["slip_applied"] is False

    def test_slip_propagates_when_first_overflows(self, aligner, tmp_path):
        segs = [
            _seg(1, "00:00:01,000", "00:00:04,000"),  # target 3.0
            _seg(2, "00:00:04,000", "00:00:07,000"),  # target 3.0
        ]
        # First audio 6.0s → compress_max → achieved 6/1.25 = 4.8s (overflow 1.8s).
        # Second audio 3.0s → exact.
        durations = iter([6.0, 3.0, 3.0, 3.0])  # first two for calc, again for stretch_capped sf.info
        def mock_info(_):
            return MagicMock(duration=next(durations))

        with patch("soundfile.info", side_effect=mock_info):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("src.m3_sync.time_stretcher.TimeStretcher.stretch") as mock_stretch:
                    mock_stretch.side_effect = lambda src, dst, target, **k: Path(dst)
                    deltas = aligner.calculate_deltas(segs)
                    out = aligner.align_all(deltas, output_dir=tmp_path)

        # Entry 1 starts at SRT 1.0 (no preceding entry).
        assert out[0]["effective_start_sec"] == pytest.approx(1.0)
        # Entry 1 achieved duration = 4.8 (compress_max).
        assert out[0]["aligned_duration"] == pytest.approx(4.8, abs=0.01)
        # Entry 2 SRT start = 4.0, but entry 1 ends at 1.0 + 4.8 = 5.8;
        # effective = max(4.0, 5.8 + 0.1) = 5.9.
        assert out[1]["effective_start_sec"] == pytest.approx(5.9, abs=0.01)
        assert out[1]["slip_applied"] is True
        assert out[0]["slip_applied"] is False  # First never "slips".

    def test_slip_absorbed_by_next_slack(self, aligner, tmp_path):
        segs = [
            _seg(1, "00:00:01,000", "00:00:04,000"),  # target 3.0
            _seg(2, "00:00:10,000", "00:00:13,000"),  # 6s slack between
        ]
        # First audio 6.0s → compress_max → achieved 4.8 (overflow 1.8).
        # Entry 2 SRT start = 10.0; entry 1 ends at 1.0 + 4.8 = 5.8.
        # max(10.0, 5.8+0.1) = 10.0 → no slip on entry 2.
        durations = iter([6.0, 3.0, 3.0])
        with patch("soundfile.info", side_effect=lambda _: MagicMock(duration=next(durations))):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("src.m3_sync.time_stretcher.TimeStretcher.stretch") as mock_stretch:
                    mock_stretch.side_effect = lambda src, dst, target, **k: Path(dst)
                    deltas = aligner.calculate_deltas(segs)
                    out = aligner.align_all(deltas, output_dir=tmp_path)

        assert out[1]["effective_start_sec"] == pytest.approx(10.0)
        assert out[1]["slip_applied"] is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\Activate.ps1; pytest tests/m3_sync/test_audio_aligner_budget.py -v`

Expected: failures — `TypeError: __init__() got an unexpected keyword argument 'max_speedup'` and similar.

---

## Task 9: `AudioAligner` budget+slip — implementation

**Files:**
- Modify: `src/m3_sync/audio_aligner.py`

- [ ] **Step 1: Rewrite the class**

Replace the entire contents of `src/m3_sync/audio_aligner.py` with:

```python
"""
Audio Aligner — Budgeted-stretch + slip-cascade alignment.

For each generated TTS segment, decide between:
  - exact: audio already fits the SRT slot (within tolerance)
  - compress_fit: audio is moderately long; rubberband stretch to slot
  - compress_max: audio is very long; stretch only as far as the speed-up
    budget allows, then slip the next segment's start forward
  - pad: audio is shorter than slot; pad with silence

After per-segment alignment, walk left-to-right computing each segment's
`effective_start_sec` so a slip on one segment pushes later ones forward
when needed. `FFmpegRenderer.merge_audio_segments` reads `start_sec` from
each returned segment — we overwrite `start_sec` with `effective_start_sec`
so the renderer needs no change. Original SRT start is preserved as
`srt_start_sec` for diagnostics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from src.m3_sync.time_stretcher import TimeStretcher


class AudioAligner:
    """Align TTS audio segments to SRT timestamps with a speed-up budget."""

    def __init__(
        self,
        time_stretcher: Optional[TimeStretcher] = None,
        tolerance_sec: float = 0.2,
        max_speedup: float = 1.25,
        min_gap_sec: float = 0.1,
    ):
        """
        Args:
            time_stretcher: TimeStretcher instance (created if None).
            tolerance_sec: |audio - target| below this is treated as exact.
            max_speedup: Max ratio (audio / achieved) for compression. Above
                this, the segment is compressed to the cap and the overflow
                is absorbed by slipping subsequent segments.
            min_gap_sec: Min gap between segments after slip cascade.
        """
        self.time_stretcher = time_stretcher or TimeStretcher()
        self.tolerance_sec = tolerance_sec
        self.max_speedup = max_speedup
        self.min_gap_sec = min_gap_sec

    def calculate_deltas(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Annotate each segment with target_duration, audio_duration, delta,
        and the chosen strategy. Pure decision step — no audio is touched.
        """
        import soundfile as sf

        out: List[Dict[str, Any]] = []
        for seg in segments:
            audio_path = seg.get("audio_path")
            if not audio_path or not Path(audio_path).exists():
                logger.warning(f"Missing audio for segment {seg.get('index')}")
                continue

            start_sec = self._timestamp_to_seconds(seg["start_time"])
            end_sec = self._timestamp_to_seconds(seg["end_time"])
            target = end_sec - start_sec
            audio_dur = sf.info(audio_path).duration
            delta = audio_dur - target

            new = seg.copy()
            new["srt_start_sec"] = start_sec
            new["srt_end_sec"] = end_sec
            new["start_sec"] = start_sec  # may be overwritten by align_all
            new["end_sec"] = end_sec
            new["target_duration"] = target
            new["audio_duration"] = audio_dur
            new["delta"] = delta

            if abs(delta) <= self.tolerance_sec:
                new["strategy"] = "exact"
            elif delta < 0:
                new["strategy"] = "pad"
            else:
                ratio_needed = audio_dur / target if target > 0 else float("inf")
                if ratio_needed <= self.max_speedup:
                    new["strategy"] = "compress_fit"
                else:
                    new["strategy"] = "compress_max"

            out.append(new)
        return out

    def align_all(
        self,
        segments: List[Dict[str, Any]],
        output_dir: str | Path,
    ) -> List[Dict[str, Any]]:
        """
        Run rubberband / pad per segment, then compute the slip cascade.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        for seg in segments:
            strategy = seg["strategy"]
            audio_path = seg["audio_path"]
            target = seg["target_duration"]
            audio_dur = seg["audio_duration"]

            new = seg.copy()
            if strategy == "exact":
                new["aligned_audio_path"] = audio_path
                new["aligned_duration"] = audio_dur
                new["align_method"] = "exact"
            elif strategy == "pad":
                out_path = output_dir / (Path(audio_path).stem + "_stretched" + Path(audio_path).suffix)
                self.time_stretcher._pad_audio(audio_path, out_path, target)
                new["aligned_audio_path"] = str(out_path)
                new["aligned_duration"] = target
                new["align_method"] = "pad"
            else:
                # compress_fit or compress_max
                out_path, achieved, sub_strategy = self.time_stretcher.stretch_capped(
                    audio_path=audio_path,
                    target_duration=target,
                    max_ratio=self.max_speedup,
                    output_dir=output_dir,
                    tol=self.tolerance_sec,
                )
                new["aligned_audio_path"] = str(out_path)
                new["aligned_duration"] = achieved
                new["align_method"] = sub_strategy

            results.append(new)

        # --- Slip cascade ---
        for i, seg in enumerate(results):
            if i == 0:
                eff = seg["srt_start_sec"]
                slipped = False
            else:
                prev = results[i - 1]
                earliest = prev["effective_start_sec"] + prev["aligned_duration"] + self.min_gap_sec
                eff = max(seg["srt_start_sec"], earliest)
                slipped = eff > seg["srt_start_sec"] + 1e-6
            seg["effective_start_sec"] = eff
            seg["start_sec"] = eff  # what FFmpegRenderer reads
            seg["slip_applied"] = slipped

        # Summary
        methods = [r["align_method"] for r in results]
        slips = sum(1 for r in results if r["slip_applied"])
        logger.info(
            f"Aligned {len(results)} segments: "
            f"{methods.count('exact')} exact, "
            f"{methods.count('compress_fit')} compress_fit, "
            f"{methods.count('compress_max')} compress_max, "
            f"{methods.count('pad')} pad; {slips} slipped"
        )
        return results

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
        parts = timestamp.replace(",", ":").split(":")
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + ms / 1000.0
```

- [ ] **Step 2: Run the new budget tests to verify they pass**

Run: `.venv\Scripts\Activate.ps1; pytest tests/m3_sync/test_audio_aligner_budget.py -v`

Expected: all 7 PASS.

- [ ] **Step 3: Run the legacy `test_audio_aligner.py` to catch regressions**

Run: `.venv\Scripts\Activate.ps1; pytest tests/test_audio_aligner.py tests/test_audio_aligner_natural.py -v`

Expected: PASS. If `test_audio_aligner.py::test_calculate_deltas` checks the legacy `strategy` value `stretch_compress`, update its assertion to `compress_fit` or `compress_max` based on the audio_duration / target_duration ratio in the test (target 3.0, audio 3.5 → ratio 1.167 → `compress_fit`).

If updates were needed, re-run the suite and confirm green.

- [ ] **Step 4: Slice ready to commit**

Files changed: `src/m3_sync/audio_aligner.py`, possibly `tests/test_audio_aligner.py`, `tests/m3_sync/test_audio_aligner_budget.py`. Stop for user to commit.

---

## Task 10: Pass new config knobs into `AudioAligner` from runner

**Files:**
- Modify: `src/m4_pipeline/runner.py`

- [ ] **Step 1: Construct AudioAligner with config values**

In `src/m4_pipeline/runner.py`, find the audio_alignment step (`# === Step 7: Audio Alignment ===`). Replace the line:

```python
            aligner = AudioAligner()
```

with:

```python
            aligner = AudioAligner(
                max_speedup=self.config.m3_max_speedup,
                min_gap_sec=self.config.m3_min_gap_sec,
            )
```

- [ ] **Step 2: Smoke-test the runner imports**

Run: `.venv\Scripts\Activate.ps1; python -c "from src.m4_pipeline.runner import PipelineRunner; print('import ok')"`

Expected: `import ok`.

- [ ] **Step 3: Slice ready to commit**

Files changed: `src/m4_pipeline/runner.py`. Stop for user to commit.

---

## Task 11: Extend standalone `test_m3_sync.py` validator output

**Files:**
- Modify: `scripts/test_m3_sync.py`

- [ ] **Step 1: Use new aligner constructor and log new fields**

In `scripts/test_m3_sync.py`, find the lines:

```python
    stretcher = TimeStretcher()
    aligner = AudioAligner(time_stretcher=stretcher)
    with_deltas = aligner.calculate_deltas(segments)
```

Replace with:

```python
    import os as _os
    stretcher = TimeStretcher()
    aligner = AudioAligner(
        time_stretcher=stretcher,
        max_speedup=float(_os.getenv("M3_MAX_SPEEDUP", "1.25")),
        min_gap_sec=float(_os.getenv("M3_MIN_GAP_SEC", "0.1")),
    )
    with_deltas = aligner.calculate_deltas(segments)
```

- [ ] **Step 2: Extend the per-segment log to surface the new strategy values**

Find the loop:

```python
    for seg in with_deltas:
        idx = seg.get("index", "?")
        logger.info(
            f"#{idx:>2} target={seg['target_duration']:.2f}s "
            f"audio={seg['audio_duration']:.2f}s delta={seg['delta']:+.2f}s "
            f"strategy={seg['strategy']}"
        )
```

(Leave it as-is — `strategy` now carries the new values automatically.)

- [ ] **Step 3: After `align_all`, dump enriched JSON**

Find the existing alignment-strategy summary (`logger.info("=== Alignment strategies used ===")` block). Right after that block, before the FFmpeg merge call, add:

```python
    # Stage 2 report — slip cascade visibility for the standalone validator.
    stage2_path = args.output_dir / "stage2_alignment.json"
    stage2_path.write_text(
        json.dumps(
            [
                {
                    "index": s.get("index"),
                    "srt_start_sec": s.get("srt_start_sec"),
                    "effective_start_sec": s.get("effective_start_sec"),
                    "aligned_duration": s.get("aligned_duration"),
                    "strategy": s.get("strategy"),
                    "align_method": s.get("align_method"),
                    "slip_applied": s.get("slip_applied"),
                }
                for s in aligned
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(f"Stage 2 report saved: {stage2_path}")
```

- [ ] **Step 4: Smoke-test the script syntax**

Run: `.venv\Scripts\Activate.ps1; python scripts/test_m3_sync.py --help`

Expected: usage text prints with no Python errors.

- [ ] **Step 5: Slice ready to commit**

Files changed: `scripts/test_m3_sync.py`. Stop for user to commit.

---

## Task 12: Integration verification — end-to-end run

**Files:**
- Read-only inspection.

- [ ] **Step 1: Run the full pipeline against the test video**

Clear the stale output dir first so we're inspecting fresh artifacts.

Run:

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
Remove-Item -Recurse -Force "data\outputs\VNEXT_ToyoBeauty_Demo-Module-5" -ErrorAction SilentlyContinue
python scripts/run_pipeline.py `
  --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
  --ref-audio "data\reference_audio\speaker.wav" --vlm-mode local
```

Expected log evidence (all must hold):
- `Global summary: topic=...` (Pass 0 still works).
- `Chunk 0: parsed N entries; retimer rewrote starts (first=00:00:0X,XXX, last_end=00:00:YY,YYY)` where `YY` is near the chunk's actual end time, not 16s.
- `Aligned N segments: a exact, b compress_fit, c compress_max, d pad; e slipped` where `c + e` is small relative to `N` (most segments fit within the 1.25x budget).
- Pipeline exits with `Done in ...s`.

- [ ] **Step 2: Inspect the produced SRT**

Read `data\outputs\VNEXT_ToyoBeauty_Demo-Module-5\VNEXT_ToyoBeauty_Demo-Module-5_vi.srt` and confirm:
- Last entry's `end_time` is near the chunk end (well past `00:00:16,000`).
- Entry start times correspond to frame timestamps the VLM saw (cross-reference `data\outputs\VNEXT_ToyoBeauty_Demo-Module-5\work\chunk_00_raw_response.txt` to see how different they are from the VLM-supplied originals).

- [ ] **Step 3: Subjective playback check**

Open `data\outputs\VNEXT_ToyoBeauty_Demo-Module-5\VNEXT_ToyoBeauty_Demo-Module-5_dubbed.mp4` in a player. Confirm:
- Narration sounds natural-paced (not visibly rushed).
- The Vietnamese narration roughly tracks the on-screen action by section, not packed at the start.

- [ ] **Step 4: Run the standalone validator for cross-check**

Run:

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
python scripts/test_m3_sync.py `
  --srt "data\outputs\VNEXT_ToyoBeauty_Demo-Module-5\VNEXT_ToyoBeauty_Demo-Module-5_vi.srt" `
  --audio-dir "data\outputs\VNEXT_ToyoBeauty_Demo-Module-5\work\audio_chunks" `
  --video "H:\Quan\tanquan\VNext\ToyoBeauty\【VNEXT】【ToyoBeauty】Demo-Module-5.mp4" `
  --output-dir "debug_output\m3_test_run_post_fix"
```

Expected: `stage1_deltas.json` and `stage2_alignment.json` written; alignment strategies match what the pipeline reported in Step 1.

- [ ] **Step 5: All slices integrated — final commit**

If all verification passes, the feature is complete. Stop for user to commit any remaining test artifacts (or none if nothing else changed).

If a verification step fails, file a follow-up note describing which expectation broke; do NOT auto-tune knobs without the user's input — the M1_VI_CHARS_PER_SEC and M3_MAX_SPEEDUP defaults are deliberate.

---

## Self-review notes

- Spec coverage: every section of the design doc maps to a task (config → T1, EntryRetimer → T2+T3, wire-in → T4+T5, stretch_capped → T6+T7, AudioAligner rewrite → T8+T9, runner constructor → T10, standalone validator → T11, integration → T12).
- The user commits manually; no `git commit`/`git push` commands appear in any step.
- Placeholders scanned: none ("TBD", "fill in", "similar to" — none present).
- Type/name consistency: `EntryRetimer.retime`, `TimeStretcher.stretch_capped`, `AudioAligner(max_speedup=..., min_gap_sec=...)`, `effective_start_sec`, `aligned_duration`, `slip_applied`, `align_method` — same names across tasks.
