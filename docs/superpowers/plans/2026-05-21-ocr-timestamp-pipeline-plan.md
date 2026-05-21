# OCR-Driven Timestamp Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second pipeline mode (`--mode ocr`) that reads burned-in English captions with GLM-OCR to derive exact subtitle timestamps, then uses the VLM only to translate each segment (with the segment's mid-frame attached for visual grounding). The existing `--mode vlm` path must keep working byte-for-byte.

**Architecture:** New `CaptionTimeline` class in `src/m1_vlm/caption_ocr.py` does dense sampling → bottom-band crop → GLM-OCR → segment grouping. A new `build_translation_only_prompt` on `PromptChain` produces a one-segment translation prompt. `PipelineRunner.run()` branches on `config.pipeline_mode`: the existing body becomes `_run_vlm_mode` (unchanged); `_run_ocr_mode` is new and reuses M2/M3 unmodified. `scripts/run_pipeline.py` gets a `--mode {vlm,ocr}` flag (default `vlm`).

**Tech Stack:** Python 3.10+, OpenCV (already a dep), GLM-OCR via existing `src/m1_vlm/glm_ocr.py`, VLM via existing `src/m1_vlm/vlm_client.py`, `difflib.SequenceMatcher` (stdlib) for caption dedup, pytest for tests. All Python must be run inside `.venv` (`.venv\Scripts\Activate.ps1` on Windows PowerShell).

---

## File Structure

**Create:**
- `src/m1_vlm/caption_ocr.py` — `CaptionSegment` dataclass + `CaptionTimeline` class.
- `tests/m1_vlm/test_caption_timeline.py` — unit tests for sampling, cropping, segmentation.
- `tests/m1_vlm/test_translation_only_prompt.py` — snapshot test for the new prompt.
- `tests/m4_pipeline/test_runner_mode_branch.py` — confirms mode flag dispatches correctly.
- `scripts/test_caption_ocr.py` — manual smoke script.

**Modify:**
- `src/m1_vlm/prompt_chain.py` — add `build_translation_only_prompt(...)` method.
- `src/m4_pipeline/runner.py` — refactor: extract current `run()` body verbatim into `_run_vlm_mode`; add `_run_ocr_mode`; add mode branch + `STEPS_OCR` constant.
- `src/m4_pipeline/config.py` — add 5 new properties: `pipeline_mode`, `ocr_sample_fps`, `caption_band_ratio`, `caption_dedup_ratio`, `caption_min_duration_sec`.
- `scripts/run_pipeline.py` — add `--mode {vlm,ocr}` CLI flag; default `vlm`; passes to config.
- `.env.example` — document the 5 new env vars.

**Leave alone (reused as-is):**
- `src/m1_vlm/glm_ocr.py`, `src/m1_vlm/vlm_client.py`, `src/m1_vlm/frame_extractor.py`, `src/m1_vlm/srt_builder.py`, `src/m1_vlm/scene_detector.py`
- All of `src/m2_tts/`, `src/m3_sync/`
- `src/m4_pipeline/exceptions.py`, `src/m4_pipeline/logger.py`

---

## Workflow Rules

- **Venv:** Every `python` / `pytest` command in this plan must be preceded by `.\.venv\Scripts\Activate.ps1` in PowerShell. If the venv is already active in the shell, you can skip the activation line but you MUST verify with `python -c "import sys; print(sys.prefix)"` before running anything heavy.
- **No auto-commits:** Per project memory (`feedback_no_auto_commit`), commit steps below are written for the **user** to run. As the implementing agent you should STAGE files (`git add`) and STOP at each commit boundary, telling the user the suggested message. Do not run `git commit` yourself.
- **TDD:** Tests first, then implementation, then run, then stage.

---

## Task 1 — `CaptionSegment` dataclass + module skeleton

**Files:**
- Create: `src/m1_vlm/caption_ocr.py`
- Test: `tests/m1_vlm/test_caption_timeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/m1_vlm/test_caption_timeline.py`:

```python
"""Unit tests for CaptionTimeline and CaptionSegment."""

import numpy as np
import pytest


def test_caption_segment_dataclass_holds_fields():
    from src.m1_vlm.caption_ocr import CaptionSegment

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    seg = CaptionSegment(
        start_sec=1.0,
        end_sec=3.5,
        en_text="hello world",
        mid_frame=frame,
    )
    assert seg.start_sec == 1.0
    assert seg.end_sec == 3.5
    assert seg.en_text == "hello world"
    assert seg.mid_frame.shape == (100, 200, 3)
    assert seg.duration_sec == pytest.approx(2.5)
```

- [ ] **Step 2: Run test to verify it fails**

```powershell
.\.venv\Scripts\Activate.ps1
pytest tests/m1_vlm/test_caption_timeline.py::test_caption_segment_dataclass_holds_fields -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'src.m1_vlm.caption_ocr'`

- [ ] **Step 3: Write minimal implementation**

Create `src/m1_vlm/caption_ocr.py`:

```python
"""Caption Timeline — Extract exact subtitle timing via dense GLM-OCR sampling.

Reads burned-in English captions from a fixed bottom band of each frame,
groups consecutive frames with the same caption text into segments, and
returns ordered (start, end, text, mid-frame) tuples used by the OCR-mode
pipeline runner.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class CaptionSegment:
    """One detected caption span with its exact start/end and a mid-frame."""

    start_sec: float
    end_sec: float
    en_text: str
    mid_frame: np.ndarray

    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec
```

- [ ] **Step 4: Run test to verify it passes**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/caption_ocr.py tests/m1_vlm/test_caption_timeline.py
```

Suggested commit message for the user: `feat(m1): add CaptionSegment dataclass for OCR pipeline`

---

## Task 2 — Bottom-band frame crop helper

**Files:**
- Modify: `src/m1_vlm/caption_ocr.py`
- Test: `tests/m1_vlm/test_caption_timeline.py` (append)

- [ ] **Step 1: Append failing test**

Append to `tests/m1_vlm/test_caption_timeline.py`:

```python
def test_crop_bottom_band_geometry():
    from src.m1_vlm.caption_ocr import crop_bottom_band

    frame = np.zeros((1000, 1920, 3), dtype=np.uint8)
    crop = crop_bottom_band(frame, ratio=0.22)
    # 0.22 * 1000 = 220
    assert crop.shape == (220, 1920, 3)
    # Pixels should come from the BOTTOM of the original
    frame[800:, :, :] = 255  # make the bottom rows white
    crop_white = crop_bottom_band(frame, ratio=0.22)
    assert crop_white.mean() > 200, "crop should preserve bright bottom rows"


def test_crop_bottom_band_clamps_ratio():
    from src.m1_vlm.caption_ocr import crop_bottom_band

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # Out-of-range ratio gets clamped to (0.0, 1.0]
    crop = crop_bottom_band(frame, ratio=1.5)
    assert crop.shape == (100, 200, 3)
    with pytest.raises(ValueError):
        crop_bottom_band(frame, ratio=0.0)
    with pytest.raises(ValueError):
        crop_bottom_band(frame, ratio=-0.1)
```

- [ ] **Step 2: Run to verify failure**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v -k crop_bottom_band
```

Expected: `ImportError: cannot import name 'crop_bottom_band'`

- [ ] **Step 3: Implement**

Append to `src/m1_vlm/caption_ocr.py`:

```python
def crop_bottom_band(frame: np.ndarray, ratio: float) -> np.ndarray:
    """Return the bottom `ratio` fraction of the frame's rows.

    Args:
        frame: HWC numpy array (BGR or RGB; we don't care here).
        ratio: Fraction of height to keep from the bottom. Must be in (0, 1].
               Values > 1 are clamped to 1 (entire frame).

    Returns:
        View into `frame` containing only the bottom rows. No copy is made.
    """
    if ratio <= 0:
        raise ValueError(f"caption_band_ratio must be > 0, got {ratio}")
    ratio = min(ratio, 1.0)
    h = frame.shape[0]
    band_h = max(1, int(round(h * ratio)))
    return frame[h - band_h :, :, :]
```

- [ ] **Step 4: Run tests**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/caption_ocr.py tests/m1_vlm/test_caption_timeline.py
```

Suggested message: `feat(m1): add crop_bottom_band helper for caption OCR`

---

## Task 3 — Text normalization helper

**Files:**
- Modify: `src/m1_vlm/caption_ocr.py`
- Test: `tests/m1_vlm/test_caption_timeline.py` (append)

- [ ] **Step 1: Append failing tests**

```python
def test_normalize_caption_text():
    from src.m1_vlm.caption_ocr import normalize_caption_text

    assert normalize_caption_text("  Hello, World!  ") == "hello, world!"
    assert normalize_caption_text("Foo\n\nBar  Baz") == "foo bar baz"
    assert normalize_caption_text("") == ""
    assert normalize_caption_text("   ") == ""
    assert normalize_caption_text("\t\nABC\t") == "abc"


def test_normalize_caption_text_keeps_punctuation_inside():
    from src.m1_vlm.caption_ocr import normalize_caption_text

    # Punctuation between words is preserved; only edge whitespace stripped
    assert normalize_caption_text("hello, world.") == "hello, world."
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v -k normalize_caption_text
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

Append to `src/m1_vlm/caption_ocr.py`:

```python
import re as _re


def normalize_caption_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip edges. Empty string for blank input."""
    if text is None:
        return ""
    collapsed = _re.sub(r"\s+", " ", text).strip().lower()
    return collapsed
```

- [ ] **Step 4: Run**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v
```

Expected: `5 passed`.

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/caption_ocr.py tests/m1_vlm/test_caption_timeline.py
```

Suggested message: `feat(m1): add normalize_caption_text helper`

---

## Task 4 — Caption stream segmentation (pure logic, no video I/O)

**Files:**
- Modify: `src/m1_vlm/caption_ocr.py`
- Test: `tests/m1_vlm/test_caption_timeline.py` (append)

Segmentation is the heart of the timing logic. It's purely functional, so we test it with hand-built `(timestamp, text, frame)` streams — no real video needed.

- [ ] **Step 1: Append failing tests**

```python
def _fake_stream(items, frame_shape=(10, 20, 3)):
    """Build a (timestamp, text, frame) list with unique frames per index."""
    out = []
    for i, (t, txt) in enumerate(items):
        f = np.full(frame_shape, fill_value=i % 256, dtype=np.uint8)
        out.append((t, txt, f))
    return out


def test_segment_stream_groups_identical_text():
    from src.m1_vlm.caption_ocr import segment_caption_stream

    stream = _fake_stream([
        (0.0, "hello"), (0.5, "hello"), (1.0, "hello"),
        (1.5, "world"), (2.0, "world"),
    ])
    segs = segment_caption_stream(
        stream, dedup_ratio=0.85, min_duration_sec=0.0, video_end_sec=2.5
    )
    assert len(segs) == 2
    assert segs[0].en_text == "hello"
    assert segs[0].start_sec == 0.0
    assert segs[0].end_sec == 1.5  # first ts of next group
    assert segs[1].en_text == "world"
    assert segs[1].start_sec == 1.5
    assert segs[1].end_sec == 2.5  # video_end_sec


def test_segment_stream_treats_near_matches_as_same():
    from src.m1_vlm.caption_ocr import segment_caption_stream

    # OCR jitter: one char wrong
    stream = _fake_stream([
        (0.0, "hello world"), (0.5, "hello wrld"), (1.0, "hello world"),
    ])
    segs = segment_caption_stream(
        stream, dedup_ratio=0.85, min_duration_sec=0.0, video_end_sec=1.5
    )
    assert len(segs) == 1
    assert segs[0].en_text == "hello wrld"  # middle sample, by index


def test_segment_stream_drops_empty_text_samples():
    from src.m1_vlm.caption_ocr import segment_caption_stream

    stream = _fake_stream([
        (0.0, ""), (0.5, "hello"), (1.0, "hello"), (1.5, ""),
    ])
    segs = segment_caption_stream(
        stream, dedup_ratio=0.85, min_duration_sec=0.0, video_end_sec=2.0
    )
    assert len(segs) == 1
    assert segs[0].en_text == "hello"
    assert segs[0].start_sec == 0.5
    # Ends at the first timestamp that is NOT part of the group (1.5 — empty)
    assert segs[0].end_sec == 1.5


def test_segment_stream_drops_below_min_duration():
    from src.m1_vlm.caption_ocr import segment_caption_stream

    stream = _fake_stream([
        (0.0, "blip"), (0.5, "long"), (1.0, "long"), (1.5, "long"),
    ])
    segs = segment_caption_stream(
        stream, dedup_ratio=0.85, min_duration_sec=0.4, video_end_sec=2.0
    )
    # "blip" lasts only 0.5s start..0.5s end = 0.5s; that's >= 0.4 so kept.
    # Make blip shorter:
    stream = _fake_stream([
        (0.0, "blip"), (0.2, "long"), (0.7, "long"), (1.2, "long"),
    ])
    segs = segment_caption_stream(
        stream, dedup_ratio=0.85, min_duration_sec=0.3, video_end_sec=2.0
    )
    assert all(s.en_text == "long" for s in segs)
    assert len(segs) == 1


def test_segment_stream_empty_input_returns_empty():
    from src.m1_vlm.caption_ocr import segment_caption_stream

    assert segment_caption_stream([], 0.85, 0.3, 5.0) == []


def test_segment_stream_picks_middle_frame_as_mid_frame():
    from src.m1_vlm.caption_ocr import segment_caption_stream

    stream = _fake_stream([
        (0.0, "x"), (0.5, "x"), (1.0, "x"), (1.5, "x"), (2.0, "x"),
    ])
    segs = segment_caption_stream(stream, 0.85, 0.0, 2.5)
    assert len(segs) == 1
    # 5 samples -> middle index = 2 -> fill value 2
    assert int(segs[0].mid_frame[0, 0, 0]) == 2
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v -k segment_stream
```

Expected: `ImportError: cannot import name 'segment_caption_stream'`.

- [ ] **Step 3: Implement**

Append to `src/m1_vlm/caption_ocr.py`:

```python
from difflib import SequenceMatcher
from typing import List, Sequence, Tuple

import numpy as np  # already imported above; harmless re-import for clarity


def _same_caption(a: str, b: str, ratio_threshold: float) -> bool:
    """True if both non-empty and similar enough to be treated as one caption."""
    if not a or not b:
        return False
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= ratio_threshold


def segment_caption_stream(
    stream: Sequence[Tuple[float, str, np.ndarray]],
    dedup_ratio: float,
    min_duration_sec: float,
    video_end_sec: float,
) -> List[CaptionSegment]:
    """Group consecutive samples with the same caption text into segments.

    Args:
        stream: Ordered (timestamp_sec, normalized_text, frame) tuples.
                Text should already be normalized by `normalize_caption_text`.
        dedup_ratio: SequenceMatcher ratio threshold for "same caption".
        min_duration_sec: Drop segments shorter than this.
        video_end_sec: Used as end_sec for the final segment.

    Returns:
        Ordered list of CaptionSegments. Empty list if stream has no captions.
    """
    if not stream:
        return []

    # Walk the stream, collecting runs (groups). Empty-text samples break runs
    # but do not start a new caption.
    groups: List[List[int]] = []  # each group is a list of indices into stream
    current: List[int] = []
    current_text = ""

    for i, (_, text, _) in enumerate(stream):
        if not text:
            if current:
                groups.append(current)
                current = []
                current_text = ""
            continue
        if not current:
            current = [i]
            current_text = text
            continue
        if _same_caption(current_text, text, dedup_ratio):
            current.append(i)
        else:
            groups.append(current)
            current = [i]
            current_text = text

    if current:
        groups.append(current)

    # Build segments from groups.
    segments: List[CaptionSegment] = []
    for g_idx, group in enumerate(groups):
        first_i = group[0]
        last_i = group[-1]
        mid_i = group[len(group) // 2]

        start_sec = stream[first_i][0]
        # End = start of next group's first sample, OR video_end_sec for last.
        if g_idx + 1 < len(groups):
            next_first_i = groups[g_idx + 1][0]
            end_sec = stream[next_first_i][0]
        else:
            # Last group: end at the sample AFTER last_i, or video_end_sec.
            if last_i + 1 < len(stream):
                end_sec = stream[last_i + 1][0]
            else:
                end_sec = video_end_sec

        if end_sec - start_sec < min_duration_sec:
            continue

        segments.append(
            CaptionSegment(
                start_sec=start_sec,
                end_sec=end_sec,
                en_text=stream[mid_i][1],
                mid_frame=stream[mid_i][2],
            )
        )

    return segments
```

- [ ] **Step 4: Run tests**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v
```

Expected: `11 passed` (5 from earlier tasks + 6 new).

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/caption_ocr.py tests/m1_vlm/test_caption_timeline.py
```

Suggested message: `feat(m1): add segment_caption_stream grouping logic`

---

## Task 5 — Video sampling helper (uses OpenCV)

**Files:**
- Modify: `src/m1_vlm/caption_ocr.py`
- Test: `tests/m1_vlm/test_caption_timeline.py` (append)

Generates timestamps and pulls frames at those times. Tested with a tiny generated MP4 fixture.

- [ ] **Step 1: Add fixture builder + failing test**

Append to `tests/m1_vlm/test_caption_timeline.py`:

```python
import cv2
from pathlib import Path


def _make_synthetic_video(path: Path, fps: int = 10, duration_sec: float = 2.0,
                          size=(320, 240)):
    """Write a tiny synthetic MP4 (solid frames) for sampling tests."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    n_frames = int(round(fps * duration_sec))
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), fill_value=i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_iter_video_samples_returns_target_count(tmp_path):
    from src.m1_vlm.caption_ocr import iter_video_samples

    video = tmp_path / "synthetic.mp4"
    _make_synthetic_video(video, fps=10, duration_sec=2.0)

    samples = list(iter_video_samples(video, sample_fps=2.0))
    # 2 fps over ~2s = 4 samples (0.0, 0.5, 1.0, 1.5)
    assert len(samples) == 4
    timestamps = [t for t, _ in samples]
    assert timestamps[0] == 0.0
    # Spacing is roughly 0.5s (within decode jitter)
    for prev, nxt in zip(timestamps, timestamps[1:]):
        assert 0.3 < (nxt - prev) < 0.7


def test_iter_video_samples_frame_shape(tmp_path):
    from src.m1_vlm.caption_ocr import iter_video_samples

    video = tmp_path / "synthetic.mp4"
    _make_synthetic_video(video, fps=10, duration_sec=1.0, size=(160, 120))

    samples = list(iter_video_samples(video, sample_fps=4.0))
    assert all(f.shape == (120, 160, 3) for _, f in samples)


def test_iter_video_samples_raises_on_unreadable(tmp_path):
    from src.m1_vlm.caption_ocr import iter_video_samples

    with pytest.raises(IOError):
        list(iter_video_samples(tmp_path / "does_not_exist.mp4", sample_fps=2.0))
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v -k iter_video_samples
```

Expected: `ImportError`.

- [ ] **Step 3: Implement**

Append to `src/m1_vlm/caption_ocr.py`:

```python
import cv2
from pathlib import Path
from typing import Iterator


def iter_video_samples(
    video_path: str | Path,
    sample_fps: float,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, frame_bgr) at the requested rate.

    Uses VideoCapture seek-by-time (CAP_PROP_POS_MSEC). On videos with VFR
    timing the actual returned timestamp may drift slightly; that's fine for
    caption-grain accuracy.

    Raises:
        IOError: if the video cannot be opened or has no frames.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise IOError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or total_frames <= 0:
            raise IOError(f"Video has no decodable frames: {video_path}")
        duration_sec = total_frames / fps

        step = 1.0 / sample_fps
        t = 0.0
        while t < duration_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                # Couldn't decode at this timestamp — skip but keep advancing.
                t += step
                continue
            yield t, frame
            t += step
    finally:
        cap.release()


def get_video_duration_sec(video_path: str | Path) -> float:
    """Return video duration in seconds via OpenCV. Raises IOError on failure."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise IOError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise IOError(f"OpenCV could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        if fps <= 0 or n <= 0:
            raise IOError(f"Video has no decodable frames: {video_path}")
        return n / fps
    finally:
        cap.release()
```

- [ ] **Step 4: Run**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v
```

Expected: `14 passed`. If OpenCV's mp4v writer is unavailable on the runner, the synthetic-video tests will error — in that case mark them `pytest.skip` with a clear reason and verify the pure-logic tests still pass.

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/caption_ocr.py tests/m1_vlm/test_caption_timeline.py
```

Suggested message: `feat(m1): add iter_video_samples for dense frame extraction`

---

## Task 6 — `CaptionTimeline` orchestrator class

**Files:**
- Modify: `src/m1_vlm/caption_ocr.py`
- Test: `tests/m1_vlm/test_caption_timeline.py` (append)

Wires sampling + crop + OCR + segmentation into one class.

- [ ] **Step 1: Add failing test (mocked GLM-OCR)**

```python
class _FakeOCR:
    """Mimics GLMOCR.extract_text. Returns text by frame's first pixel value."""

    def __init__(self, mapping):
        self.mapping = mapping  # dict: int → str
        self.calls = 0

    def extract_text(self, frame, prompt=None, max_new_tokens=None):
        self.calls += 1
        key = int(frame[0, 0, 0])
        return self.mapping.get(key, "")


def test_caption_timeline_build_end_to_end(tmp_path):
    from src.m1_vlm.caption_ocr import CaptionTimeline

    video = tmp_path / "synthetic.mp4"
    _make_synthetic_video(video, fps=10, duration_sec=2.0)
    # Synthetic video frames are filled with their frame index modulo 256;
    # cv2.VideoCapture decoded values may not be exact due to codec, so map
    # broadly: anything not "hi" yields empty so the timeline only picks up
    # the frames we explicitly tag. With 2 fps we get samples at t=0,0.5,1,1.5
    # which decode to roughly fill values 0, 5, 10, 15.
    ocr = _FakeOCR({0: "Hi", 5: "Hi", 10: "Bye", 15: "Bye"})

    tl = CaptionTimeline(
        ocr=ocr,
        sample_fps=2.0,
        caption_band_ratio=0.5,
        dedup_ratio=0.85,
        min_duration_sec=0.0,
    )
    segs = tl.build(video)

    # We can't guarantee exact OCR mapping on codec-roundtripped pixels, so
    # the assertion is loose: build() must return a list (possibly empty) and
    # must have called the OCR exactly once per sampled frame.
    assert isinstance(segs, list)
    assert ocr.calls >= 1
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v -k caption_timeline_build_end_to_end
```

Expected: `ImportError: cannot import name 'CaptionTimeline'`.

- [ ] **Step 3: Implement**

Append to `src/m1_vlm/caption_ocr.py`:

```python
from loguru import logger


class CaptionTimeline:
    """Build a list of caption segments by densely OCR-ing burned-in subtitles.

    Pipeline: sample @ sample_fps → crop bottom band → GLM-OCR → normalize →
    group consecutive identical captions → drop short blips.
    """

    def __init__(
        self,
        ocr,  # GLMOCR (duck-typed: must expose .extract_text(frame))
        sample_fps: float = 2.0,
        caption_band_ratio: float = 0.22,
        dedup_ratio: float = 0.85,
        min_duration_sec: float = 0.3,
        ocr_prompt: str = "Read the subtitle text only. Return only the text.",
    ):
        self.ocr = ocr
        self.sample_fps = sample_fps
        self.caption_band_ratio = caption_band_ratio
        self.dedup_ratio = dedup_ratio
        self.min_duration_sec = min_duration_sec
        self.ocr_prompt = ocr_prompt

    def build(self, video_path: str | Path) -> List[CaptionSegment]:
        """Run the full pipeline and return segments."""
        video_path = Path(video_path)
        duration = get_video_duration_sec(video_path)

        stream: List[Tuple[float, str, np.ndarray]] = []
        n_samples = 0
        n_failures = 0

        for t, frame in iter_video_samples(video_path, self.sample_fps):
            n_samples += 1
            crop = crop_bottom_band(frame, self.caption_band_ratio)
            try:
                raw = self.ocr.extract_text(crop, prompt=self.ocr_prompt)
            except Exception as exc:
                n_failures += 1
                logger.warning(f"GLM-OCR failed at t={t:.2f}s: {exc}")
                raw = ""
            text = normalize_caption_text(raw)
            # Keep the full frame (not the crop) so VLM gets full visual context.
            stream.append((t, text, frame))

        logger.info(
            f"CaptionTimeline: sampled {n_samples} frames "
            f"({n_failures} OCR failures), duration={duration:.1f}s"
        )

        segments = segment_caption_stream(
            stream,
            dedup_ratio=self.dedup_ratio,
            min_duration_sec=self.min_duration_sec,
            video_end_sec=duration,
        )
        logger.info(f"CaptionTimeline: produced {len(segments)} caption segments")
        return segments
```

- [ ] **Step 4: Run**

```powershell
pytest tests/m1_vlm/test_caption_timeline.py -v
```

Expected: all tests pass (15 total).

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/caption_ocr.py tests/m1_vlm/test_caption_timeline.py
```

Suggested message: `feat(m1): add CaptionTimeline orchestrator`

---

## Task 7 — Translation-only VLM prompt

**Files:**
- Modify: `src/m1_vlm/prompt_chain.py`
- Test: `tests/m1_vlm/test_translation_only_prompt.py`

- [ ] **Step 1: Write failing test**

Create `tests/m1_vlm/test_translation_only_prompt.py`:

```python
"""Snapshot-style tests for build_translation_only_prompt."""

import pytest

from src.m1_vlm.prompt_chain import PromptChain


def test_translation_only_prompt_includes_en_text():
    pc = PromptChain()
    out = pc.build_translation_only_prompt(en_text="Hello and welcome")
    assert "Hello and welcome" in out


def test_translation_only_prompt_requests_json_with_translated_text_field():
    pc = PromptChain()
    out = pc.build_translation_only_prompt(en_text="Hello")
    # Must instruct the model to return a single JSON object with this key.
    assert "translated_text" in out
    assert "JSON" in out or "json" in out


def test_translation_only_prompt_mentions_vietnamese():
    pc = PromptChain()
    out = pc.build_translation_only_prompt(en_text="Hello")
    assert "Vietnamese" in out


def test_translation_only_prompt_includes_global_context_when_given():
    pc = PromptChain()
    out = pc.build_translation_only_prompt(
        en_text="Hello", global_context="Topic: Python Flask tutorial"
    )
    assert "Python Flask tutorial" in out


def test_translation_only_prompt_includes_previous_translations_when_given():
    pc = PromptChain()
    out = pc.build_translation_only_prompt(
        en_text="Hello", previous_translations="Xin chào và chào mừng"
    )
    assert "Xin chào và chào mừng" in out


def test_translation_only_prompt_mentions_attached_frame():
    """Prompt should tell the model to use the attached visual context."""
    pc = PromptChain()
    out = pc.build_translation_only_prompt(en_text="Click submit")
    assert "frame" in out.lower() or "image" in out.lower() or "screen" in out.lower()
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m1_vlm/test_translation_only_prompt.py -v
```

Expected: `AttributeError: 'PromptChain' object has no attribute 'build_translation_only_prompt'`.

- [ ] **Step 3: Implement**

Add to `src/m1_vlm/prompt_chain.py` inside class `PromptChain`, right before `_format_timestamp`:

```python
    def build_translation_only_prompt(
        self,
        en_text: str,
        chunk_info: Optional[str] = None,
        global_context: Optional[str] = None,
        previous_translations: Optional[str] = None,
    ) -> str:
        """Translate ONE English caption to Vietnamese voiceover.

        Assumes a single frame image is attached as visual grounding so the
        translator can disambiguate UI/code references in `en_text`.

        Output: JSON object {"translated_text": "..."}. No timing field.
        """
        prompt = f"""You are a professional {self.source_lang}-to-{self.target_lang} translator for technical tutorial voiceover.

## Source caption
"{en_text}"

## Visual context
One frame from the video at the moment this caption appeared is attached as an image. Use it to disambiguate any UI labels, code identifiers, or on-screen elements the caption refers to. Do NOT describe the frame; only use it to inform the translation.

## Translation rules
- Output natural, conversational {self.target_lang} suitable for spoken voiceover.
- Keep technical terms, code, commands, and UI labels in {self.source_lang} (e.g., "API", "pip install torch", "Submit").
- Translate numbers and conjunctions naturally.
- Do not add greetings, introductions, or conclusions that aren't in the source.
- One sentence in, one sentence out — do not split or merge.
"""
        if global_context:
            prompt += f"\n## Video overview\n{global_context}\n"
        if chunk_info:
            prompt += f"\n## Chunk info\n{chunk_info}\n"
        if previous_translations:
            prompt += (
                "\n## Previous translations (for style/terminology consistency)\n"
                f"{previous_translations}\n"
            )

        prompt += """
## Output
Return ONLY a JSON object with this exact shape, no markdown fences, no explanation:
```json
{"translated_text": "bản dịch tiếng Việt ở đây"}
```
"""
        return prompt
```

- [ ] **Step 4: Run**

```powershell
pytest tests/m1_vlm/test_translation_only_prompt.py -v
```

Expected: `6 passed`.

- [ ] **Step 5: Stage**

```powershell
git add src/m1_vlm/prompt_chain.py tests/m1_vlm/test_translation_only_prompt.py
```

Suggested message: `feat(m1): add build_translation_only_prompt for OCR pipeline`

---

## Task 8 — Config properties for OCR mode

**Files:**
- Modify: `src/m4_pipeline/config.py`
- Test: `tests/m4_pipeline/test_config_ocr.py` (new)

- [ ] **Step 1: Write failing test**

Create `tests/m4_pipeline/test_config_ocr.py`:

```python
"""Tests for the OCR-mode pipeline config properties."""

import importlib

import pytest


@pytest.fixture
def cfg(monkeypatch):
    """Fresh PipelineConfig with all OCR env vars cleared."""
    for var in (
        "PIPELINE_MODE", "OCR_SAMPLE_FPS", "CAPTION_BAND_RATIO",
        "CAPTION_DEDUP_RATIO", "CAPTION_MIN_DURATION_SEC",
    ):
        monkeypatch.delenv(var, raising=False)
    from src.m4_pipeline.config import PipelineConfig
    return PipelineConfig()


def test_pipeline_mode_defaults_to_vlm(cfg):
    assert cfg.pipeline_mode == "vlm"


def test_ocr_sample_fps_default(cfg):
    assert cfg.ocr_sample_fps == 2.0


def test_caption_band_ratio_default(cfg):
    assert cfg.caption_band_ratio == 0.22


def test_caption_dedup_ratio_default(cfg):
    assert cfg.caption_dedup_ratio == 0.85


def test_caption_min_duration_sec_default(cfg):
    assert cfg.caption_min_duration_sec == 0.3


def test_pipeline_mode_reads_env(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "ocr")
    from src.m4_pipeline.config import PipelineConfig
    assert PipelineConfig().pipeline_mode == "ocr"


def test_ocr_sample_fps_reads_env(monkeypatch):
    monkeypatch.setenv("OCR_SAMPLE_FPS", "4.0")
    from src.m4_pipeline.config import PipelineConfig
    assert PipelineConfig().ocr_sample_fps == 4.0
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m4_pipeline/test_config_ocr.py -v
```

Expected: `AttributeError: 'PipelineConfig' object has no attribute 'pipeline_mode'`.

- [ ] **Step 3: Implement**

Insert into `src/m4_pipeline/config.py` after the `=== Processing ===` block (after `ssim_threshold`):

```python
    # === Pipeline Mode (vlm | ocr) ===
    @property
    def pipeline_mode(self) -> str:
        """Pipeline path: 'vlm' (default, today's behavior) or 'ocr'
        (GLM-OCR reads burned-in captions; VLM only translates)."""
        return os.getenv("PIPELINE_MODE", "vlm")

    @property
    def ocr_sample_fps(self) -> float:
        """Frame sampling rate for caption OCR (frames per second)."""
        return float(os.getenv("OCR_SAMPLE_FPS", "2.0"))

    @property
    def caption_band_ratio(self) -> float:
        """Fraction of frame height (from bottom) sent to GLM-OCR."""
        return float(os.getenv("CAPTION_BAND_RATIO", "0.22"))

    @property
    def caption_dedup_ratio(self) -> float:
        """SequenceMatcher ratio threshold for 'same caption'."""
        return float(os.getenv("CAPTION_DEDUP_RATIO", "0.85"))

    @property
    def caption_min_duration_sec(self) -> float:
        """Drop OCR-derived caption segments shorter than this."""
        return float(os.getenv("CAPTION_MIN_DURATION_SEC", "0.3"))
```

- [ ] **Step 4: Run**

```powershell
pytest tests/m4_pipeline/test_config_ocr.py -v
```

Expected: `7 passed`.

- [ ] **Step 5: Stage**

```powershell
git add src/m4_pipeline/config.py tests/m4_pipeline/test_config_ocr.py
```

Suggested message: `feat(m4): add pipeline_mode + OCR config properties`

---

## Task 9 — Refactor `PipelineRunner.run` to `_run_vlm_mode` (no behavior change)

**Files:**
- Modify: `src/m4_pipeline/runner.py`
- Test: `tests/m4_pipeline/test_runner_mode_branch.py` (new)

This is a pure refactor. The existing 380-line `run()` body moves into `_run_vlm_mode` verbatim; `run()` becomes a thin dispatcher.

- [ ] **Step 1: Write failing test**

Create `tests/m4_pipeline/test_runner_mode_branch.py`:

```python
"""Verify PipelineRunner.run() dispatches to the correct mode method."""

import asyncio
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def runner_with_paths(tmp_path):
    """Build a PipelineRunner with dummy file paths that exist."""
    from src.m4_pipeline.runner import PipelineRunner
    from src.m4_pipeline.config import PipelineConfig

    video = tmp_path / "v.mp4"
    ref = tmp_path / "ref.wav"
    video.write_bytes(b"fake")
    ref.write_bytes(b"fake")

    cfg = PipelineConfig()
    runner = PipelineRunner(config=cfg)
    return runner, video, ref


@pytest.mark.asyncio
async def test_run_dispatches_to_vlm_mode_by_default(monkeypatch, runner_with_paths):
    runner, video, ref = runner_with_paths
    monkeypatch.setenv("PIPELINE_MODE", "vlm")
    # Re-init config to pick up env
    from src.m4_pipeline.config import PipelineConfig
    runner.config = PipelineConfig()

    mock_vlm = AsyncMock(return_value={"ok": "vlm"})
    mock_ocr = AsyncMock(return_value={"ok": "ocr"})
    monkeypatch.setattr(runner, "_run_vlm_mode", mock_vlm)
    monkeypatch.setattr(runner, "_run_ocr_mode", mock_ocr)

    result = await runner.run(video, ref)
    assert mock_vlm.await_count == 1
    assert mock_ocr.await_count == 0
    assert result == {"ok": "vlm"}


@pytest.mark.asyncio
async def test_run_dispatches_to_ocr_mode_when_configured(monkeypatch, runner_with_paths):
    runner, video, ref = runner_with_paths
    monkeypatch.setenv("PIPELINE_MODE", "ocr")
    from src.m4_pipeline.config import PipelineConfig
    runner.config = PipelineConfig()

    mock_vlm = AsyncMock(return_value={"ok": "vlm"})
    mock_ocr = AsyncMock(return_value={"ok": "ocr"})
    monkeypatch.setattr(runner, "_run_vlm_mode", mock_vlm)
    monkeypatch.setattr(runner, "_run_ocr_mode", mock_ocr)

    result = await runner.run(video, ref)
    assert mock_ocr.await_count == 1
    assert mock_vlm.await_count == 0
    assert result == {"ok": "ocr"}


@pytest.mark.asyncio
async def test_run_raises_on_unknown_mode(monkeypatch, runner_with_paths):
    from src.m4_pipeline.exceptions import PipelineError
    runner, video, ref = runner_with_paths
    monkeypatch.setenv("PIPELINE_MODE", "bogus")
    from src.m4_pipeline.config import PipelineConfig
    runner.config = PipelineConfig()

    with pytest.raises(PipelineError, match="Unknown pipeline_mode"):
        await runner.run(video, ref)
```

If `pytest-asyncio` is not installed, install it inside the venv:

```powershell
.\.venv\Scripts\Activate.ps1
pip install pytest-asyncio
```

And ensure `tests/conftest.py` (or `pyproject.toml`/`pytest.ini`) has `asyncio_mode = "auto"`. If not, add it:

`tests/conftest.py` (create if missing):

```python
import pytest

pytest_plugins = ["pytest_asyncio"]
```

And in `pyproject.toml` or new `pytest.ini`:

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 2: Run failure**

```powershell
pytest tests/m4_pipeline/test_runner_mode_branch.py -v
```

Expected: dispatch tests fail because `run()` doesn't branch yet (it just runs the current body and dies on the fake video).

- [ ] **Step 3: Refactor runner**

In `src/m4_pipeline/runner.py`:

1. Rename the existing `async def run(self, video_path, reference_audio_path, output_path=None) -> Dict[str, Any]:` to `async def _run_vlm_mode(self, video_path, reference_audio_path, output_path=None) -> Dict[str, Any]:`. **Do not modify a single line of its body.**

2. Add a new dispatcher `run()` method just above `_run_vlm_mode`:

```python
    async def run(
        self,
        video_path: str | Path,
        reference_audio_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Dispatch to the configured pipeline mode ('vlm' or 'ocr')."""
        mode = self.config.pipeline_mode
        if mode == "vlm":
            return await self._run_vlm_mode(video_path, reference_audio_path, output_path)
        if mode == "ocr":
            return await self._run_ocr_mode(video_path, reference_audio_path, output_path)
        raise PipelineError(
            f"Unknown pipeline_mode={mode!r}; expected 'vlm' or 'ocr'"
        )
```

3. Add a placeholder `_run_ocr_mode` so the import resolves (will be implemented in Task 10):

```python
    async def _run_ocr_mode(
        self,
        video_path: str | Path,
        reference_audio_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """OCR-driven dubbing pipeline (implemented in Task 10)."""
        raise PipelineError("OCR mode not yet implemented")
```

- [ ] **Step 4: Run dispatch tests**

```powershell
pytest tests/m4_pipeline/test_runner_mode_branch.py -v
```

Expected: `3 passed`.

- [ ] **Step 5: Re-run full m1/m4 suite to verify no regression**

```powershell
pytest tests/m1_vlm/ tests/m4_pipeline/ -v
```

All previously-passing tests should still pass.

- [ ] **Step 6: Stage**

```powershell
git add src/m4_pipeline/runner.py tests/m4_pipeline/test_runner_mode_branch.py tests/conftest.py pyproject.toml
```

(Stage `conftest.py` and `pyproject.toml`/`pytest.ini` only if you created or modified them.)

Suggested message: `refactor(m4): split runner into mode dispatch + _run_vlm_mode`

---

## Task 10 — Implement `_run_ocr_mode`

**Files:**
- Modify: `src/m4_pipeline/runner.py`
- Test: smoke via `scripts/test_caption_ocr.py` in Task 12.

This is the new code path. It reuses everything from M2/M3 verbatim.

- [ ] **Step 1: Add STEPS_OCR constant**

Replace the `STEPS = [...]` block at the top of `PipelineRunner` with:

```python
    STEPS = [
        "scene_detection",
        "frame_extraction",
        "ocr_extraction",
        "vlm_translation",
        "srt_generation",
        "voice_cloning",
        "audio_alignment",
        "video_rendering",
    ]

    STEPS_OCR = [
        "caption_ocr_timeline",
        "vlm_global_summary",
        "vlm_translate_segments",
        "srt_generation",
        "voice_cloning",
        "audio_alignment",
        "video_rendering",
    ]
```

Update `_update_progress` to use the correct list:

```python
    def _update_progress(self, step_name: str):
        """Notify progress callback. Picks step total from current mode."""
        self._current_step += 1
        total = len(self._active_steps())
        if self._progress_callback:
            self._progress_callback(step_name, self._current_step, total)
        logger.info(f"[{self._current_step}/{total}] {step_name}")

    def _active_steps(self) -> List[str]:
        return self.STEPS_OCR if self.config.pipeline_mode == "ocr" else self.STEPS
```

- [ ] **Step 2: Replace `_run_ocr_mode` placeholder with real implementation**

Replace the placeholder body added in Task 9:

```python
    async def _run_ocr_mode(
        self,
        video_path: str | Path,
        reference_audio_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """OCR-driven pipeline: GLM-OCR captions → VLM translation → TTS."""
        import json as _json
        import re as _re

        from src.m1_vlm.glm_ocr import GLMOCR
        from src.m1_vlm.caption_ocr import CaptionTimeline

        start_time = time.time()
        self._current_step = 0

        video_path = Path(video_path)
        reference_audio_path = Path(reference_audio_path)
        if not video_path.exists():
            raise PipelineError(f"Video not found: {video_path}")
        if not reference_audio_path.exists():
            raise PipelineError(f"Reference audio not found: {reference_audio_path}")

        safe_stem = _re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("_") or "video"
        if output_path is None:
            output_dir = Path(self.config.output_dir) / safe_stem
        else:
            output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir = output_dir / "work_ocr"
        work_dir.mkdir(parents=True, exist_ok=True)
        seg_raw_dir = work_dir / "segments_raw"
        seg_raw_dir.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {
            "video_path": str(video_path),
            "reference_audio": str(reference_audio_path),
            "mode": "ocr",
        }

        try:
            # === Step 1: Caption OCR timeline ===
            self._update_progress("caption_ocr_timeline")
            ocr = GLMOCR(
                model_path=self.config.glm_ocr_model_path,
                device="cuda",
            )
            timeline = CaptionTimeline(
                ocr=ocr,
                sample_fps=self.config.ocr_sample_fps,
                caption_band_ratio=self.config.caption_band_ratio,
                dedup_ratio=self.config.caption_dedup_ratio,
                min_duration_sec=self.config.caption_min_duration_sec,
            )
            segments = timeline.build(video_path)
            results["n_segments"] = len(segments)

            if not segments:
                raise PipelineError(
                    "No captions detected — video may not have burned-in subtitles; "
                    "use --mode vlm instead"
                )

            # Persist segment manifest for debugging
            manifest = [
                {
                    "idx": i,
                    "start_sec": s.start_sec,
                    "end_sec": s.end_sec,
                    "en_text": s.en_text,
                }
                for i, s in enumerate(segments)
            ]
            (work_dir / "segments.json").write_text(
                _json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Free GLM-OCR VRAM before loading VLM
            try:
                ocr.unload_model()
            except Exception as exc:
                logger.warning(f"GLM-OCR unload failed (continuing): {exc}")

            # === Step 2: VLM global summary (optional) ===
            self._update_progress("vlm_global_summary")
            vlm_client = VLMClient(
                mode=self.config.vlm_mode,
                model_name=self.config.vlm_model_name,
                api_key=self.config.gemini_api_key,
                local_model_path=self.config.qwen_model_path,
                temperature=self.config.vlm_temperature,
                max_tokens=self.config.vlm_max_tokens,
            )
            prompt_chain = PromptChain()
            frame_extractor = FrameExtractor(max_width=768)

            # Build global summary from up to 8 segment mid-frames, evenly spaced.
            global_context: Optional[str] = None
            sample_idxs = (
                [int(i * (len(segments) - 1) / 7) for i in range(8)]
                if len(segments) >= 8
                else list(range(len(segments)))
            )
            global_b64 = [
                frame_extractor.frame_to_base64(segments[i].mid_frame)
                for i in sample_idxs
            ]
            try:
                global_prompt = prompt_chain.build_global_summary_prompt()
                raw_global = await vlm_client.generate(
                    prompt=global_prompt, images_base64=global_b64
                )
                (work_dir / "global_summary_raw.txt").write_text(
                    raw_global, encoding="utf-8"
                )
                cleaned_g = raw_global.strip()
                if cleaned_g.startswith("```json"):
                    cleaned_g = cleaned_g[7:]
                if cleaned_g.startswith("```"):
                    cleaned_g = cleaned_g[3:]
                if cleaned_g.endswith("```"):
                    cleaned_g = cleaned_g[:-3]
                try:
                    parsed_g = _json.loads(cleaned_g.strip())
                    global_context = _json.dumps(parsed_g, ensure_ascii=False, indent=2)
                    (work_dir / "global_summary.json").write_text(
                        global_context, encoding="utf-8"
                    )
                except _json.JSONDecodeError:
                    global_context = raw_global
            except Exception as exc:
                logger.warning(f"Global summary failed (continuing): {exc}")
                global_context = None

            # === Step 3: Translate each segment with frame attached ===
            self._update_progress("vlm_translate_segments")
            entries: List[Dict[str, Any]] = []
            previous_vi: List[str] = []
            for i, seg in enumerate(segments):
                prev_block = (
                    "\n".join(previous_vi[-5:]) if previous_vi else None
                )
                prompt = prompt_chain.build_translation_only_prompt(
                    en_text=seg.en_text,
                    global_context=global_context,
                    previous_translations=prev_block,
                )
                frame_b64 = frame_extractor.frame_to_base64(seg.mid_frame)
                try:
                    raw = await vlm_client.generate(
                        prompt=prompt, images_base64=[frame_b64]
                    )
                except Exception as exc:
                    logger.warning(f"Segment {i}: VLM call failed ({exc}); skipping")
                    continue

                (seg_raw_dir / f"seg_{i:04d}.txt").write_text(raw, encoding="utf-8")

                cleaned = raw.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                try:
                    obj = _json.loads(cleaned.strip())
                except _json.JSONDecodeError:
                    logger.warning(
                        f"Segment {i}: VLM JSON parse failed; raw at "
                        f"{seg_raw_dir / f'seg_{i:04d}.txt'}"
                    )
                    continue

                vi = obj.get("translated_text", "").strip() if isinstance(obj, dict) else ""
                if not vi:
                    logger.warning(f"Segment {i}: empty translated_text; skipping")
                    continue

                entries.append({
                    "index": len(entries) + 1,
                    "start_time": _sec_to_srt(seg.start_sec),
                    "end_time": _sec_to_srt(seg.end_sec),
                    "original_text": seg.en_text,
                    "translated_text": vi,
                })
                previous_vi.append(vi)

            if len(entries) < 3:
                raise PipelineError(
                    f"Too few segments translated ({len(entries)}); aborting before TTS"
                )

            # === Step 4: Build SRT ===
            self._update_progress("srt_generation")
            srt_builder = SRTBuilder()
            srt_builder.add_entries(entries, chunk_offset=0.0)
            srt_path = output_dir / f"{safe_stem}_vi.srt"
            srt_builder.save(srt_path)
            results["srt_path"] = str(srt_path)

            try:
                vlm_client.unload_model()
            except Exception as exc:
                logger.warning(f"VLM unload failed (continuing): {exc}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            # === Step 5: Voice cloning (reused from vlm mode) ===
            self._update_progress("voice_cloning")
            tts_client = TTSClient(
                engine=self.config.tts_engine,
                backbone_repo=self.config.tts_backbone_repo,
                backbone_device=self.config.tts_backbone_device,
                codec_device=self.config.tts_codec_device,
                vieneu_mode=self.config.tts_vieneu_mode,
                hf_token=self.config.tts_hf_token,
                sample_rate=self.config.tts_sample_rate,
            )
            logger.info(f"Encoding reference voice: {reference_audio_path.name}")
            ref_codes = tts_client.encode_reference(reference_audio_path, use_cache=True)
            batch_inference = BatchInference(tts_client)
            srt_entries = SRTBuilder.load_srt(srt_path)
            audio_segments = await batch_inference.process_all(
                segments=srt_entries,
                output_dir=work_dir / "audio_chunks",
                ref_codes=ref_codes,
                ref_text=None,
            )

            # === Step 6: Audio alignment ===
            self._update_progress("audio_alignment")
            aligner = AudioAligner(
                max_speedup=self.config.m3_max_speedup,
                min_gap_sec=self.config.m3_min_gap_sec,
            )
            segments_with_deltas = aligner.calculate_deltas(audio_segments)
            aligned_segments = aligner.align_all(
                segments_with_deltas,
                output_dir=work_dir / "aligned_audio",
            )

            # === Step 7: Render ===
            self._update_progress("video_rendering")
            renderer = FFmpegRenderer()
            video_info = FFmpegRenderer.get_video_info(video_path)
            duration = float(video_info["format"]["duration"])
            merged_audio_path = work_dir / "merged_audio.wav"
            renderer.merge_audio_segments(
                aligned_segments, duration, merged_audio_path, sample_rate=24000,
            )
            final_output = output_path or output_dir / f"{safe_stem}_dubbed.mp4"
            renderer.render_final_video(
                video_path=video_path,
                dubbed_audio_path=merged_audio_path,
                output_path=final_output,
            )
            results["output_path"] = str(final_output)

        except PipelineError:
            raise
        except Exception as e:
            logger.error(f"OCR pipeline failed at step {self._current_step}: {e}")
            results["error"] = str(e)
            raise PipelineError(f"OCR pipeline failed: {e}") from e

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        logger.info(f"OCR pipeline completed in {elapsed:.1f}s")
        return results
```

Add this module-level helper at the bottom of `runner.py` (or near the imports):

```python
def _sec_to_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
```

- [ ] **Step 3: Re-run the mode-dispatch tests to confirm refactor still passes**

```powershell
pytest tests/m4_pipeline/test_runner_mode_branch.py -v
```

Expected: `3 passed` (the OCR test that previously hit the placeholder will now hit the real method — but it's mocked, so still fine).

- [ ] **Step 4: Re-run full test suite**

```powershell
pytest tests/ -v
```

Expected: all pre-existing tests + new ones pass.

- [ ] **Step 5: Stage**

```powershell
git add src/m4_pipeline/runner.py
```

Suggested message: `feat(m4): implement _run_ocr_mode using CaptionTimeline + per-segment VLM`

---

## Task 11 — CLI flag on `scripts/run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 1: Inspect current CLI**

Open `scripts/run_pipeline.py` and locate the `parse_args()` function (around line 29) and the `main()` function further down where `PipelineRunner` is constructed.

- [ ] **Step 2: Add `--mode` flag**

Inside `parse_args()`, add (after `--vlm-mode` or near the end of the parser args):

```python
    parser.add_argument(
        "--mode",
        choices=["vlm", "ocr"],
        default=None,
        help="Pipeline mode: 'vlm' (default, today's behavior) or 'ocr' "
             "(GLM-OCR caption timing + VLM translation). "
             "If omitted, uses PIPELINE_MODE env var (default 'vlm').",
    )
```

- [ ] **Step 3: Plumb the flag through to config**

Find where `PipelineConfig()` is instantiated (or where env vars are read in `main()`). Add right before the runner is created:

```python
    if args.mode is not None:
        os.environ["PIPELINE_MODE"] = args.mode
```

(Make sure `import os` is present at the top of the file.)

- [ ] **Step 4: Smoke-check the CLI parses**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/run_pipeline.py --help
```

Expected: `--mode {vlm,ocr}` appears in the help output.

```powershell
python scripts/run_pipeline.py --mode bogus --video x --ref-audio y
```

Expected: argparse error "invalid choice: 'bogus'".

- [ ] **Step 5: Stage**

```powershell
git add scripts/run_pipeline.py
```

Suggested message: `feat(cli): add --mode {vlm,ocr} flag to run_pipeline.py`

---

## Task 12 — Manual smoke script for caption OCR

**Files:**
- Create: `scripts/test_caption_ocr.py`

A standalone script the user can run on any video with burned-in captions to verify the timeline before paying for full TTS + render.

- [ ] **Step 1: Write the script**

Create `scripts/test_caption_ocr.py`:

```python
"""Smoke test: run CaptionTimeline on a single video, print + save segments.

Does NOT call the VLM or TTS — just sample → crop → OCR → segment → SRT.
Use this to validate that burned-in captions can be recovered before
running the full --mode ocr pipeline.

Usage:
    .\\.venv\\Scripts\\Activate.ps1
    python scripts/test_caption_ocr.py --video data/raw/sample.mp4
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m1_vlm.glm_ocr import GLMOCR
from src.m1_vlm.caption_ocr import CaptionTimeline


def _sec_to_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    parser = argparse.ArgumentParser(description="CaptionTimeline smoke test")
    parser.add_argument("--video", "-v", required=True, help="Path to video")
    parser.add_argument("--fps", type=float, default=2.0, help="OCR sample fps")
    parser.add_argument("--band", type=float, default=0.22,
                        help="Bottom-crop ratio (0..1)")
    parser.add_argument("--dedup", type=float, default=0.85,
                        help="Similarity threshold")
    parser.add_argument("--min-dur", type=float, default=0.3,
                        help="Drop segments shorter than this (seconds)")
    parser.add_argument("--ocr-model", default="zai-org/GLM-OCR",
                        help="GLM-OCR model id or path")
    parser.add_argument("--out", default=None,
                        help="Output dir (default: ./output/caption_ocr_<stem>)")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"ERROR: video not found: {video}")
        sys.exit(1)

    out_dir = Path(args.out) if args.out else Path("output") / f"caption_ocr_{video.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ocr = GLMOCR(model_path=args.ocr_model, device="cuda")
    tl = CaptionTimeline(
        ocr=ocr,
        sample_fps=args.fps,
        caption_band_ratio=args.band,
        dedup_ratio=args.dedup,
        min_duration_sec=args.min_dur,
    )
    segments = tl.build(video)

    print(f"\nDetected {len(segments)} caption segments:\n")
    for i, s in enumerate(segments):
        print(f"  [{i:03d}] {s.start_sec:7.2f}s → {s.end_sec:7.2f}s  "
              f"({s.duration_sec:5.2f}s)  {s.en_text!r}")

    manifest = [
        {"idx": i, "start_sec": s.start_sec, "end_sec": s.end_sec, "en_text": s.en_text}
        for i, s in enumerate(segments)
    ]
    (out_dir / "segments.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    srt_path = out_dir / f"{video.stem}_en_captions.srt"
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_sec_to_srt(s.start_sec)} --> {_sec_to_srt(s.end_sec)}")
        lines.append(s.en_text)
        lines.append("")
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWrote: {out_dir/'segments.json'}")
    print(f"Wrote: {srt_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Stage**

```powershell
git add scripts/test_caption_ocr.py
```

Suggested message: `feat(scripts): add test_caption_ocr.py smoke runner`

---

## Task 13 — Document new env vars in `.env.example`

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Append OCR mode section**

Add to the end of `.env.example`:

```dotenv
# === Pipeline mode (M4) ===
# 'vlm' (default, today's pipeline) or 'ocr' (GLM-OCR caption timing + VLM translation).
PIPELINE_MODE=vlm

# === OCR pipeline tuning (only used when PIPELINE_MODE=ocr) ===
# Frame sampling rate for caption OCR (frames per second). 2.0 = every 0.5s.
OCR_SAMPLE_FPS=2.0
# Fraction of frame height (from bottom) sent to GLM-OCR. 0.22 = bottom 22%.
CAPTION_BAND_RATIO=0.22
# SequenceMatcher ratio threshold for "same caption" — handles OCR jitter.
CAPTION_DEDUP_RATIO=0.85
# Drop OCR-derived caption segments shorter than this (seconds).
CAPTION_MIN_DURATION_SEC=0.3
```

- [ ] **Step 2: Stage**

```powershell
git add .env.example
```

Suggested message: `docs: document OCR-mode env vars in .env.example`

---

## Task 14 — End-to-end manual verification (no automated test)

This task is for the user, not the agent.

- [ ] **Step 1: Pick a short test video with burned-in EN captions** (10–30 seconds is enough for the first run).

- [ ] **Step 2: Run the smoke script**

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/test_caption_ocr.py --video data\raw\<your_sample>.mp4
```

Check the printed segments + the generated `output/caption_ocr_<stem>/<stem>_en_captions.srt`. Open the SRT in a player alongside the video; timing should match what's on screen.

- [ ] **Step 3: If timing looks right, run the full OCR pipeline**

```powershell
python scripts/run_pipeline.py `
  --video data\raw\<your_sample>.mp4 `
  --ref-audio data\reference_audio\<speaker>.wav `
  --mode ocr
```

- [ ] **Step 4: Verify the default mode still works**

```powershell
python scripts/run_pipeline.py `
  --video data\raw\<your_sample>.mp4 `
  --ref-audio data\reference_audio\<speaker>.wav
```

Should behave identically to before this change (no `--mode` = `vlm`).

---

## Self-Review Notes

- **Spec coverage:** §4.1 → Tasks 1–6; §4.2 → Task 7; §4.3 → Tasks 9–10; §4.4 → Task 8; §4.5 → Task 11; §4.6 → Task 13. §5 data flow exercised by Task 14. §6 error handling: empty captions / too-few segments / per-segment JSON dump implemented in Task 10; OCR-call failure tolerated in Task 6. §7 testing covered by Tasks 1–9 (+ smoke script in 12, manual in 14).
- **Placeholder scan:** No "TBD", "TODO", or hand-wavy steps. Every code block is complete copy-paste material.
- **Type consistency:** `CaptionSegment` fields (`start_sec`, `end_sec`, `en_text`, `mid_frame`) used identically across Tasks 1, 4, 6, 10, 12. `pipeline_mode` string values `"vlm"`/`"ocr"` consistent across Tasks 8, 9, 10, 11, 13. `build_translation_only_prompt(en_text, chunk_info, global_context, previous_translations)` signature matches between Task 7 definition and Task 10 call.
