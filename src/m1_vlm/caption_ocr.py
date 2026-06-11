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


import re as _re


# Common GLM-OCR meta-prefix hallucinations to strip before downstream use.
# These are framing artifacts the model adds when uncertain, NOT real caption text.
_OCR_META_PREFIXES = (
    "the text is:",
    "the text in the image is:",
    "the subtitle text is:",
    "subtitle text:",
    "subtitle:",
    "caption:",
    "the caption is:",
    "子标题：",
    "子标题:",
)

# Whole-string OCR outputs that mean "no readable text" — treated as empty.
_OCR_NULL_SENTINELS = frozenset(
    s.lower()
    for s in (
        "---",
        "--",
        "...",
        "n/a",
        "none",
        "no text",
        "no text visible",
        "no text in the image",
        "no text in image",
        "no readable text",
        "the subtitle text is not provided in the image",
        "the subtitle text is not provided in the image.",
        "subtitle text is not provided",
        "the text is not provided",
        "text not provided",
        "no caption",
        "no caption visible",
    )
)


def _strip_ocr_meta_prefix(text: str) -> str:
    """Iteratively strip common GLM-OCR meta-prefixes from the start of `text`.

    Also trims matching surrounding quotes so the cleaned text is the bare caption.
    """
    s = text.lstrip().lstrip('"').lstrip("'")
    changed = True
    while changed:
        changed = False
        lower = s.lower()
        for prefix in _OCR_META_PREFIXES:
            if lower.startswith(prefix):
                s = s[len(prefix) :].lstrip().lstrip('"').lstrip("'")
                changed = True
                break
    # Tidy trailing quotes/whitespace introduced by the OCR's "X" quoting style.
    s = s.rstrip().rstrip('"').rstrip("'").rstrip()
    return s


def normalize_caption_text(text: str) -> str:
    """Strip OCR meta-prefixes, lowercase, collapse whitespace.
    Returns empty string for blank input or for GLM-OCR null-sentinel outputs
    like '---' or 'the subtitle text is not provided in the image'."""
    if text is None:
        return ""
    cleaned = _strip_ocr_meta_prefix(text)
    collapsed = _re.sub(r"\s+", " ", cleaned).strip().lower()
    if collapsed in _OCR_NULL_SENTINELS:
        return ""
    return collapsed


def is_ocr_repetition_artifact(
    text: str,
    min_length: int = 120,
    window_chars: int = 12,
    min_repeats: int = 6,
) -> bool:
    """Detect GLM-OCR repetition-loop hallucinations.

    When the cropped input has no readable text, GLM-OCR sometimes outputs
    the same short phrase hundreds of times in a row (e.g.,
    "the text is: the text is: the text is: ..."). These artifacts pollute
    the caption timeline and look superficially long. Heuristic: if any
    `window_chars`-length sliding window appears at least `min_repeats`
    times in a sample of the text, treat as an artifact.

    Returns False for short or non-repetitive text (the common case).
    """
    if not text or len(text) < min_length:
        return False
    # Look at a bounded prefix to keep this O(1) per call regardless of length.
    sample = text[:1200]
    if len(sample) <= window_chars:
        return False
    counts: dict[str, int] = {}
    best = 0
    for i in range(len(sample) - window_chars):
        key = sample[i : i + window_chars]
        c = counts.get(key, 0) + 1
        counts[key] = c
        if c > best:
            best = c
            if best >= min_repeats:
                return True
    return False


from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

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


import cv2
from pathlib import Path
from typing import Iterator


def iter_video_samples(
    video_path: str | Path,
    sample_fps: float,
) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, frame_bgr) at the requested rate.

    Reads frames sequentially and yields every Nth frame to match the
    requested sample_fps. Sequential decode avoids the keyframe-snap
    problem of CAP_PROP_POS_MSEC seeking where multiple timestamps can
    return the same keyframe, causing missed captions.

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

        frame_interval = max(1, int(round(fps / sample_fps)))
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % frame_interval == 0:
                t = frame_idx / fps
                yield t, frame
            frame_idx += 1
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


from loguru import logger


def _dhash(gray: np.ndarray) -> int:
    """Compute a 64-bit difference hash from a grayscale image.

    Resize to 9x8, compare each pixel to its right neighbor → 64 bits.
    """
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]  # 8x8 bool
    return int(np.packbits(diff.flatten()).view(np.uint64)[0])


def _dhash_distance(h1: int, h2: int) -> int:
    """Hamming distance between two 64-bit hashes."""
    return bin(h1 ^ h2).count("1")


class CaptionTimeline:
    """Build a list of caption segments by densely OCR-ing burned-in subtitles.

    Pipeline: sample @ sample_fps → crop bottom band → dHash skip (if unchanged)
    → GLM-OCR → normalize → group consecutive identical captions → drop short blips.
    """

    def __init__(
        self,
        ocr,  # GLMOCR (duck-typed: must expose .extract_text(frame))
        sample_fps: float = 2.0,
        caption_band_ratio: float = 0.22,
        dedup_ratio: float = 0.85,
        min_duration_sec: float = 0.3,
        ocr_prompt: str = "Read the subtitle text only. Return only the text.",
        dhash_max_distance: int = 6,
    ):
        self.ocr = ocr
        self.sample_fps = sample_fps
        self.caption_band_ratio = caption_band_ratio
        self.dedup_ratio = dedup_ratio
        self.min_duration_sec = min_duration_sec
        self.ocr_prompt = ocr_prompt
        self.dhash_max_distance = dhash_max_distance

    def build(self, video_path: str | Path) -> List[CaptionSegment]:
        """Run the full pipeline and return segments."""
        video_path = Path(video_path)
        duration = get_video_duration_sec(video_path)

        stream: List[Tuple[float, str, np.ndarray]] = []
        n_samples = 0
        n_failures = 0
        n_hallucinations = 0
        n_skip = 0

        prev_hash: int | None = None
        prev_text = ""

        for t, frame in iter_video_samples(video_path, self.sample_fps):
            n_samples += 1
            crop = crop_bottom_band(frame, self.caption_band_ratio)
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            h = _dhash(crop_gray)

            if prev_hash is not None and _dhash_distance(prev_hash, h) <= self.dhash_max_distance:
                n_skip += 1
                stream.append((t, prev_text, frame))
                continue

            prev_hash = h

            try:
                raw = self.ocr.extract_text(crop, prompt=self.ocr_prompt)
            except Exception as exc:
                n_failures += 1
                logger.warning(f"GLM-OCR failed at t={t:.2f}s: {exc}")
                raw = ""
            if is_ocr_repetition_artifact(raw):
                n_hallucinations += 1
                raw = ""
            text = normalize_caption_text(raw)
            prev_text = text
            stream.append((t, text, frame))

        logger.info(
            f"CaptionTimeline: sampled {n_samples} frames, "
            f"OCR called {n_samples - n_skip} "
            f"(skipped {n_skip} by dHash, max_dist={self.dhash_max_distance}), "
            f"{n_failures} OCR failures, {n_hallucinations} repetition artifacts, "
            f"duration={duration:.1f}s"
        )

        segments = segment_caption_stream(
            stream,
            dedup_ratio=self.dedup_ratio,
            min_duration_sec=self.min_duration_sec,
            video_end_sec=duration,
        )
        logger.info(f"CaptionTimeline: produced {len(segments)} caption segments")
        return segments


def _is_prefix_animation(a: str, b: str, min_short_chars: int = 12) -> bool:
    """True iff one string is essentially a prefix of the other.

    Used to collapse animated/typewriter captions where consecutive frames
    show progressively longer versions of the same sentence
    (e.g. "the generation time" → "the generation time depends on the length").
    """
    if len(a) < min_short_chars or len(b) < min_short_chars:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    prefix_of_long = long[: len(short)]
    return SequenceMatcher(None, short, prefix_of_long).ratio() >= 0.85


def merge_short_segments(
    segments: List[CaptionSegment],
    max_gap_sec: float = 0.3,
    max_combined_chars: int = 100,
    animation_gap_multiplier: float = 4.0,
) -> List[CaptionSegment]:
    """Collapse adjacent narration fragments into smoother units.

    Two strategies, checked in order:

    1. **Animation overlap** (looser gap allowance) — if one segment's text
       is essentially a prefix of the other (animated typewriter caption),
       keep the longer text and widen the time span. The shorter, partial
       version is discarded. Allowed gap up to
       ``max_gap_sec * animation_gap_multiplier`` so that two separate
       cycles of the same animation collapse into one entry.
    2. **Tight fragmentation** (tight gap) — otherwise, if joining the two
       texts with a space stays within ``max_combined_chars`` AND the gap
       is at most ``max_gap_sec``, concatenate them.

    Beyond ``max_combined_chars``, segments are left separate even if the
    gap is small. The merged segment inherits the earlier segment's
    ``mid_frame`` so visual context tracks the start of the phrase.
    """
    if not segments:
        return []

    animation_gap_cap = max_gap_sec * animation_gap_multiplier
    identical_gap_cap = max_gap_sec * 25  # ~7.5s at default — covers animation cycles
    merged: List[CaptionSegment] = [segments[0]]
    for nxt in segments[1:]:
        last = merged[-1]
        gap = nxt.start_sec - last.end_sec

        # Strategy 1: identical/near-identical text — collapse repeated captions
        # from animation cycles (same text appears, fades, reappears).
        if gap <= identical_gap_cap and _same_caption(last.en_text, nxt.en_text, 0.85):
            longer = last if len(last.en_text) >= len(nxt.en_text) else nxt
            merged[-1] = CaptionSegment(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, nxt.end_sec),
                en_text=longer.en_text,
                mid_frame=last.mid_frame,
            )
            continue

        # Strategy 2: animation/typewriter overlap — keep the longer text.
        if gap <= animation_gap_cap and _is_prefix_animation(last.en_text, nxt.en_text):
            longer = last if len(last.en_text) >= len(nxt.en_text) else nxt
            merged[-1] = CaptionSegment(
                start_sec=min(last.start_sec, nxt.start_sec),
                end_sec=max(last.end_sec, nxt.end_sec),
                en_text=longer.en_text,
                mid_frame=last.mid_frame,
            )
            continue

        if gap > max_gap_sec:
            merged.append(nxt)
            continue

        # Strategy 3: tight fragmentation — concatenate within budget.
        combined_len = len(last.en_text) + 1 + len(nxt.en_text)
        if combined_len <= max_combined_chars:
            merged[-1] = CaptionSegment(
                start_sec=last.start_sec,
                end_sec=nxt.end_sec,
                en_text=f"{last.en_text} {nxt.en_text}".strip(),
                mid_frame=last.mid_frame,
            )
        else:
            merged.append(nxt)
    return merged


def extend_end_times(
    segments: List[CaptionSegment],
    extend_sec: float = 0.5,
    min_gap_sec: float = 0.1,
    video_duration_sec: Optional[float] = None,
) -> List[CaptionSegment]:
    """Lengthen each entry's end_time to give TTS breathing room.

    For every segment except the last:
        new_end = min(seg.end_sec + extend_sec, next.start_sec - min_gap_sec)
        (never reduce — only extend)

    For the last segment:
        new_end = min(seg.end_sec + extend_sec, video_duration_sec)
        (or seg.end_sec + extend_sec if duration unknown)
    """
    if not segments:
        return []

    out: List[CaptionSegment] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        if i < n - 1:
            ceiling = segments[i + 1].start_sec - min_gap_sec
            new_end = min(seg.end_sec + extend_sec, ceiling)
        elif video_duration_sec is not None:
            new_end = min(seg.end_sec + extend_sec, video_duration_sec)
        else:
            new_end = seg.end_sec + extend_sec
        # Never shrink — keep original if the cap would push end backward.
        new_end = max(new_end, seg.end_sec)
        out.append(
            CaptionSegment(
                start_sec=seg.start_sec,
                end_sec=new_end,
                en_text=seg.en_text,
                mid_frame=seg.mid_frame,
            )
        )
    return out
