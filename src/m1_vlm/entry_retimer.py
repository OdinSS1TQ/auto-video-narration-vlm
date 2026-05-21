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
