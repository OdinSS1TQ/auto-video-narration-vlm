"""
Validator — Heuristic checks and retry logic for VLM output.

Validates subtitle entries for format correctness, overlap, and quality.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class SubtitleValidator:
    """Validate and fix VLM-generated subtitle data."""

    # SRT timestamp pattern: HH:MM:SS,mmm
    TIMESTAMP_PATTERN = re.compile(
        r"^\d{2}:\d{2}:\d{2},\d{3}$"
    )

    def __init__(
        self,
        min_duration: float = 0.5,
        max_duration: float = 10.0,
        min_gap: float = 0.05,
    ):
        """
        Args:
            min_duration: Minimum subtitle duration in seconds.
            max_duration: Maximum subtitle duration in seconds.
            min_gap: Minimum gap between subtitles in seconds.
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_gap = min_gap

    def validate_entry(self, entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single subtitle entry.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues = []

        # Check required fields
        required = ["index", "start_time", "end_time", "translated_text"]
        for field in required:
            if field not in entry:
                issues.append(f"Missing required field: {field}")

        if issues:
            return False, issues

        # Check timestamp format
        if not self.TIMESTAMP_PATTERN.match(entry["start_time"]):
            issues.append(f"Invalid start_time format: {entry['start_time']}")
        if not self.TIMESTAMP_PATTERN.match(entry["end_time"]):
            issues.append(f"Invalid end_time format: {entry['end_time']}")

        # Check duration
        if not issues:
            start_sec = self._timestamp_to_seconds(entry["start_time"])
            end_sec = self._timestamp_to_seconds(entry["end_time"])
            duration = end_sec - start_sec

            if duration <= 0:
                issues.append(f"End time must be after start time")
            elif duration < self.min_duration:
                issues.append(f"Duration too short: {duration:.3f}s")
            elif duration > self.max_duration:
                issues.append(f"Duration too long: {duration:.3f}s")

        # Check translated text is not empty
        if not entry.get("translated_text", "").strip():
            issues.append("Empty translated_text")

        return len(issues) == 0, issues

    def validate_sequence(self, entries: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate a sequence of subtitle entries.

        Returns:
            Tuple of (all_valid, list_of_issues).
        """
        all_issues = []

        for i, entry in enumerate(entries):
            valid, issues = self.validate_entry(entry)
            if not valid:
                all_issues.extend([f"Entry {i}: {issue}" for issue in issues])

        # Check for overlaps
        for i in range(1, len(entries)):
            try:
                prev_end = self._timestamp_to_seconds(entries[i - 1]["end_time"])
                curr_start = self._timestamp_to_seconds(entries[i]["start_time"])
                gap = curr_start - prev_end

                if gap < 0:
                    all_issues.append(
                        f"Overlap between entries {i-1} and {i}: {abs(gap):.3f}s"
                    )
            except (KeyError, ValueError):
                pass

        return len(all_issues) == 0, all_issues

    def fix_overlaps(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fix overlapping subtitles by adjusting timestamps.

        Args:
            entries: List of subtitle entries.

        Returns:
            Fixed list of subtitle entries.
        """
        if len(entries) <= 1:
            return entries

        fixed = [entries[0].copy()]

        for i in range(1, len(entries)):
            entry = entries[i].copy()
            prev_end = self._timestamp_to_seconds(fixed[-1]["end_time"])
            curr_start = self._timestamp_to_seconds(entry["start_time"])

            if curr_start < prev_end + self.min_gap:
                # Adjust: shrink previous end or push current start
                new_start = prev_end + self.min_gap
                entry["start_time"] = self._seconds_to_timestamp(new_start)
                logger.debug(
                    f"Fixed overlap: entry {i} start adjusted to {entry['start_time']}"
                )

            fixed.append(entry)

        return fixed

    def reindex(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Re-index subtitle entries starting from 1."""
        return [
            {**entry, "index": i + 1}
            for i, entry in enumerate(entries)
        ]

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
        parts = timestamp.replace(",", ":").split(":")
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + ms / 1000.0

    @staticmethod
    def _seconds_to_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
