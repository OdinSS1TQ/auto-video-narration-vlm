"""
SRT Builder — Merge processed chunks into a standard .srt subtitle file.

Combines validated subtitle entries from multiple chunks into a single SRT file.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pysrt
from loguru import logger


class SRTBuilder:
    """Build and manage SRT subtitle files."""

    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    def add_entries(self, entries: List[Dict[str, Any]], chunk_offset: float = 0.0):
        """
        Add subtitle entries from a processed chunk.

        Args:
            entries: List of subtitle entry dicts with keys:
                     index, start_time, end_time, translated_text, (optional) original_text
            chunk_offset: Time offset in seconds for this chunk.
        """
        for entry in entries:
            adjusted_entry = entry.copy()
            if chunk_offset > 0:
                adjusted_entry["start_time"] = self._add_offset(
                    entry["start_time"], chunk_offset
                )
                adjusted_entry["end_time"] = self._add_offset(
                    entry["end_time"], chunk_offset
                )
            self._entries.append(adjusted_entry)

    def build_srt_content(self) -> str:
        """
        Build SRT file content from all added entries.

        Returns:
            SRT formatted string.
        """
        # Sort by start time
        sorted_entries = sorted(
            self._entries,
            key=lambda e: self._timestamp_to_seconds(e["start_time"]),
        )

        lines = []
        for i, entry in enumerate(sorted_entries, 1):
            lines.append(str(i))
            lines.append(f"{entry['start_time']} --> {entry['end_time']}")
            lines.append(entry["translated_text"])
            lines.append("")  # Blank line separator

        return "\n".join(lines)

    def save(self, output_path: str | Path, encoding: str = "utf-8") -> Path:
        """
        Save SRT content to file.

        Args:
            output_path: Output file path.
            encoding: File encoding.

        Returns:
            Path to the saved file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        content = self.build_srt_content()
        output_path.write_text(content, encoding=encoding)

        logger.info(f"Saved SRT with {len(self._entries)} entries to {output_path}")
        return output_path

    def validate_with_pysrt(self, srt_path: str | Path) -> bool:
        """
        Validate an SRT file using pysrt.

        Args:
            srt_path: Path to SRT file.

        Returns:
            True if file is valid.
        """
        try:
            subs = pysrt.open(str(srt_path))
            logger.info(f"Valid SRT file: {len(subs)} subtitles")
            return True
        except Exception as e:
            logger.error(f"Invalid SRT file: {e}")
            return False

    def clear(self):
        """Clear all entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        """Number of entries currently stored."""
        return len(self._entries)

    @staticmethod
    def _timestamp_to_seconds(timestamp: str) -> float:
        """Convert SRT timestamp to seconds."""
        parts = timestamp.replace(",", ":").split(":")
        h, m, s, ms = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        return h * 3600 + m * 60 + s + ms / 1000.0

    @staticmethod
    def _add_offset(timestamp: str, offset_seconds: float) -> str:
        """Add offset to a timestamp."""
        seconds = SRTBuilder._timestamp_to_seconds(timestamp) + offset_seconds
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    @staticmethod
    def load_srt(srt_path: str | Path) -> List[Dict[str, Any]]:
        """
        Load an SRT file and return as list of dicts.

        Args:
            srt_path: Path to SRT file.

        Returns:
            List of subtitle entry dicts.
        """
        subs = pysrt.open(str(srt_path))
        return [
            {
                "index": sub.index,
                "start_time": str(sub.start).replace(".", ","),
                "end_time": str(sub.end).replace(".", ","),
                "translated_text": sub.text,
            }
            for sub in subs
        ]
