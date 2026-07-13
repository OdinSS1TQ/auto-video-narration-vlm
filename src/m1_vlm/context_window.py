"""
Context Window — Sliding context manager for maintaining coherence across chunks.

Keeps a sliding window of previous translations to ensure consistency
in terminology and context across video chunks.

AI Researcher Architecture Note:
    The ContextWindow implements a 2-layer hierarchical context system:

    Layer 1 (Global): Video-level summary from Pass 0.
        - Set once before chunk processing begins.
        - Provides topic, sections, key terms for ALL chunks.
        - Solves the "chunk isolation" problem where late chunks
          have no knowledge of the video's opening context.

    Layer 2 (Sliding): Chunk-level sliding window.
        - Maintains last N chunk results for local continuity.
        - Extracts terminology mappings for translation consistency.
        - Limited to window_size=3 by default (sufficient for local context).

    The combination ensures both global coherence and local continuity
    without exceeding the VLM's context window budget.
"""

from collections import deque
from typing import Any, Dict, List, Optional


class ContextWindow:
    """Manage sliding context across processed chunks."""

    def __init__(self, window_size: int = 3, max_entries_per_chunk: int = 10):
        """
        Args:
            window_size: Number of previous chunks to keep in context.
                         3 chunks ≈ 135s of sliding context — enough for
                         local continuity without overwhelming the prompt.
            max_entries_per_chunk: Max subtitle entries to keep per chunk.
        """
        self.window_size = window_size
        self.max_entries_per_chunk = max_entries_per_chunk
        self._history: deque = deque(maxlen=window_size)
        self._terminology: Dict[str, str] = {}
        self._global_summary: Optional[str] = None

    def add_chunk_result(
        self,
        chunk_index: int,
        entries: List[Dict[str, Any]],
        summary: Optional[str] = None,
    ):
        """
        Add processed chunk results to the context window.

        Args:
            chunk_index: Chunk sequence number.
            entries: List of subtitle entries from this chunk.
            summary: Optional content summary of the chunk.
        """
        # Keep only the last N entries per chunk
        trimmed_entries = entries[-self.max_entries_per_chunk:]

        self._history.append({
            "chunk_index": chunk_index,
            "entries": trimmed_entries,
            "summary": summary,
        })

        # Extract terminology
        for entry in entries:
            original = entry.get("original_text", "").strip()
            translated = entry.get("translated_text", "").strip()
            if original and translated:
                # Track key term mappings (simple extraction)
                self._terminology[original.lower()] = translated

    # ------------------------------------------------------------------ #
    # Global Summary (Pass 0 — video-level context)
    # ------------------------------------------------------------------ #

    def set_global_summary(self, summary: str):
        """
        Set the global video summary from Pass 0.

        AI Technical Leader Note:
            This is called ONCE before per-chunk processing starts.
            The summary contains: topic, sections outline, key terms.
            It's injected into every chunk's prompt to provide
            video-level awareness that the sliding window cannot offer.

        Args:
            summary: JSON string or text summary from global analysis pass.
        """
        self._global_summary = summary

    def get_global_summary(self) -> Optional[str]:
        """Get the global video summary for prompt injection."""
        return self._global_summary

    # ------------------------------------------------------------------ #
    # Sliding Window Context (chunk-level)
    # ------------------------------------------------------------------ #

    def get_context_summary(self) -> str:
        """
        Get a summary of recent context for the next chunk.

        Enhanced: Shows ALL entries from previous chunks so the VLM knows
        exactly what has already been generated and must NOT repeat.

        Returns:
            Formatted context string for prompt injection.
        """
        if not self._history:
            return ""

        parts = []
        for chunk_data in self._history:
            chunk_idx = chunk_data["chunk_index"]
            summary = chunk_data.get("summary", "")
            entries = chunk_data["entries"]

            part = f"[Chunk {chunk_idx} — ALREADY GENERATED, DO NOT REPEAT]"
            if summary:
                part += f" {summary}"

            # Show ALL entries so VLM can see exactly what was generated
            for entry in entries:
                original = entry.get("original_text", "")
                translated = entry.get("translated_text", "")
                st = entry.get("start_time", "")
                et = entry.get("end_time", "")
                if original and translated:
                    part += f"\n  [{st}→{et}] '{original}' → '{translated}'"

            parts.append(part)

        return "\n".join(parts)

    def get_previous_translations(self, limit: int = 5) -> str:
        """
        Get the last N translations from the context.

        Args:
            limit: Maximum number of recent translations.

        Returns:
            Formatted string of recent translations.
        """
        recent = []
        for chunk_data in reversed(list(self._history)):
            for entry in reversed(chunk_data["entries"]):
                if len(recent) >= limit:
                    break
                original = entry.get("original_text", "")
                translated = entry.get("translated_text", "")
                if original and translated:
                    recent.append(f"'{original}' → '{translated}'")
            if len(recent) >= limit:
                break

        return "\n".join(reversed(recent))

    def get_terminology_glossary(self) -> Dict[str, str]:
        """Get accumulated terminology mappings."""
        return dict(self._terminology)

    def clear(self):
        """Reset the context window (both global and sliding)."""
        self._history.clear()
        self._terminology.clear()
        self._global_summary = None
