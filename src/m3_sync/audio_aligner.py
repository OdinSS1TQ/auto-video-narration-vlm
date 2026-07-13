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
                # compress_fit or compress_max — compute achieved using the
                # audio_duration we already captured in calculate_deltas so we
                # don't re-read the file (keeps tests deterministic and avoids
                # an extra IO).
                ratio_needed = audio_dur / target if target > 0 else float("inf")
                if ratio_needed <= self.max_speedup:
                    achieved = target
                    sub_strategy = "compress_fit"
                else:
                    achieved = audio_dur / self.max_speedup
                    sub_strategy = "compress_max"

                suffix = Path(audio_path).suffix
                stem = Path(audio_path).stem
                out_path = output_dir / f"{stem}_stretched{suffix}"
                self.time_stretcher.stretch(audio_path, out_path, achieved)

                new["aligned_audio_path"] = str(out_path)
                new["aligned_duration"] = achieved
                new["align_method"] = sub_strategy

            results.append(new)

        # --- Slip cascade ---
        # Only slip when the previous segment's audio actually overruns into
        # the next segment's SRT start. When they naturally abut (prev_end ==
        # next_start) we do NOT force a min_gap — that's the renderer's job
        # if it ever needs one — to avoid spurious slips on well-fit audio.
        for i, seg in enumerate(results):
            if i == 0:
                eff = seg["srt_start_sec"]
                slipped = False
            else:
                prev = results[i - 1]
                prev_end = prev["effective_start_sec"] + prev["aligned_duration"]
                if prev_end <= seg["srt_start_sec"] + 1e-9:
                    eff = seg["srt_start_sec"]
                    slipped = False
                else:
                    eff = prev_end + self.min_gap_sec
                    slipped = True
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
