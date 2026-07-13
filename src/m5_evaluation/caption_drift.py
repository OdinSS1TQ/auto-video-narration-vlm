"""
Caption Drift — Compare predicted vs ground-truth caption timelines.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class CaptionEntry:
    start_sec: float
    end_sec: float
    text: str


@dataclass
class DriftStats:
    n_matched: int
    n_missed_in_pred: int
    n_extra_in_pred: int
    start_drift_mean: float
    start_drift_p50: float
    start_drift_p95: float
    end_drift_mean: float
    duration_jaccard_mean: float


def compare_caption_tracks(
    pred: List[CaptionEntry],
    truth: List[CaptionEntry],
    text_match_ratio: float = 0.7,
    time_match_window_sec: float = 1.5,
) -> DriftStats:
    used = set()
    starts, ends, jaccards = [], [], []
    n_missed = 0

    for gt in truth:
        match_idx = None
        for j, p in enumerate(pred):
            if j in used:
                continue
            if abs(p.start_sec - gt.start_sec) > time_match_window_sec:
                continue
            ratio = SequenceMatcher(None, p.text.lower(), gt.text.lower()).ratio()
            if ratio >= text_match_ratio:
                match_idx = j
                break
        if match_idx is None:
            n_missed += 1
            continue
        used.add(match_idx)
        p = pred[match_idx]
        starts.append(abs(p.start_sec - gt.start_sec))
        ends.append(abs(p.end_sec - gt.end_sec))
        inter = max(0.0, min(p.end_sec, gt.end_sec) - max(p.start_sec, gt.start_sec))
        union = max(p.end_sec, gt.end_sec) - min(p.start_sec, gt.start_sec)
        jaccards.append(inter / union if union > 0 else 0.0)

    n_extra = len(pred) - len(used)
    if starts:
        sd = np.array(starts)
        ed = np.array(ends)
        jc = np.array(jaccards)
        return DriftStats(
            n_matched=len(starts),
            n_missed_in_pred=n_missed,
            n_extra_in_pred=n_extra,
            start_drift_mean=float(sd.mean()),
            start_drift_p50=float(np.percentile(sd, 50)),
            start_drift_p95=float(np.percentile(sd, 95)),
            end_drift_mean=float(ed.mean()),
            duration_jaccard_mean=float(jc.mean()),
        )
    return DriftStats(0, n_missed, n_extra, 0.0, 0.0, 0.0, 0.0, 0.0)


def load_caption_entries_from_srt(path: str | Path) -> List[CaptionEntry]:
    import pysrt

    subs = pysrt.open(str(path))
    entries = []
    for sub in subs:
        start = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
        end = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
        entries.append(CaptionEntry(start_sec=start, end_sec=end, text=sub.text))
    return entries


def load_caption_entries_from_ocr_json(path: str | Path) -> List[CaptionEntry]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = []
    for seg in data:
        entries.append(CaptionEntry(
            start_sec=float(seg["start_sec"]),
            end_sec=float(seg["end_sec"]),
            text=seg.get("en_text", seg.get("text", "")),
        ))
    return entries
