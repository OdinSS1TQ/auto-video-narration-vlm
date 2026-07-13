"""
Voice Eval — Evaluate voice cloning quality across pipeline stages.
"""

from pathlib import Path

import numpy as np

from src.m5_evaluation.speaker_similarity import SpeakerSimilarity


def evaluate_voice_clone(
    reference_audio: Path,
    raw_chunks_dir: Path | None,
    aligned_chunks_dir: Path | None,
    merged_audio: Path | None,
) -> dict:
    sim = SpeakerSimilarity()
    result: dict = {}

    def _stats(dir_path):
        if dir_path is None or not Path(dir_path).exists():
            return None
        out = sim.evaluate_batch(reference_audio, dir_path)
        if out.get("count", 0) == 0:
            return None
        per_file = [
            sim.compute_similarity(reference_audio, p)
            for p in sorted(Path(dir_path).glob("*.wav"))
        ]
        return {
            "mean": out["mean_similarity"],
            "std": out["std_similarity"],
            "p05": float(np.percentile(per_file, 5)) if per_file else 0.0,
            "n_chunks": out["count"],
        }

    result["pre_align"] = _stats(raw_chunks_dir)
    result["post_align"] = _stats(aligned_chunks_dir)

    if merged_audio and Path(merged_audio).exists():
        result["merged"] = {
            "similarity": sim.compute_similarity(reference_audio, merged_audio)
        }

    return result
