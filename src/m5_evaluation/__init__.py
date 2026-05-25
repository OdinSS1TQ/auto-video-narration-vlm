"""
Evaluation Module — Metrics for translation, voice, and sync quality.
"""

from src.m5_evaluation.bleu_scorer import BLEUScorer
from src.m5_evaluation.mos_estimator import MOSEstimator
from src.m5_evaluation.speaker_similarity import SpeakerSimilarity
from src.m5_evaluation.sync_accuracy import SyncAccuracy
from src.m5_evaluation.report_generator import ReportGenerator
from src.m5_evaluation.caption_drift import compare_caption_tracks, DriftStats, CaptionEntry
from src.m5_evaluation.translation_eval import evaluate_translation_from_srts
from src.m5_evaluation.voice_eval import evaluate_voice_clone

__all__ = [
    "BLEUScorer",
    "MOSEstimator",
    "SpeakerSimilarity",
    "SyncAccuracy",
    "ReportGenerator",
    "compare_caption_tracks",
    "DriftStats",
    "CaptionEntry",
    "evaluate_translation_from_srts",
    "evaluate_voice_clone",
]
