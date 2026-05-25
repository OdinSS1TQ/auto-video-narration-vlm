"""
Translation Eval — Evaluate translation quality from SRT files.
"""

from pathlib import Path

from loguru import logger

from src.m5_evaluation.bleu_scorer import BLEUScorer
from src.m1_vlm.srt_builder import SRTBuilder


def evaluate_translation_from_srts(hyp_srt: Path, ref_srt: Path) -> dict:
    hyp = SRTBuilder.load_srt(hyp_srt)
    ref = SRTBuilder.load_srt(ref_srt)

    if len(hyp) != len(ref):
        logger.warning(f"Length mismatch: hyp={len(hyp)} ref={len(ref)}; truncating to min")

    n = min(len(hyp), len(ref))
    hyp_texts = [h["translated_text"] for h in hyp[:n]]
    ref_texts = [[r["translated_text"] for r in ref[:n]]]

    scorer = BLEUScorer()
    metrics = scorer.evaluate(hyp_texts, ref_texts)

    import sacrebleu
    sent_chrfs = [
        sacrebleu.sentence_chrf(h, [r], word_order=2).score
        for h, r in zip(hyp_texts, ref_texts[0])
    ]
    worst = sorted(range(n), key=lambda i: sent_chrfs[i])[:5]
    metrics["worst5"] = [
        {"index": i + 1, "chrf": sent_chrfs[i], "hyp": hyp_texts[i], "ref": ref_texts[0][i]}
        for i in worst
    ]
    metrics["n_entries"] = n
    return metrics
