"""
BLEU Scorer — sacrebleu wrapper for translation quality evaluation.
"""

from typing import List, Optional

from loguru import logger


class BLEUScorer:
    """Evaluate translation quality using BLEU and chrF++ metrics."""

    def __init__(self):
        """Initialize BLEU scorer."""
        import sacrebleu
        self._sacrebleu = sacrebleu

    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> dict:
        """
        Compute BLEU-4 score.

        Args:
            hypotheses: List of translated texts.
            references: List of reference translations (each can have multiple refs).

        Returns:
            Dict with score, precisions, bp, etc.
        """
        bleu = self._sacrebleu.corpus_bleu(hypotheses, references)
        logger.info(f"BLEU-4: {bleu.score:.2f}")
        return {
            "bleu4": bleu.score,
            "precisions": bleu.precisions,
            "brevity_penalty": bleu.bp,
            "sys_len": bleu.sys_len,
            "ref_len": bleu.ref_len,
        }

    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> dict:
        """
        Compute chrF++ score.

        Args:
            hypotheses: List of translated texts.
            references: List of reference translations.

        Returns:
            Dict with chrF++ score.
        """
        chrf = self._sacrebleu.corpus_chrf(hypotheses, references, word_order=2)
        logger.info(f"chrF++: {chrf.score:.2f}")
        return {
            "chrf_plus_plus": chrf.score,
        }

    def evaluate(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> dict:
        """
        Run all translation metrics.

        Returns:
            Dict with all metric scores.
        """
        bleu_result = self.compute_bleu(hypotheses, references)
        chrf_result = self.compute_chrf(hypotheses, references)
        return {**bleu_result, **chrf_result}
