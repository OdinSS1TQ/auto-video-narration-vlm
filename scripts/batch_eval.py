"""
CLI: Run evaluation on a set of test videos.

Usage:
    python scripts/batch_eval.py --eval-dir data/eval_set --output docs/eval_results/
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m5_evaluation.report_generator import ReportGenerator
from src.m4_pipeline.logger import setup_logger
from loguru import logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch evaluation on test videos"
    )
    parser.add_argument(
        "--eval-dir",
        default="data/eval_set",
        help="Directory containing test videos",
    )
    parser.add_argument(
        "--output",
        default="docs/eval_results",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--ref-audio",
        default=None,
        help="Reference audio for all videos (optional)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    setup_logger(log_level="INFO")

    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        logger.error(f"Evaluation directory not found: {eval_dir}")
        sys.exit(1)

    video_files = list(eval_dir.glob("*.mp4")) + list(eval_dir.glob("*.avi"))
    if not video_files:
        logger.error(f"No video files found in {eval_dir}")
        sys.exit(1)

    logger.info(f"Found {len(video_files)} videos for evaluation")

    report_gen = ReportGenerator(output_dir=args.output)
    reports = []

    for video_path in video_files:
        logger.info(f"Evaluating: {video_path.name}")

        # TODO: Run pipeline and collect metrics for each video
        # For now, create placeholder report
        report = report_gen.generate_report(
            video_name=video_path.stem,
            translation_metrics={"bleu4": 0.0, "chrf_plus_plus": 0.0},
            mos_metrics={"mean_mos": 0.0},
            similarity_metrics={"mean_similarity": 0.0},
            sync_metrics={"mean_delay": 0.0},
        )
        reports.append(report)

    # Generate summary CSV
    csv_path = report_gen.generate_summary_csv(reports)
    logger.info(f"Summary CSV: {csv_path}")
    print(f"\n✅ Evaluation complete. Results in {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
