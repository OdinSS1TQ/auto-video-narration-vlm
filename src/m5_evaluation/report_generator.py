"""
Report Generator — Aggregate evaluation metrics and export to CSV.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class ReportGenerator:
    """Generate evaluation reports from collected metrics."""

    def __init__(self, output_dir: str | Path = "./docs/eval_results"):
        """
        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        video_name: str,
        translation_metrics: Optional[Dict] = None,
        mos_metrics: Optional[Dict] = None,
        similarity_metrics: Optional[Dict] = None,
        sync_metrics: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Generate a complete evaluation report for a single video.

        Args:
            video_name: Name of the evaluated video.
            translation_metrics: BLEU/chrF++ scores.
            mos_metrics: MOS estimation results.
            similarity_metrics: Speaker similarity scores.
            sync_metrics: Sync accuracy measurements.

        Returns:
            Combined report dict.
        """
        report = {
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(),
            "translation": translation_metrics or {},
            "voice_quality": mos_metrics or {},
            "speaker_similarity": similarity_metrics or {},
            "sync_accuracy": sync_metrics or {},
        }

        # Save as JSON
        json_path = self.output_dir / f"{video_name}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved: {json_path}")
        return report

    def generate_summary_csv(
        self,
        reports: List[Dict[str, Any]],
        filename: str = "evaluation_summary.csv",
    ) -> Path:
        """
        Generate a summary CSV from multiple evaluation reports.

        Args:
            reports: List of report dicts.
            filename: Output CSV filename.

        Returns:
            Path to the generated CSV file.
        """
        csv_path = self.output_dir / filename

        fieldnames = [
            "video_name",
            "bleu4",
            "chrf_plus_plus",
            "mean_mos",
            "mean_similarity",
            "mean_delay",
            "timestamp",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for report in reports:
                row = {
                    "video_name": report.get("video_name", ""),
                    "bleu4": report.get("translation", {}).get("bleu4", ""),
                    "chrf_plus_plus": report.get("translation", {}).get("chrf_plus_plus", ""),
                    "mean_mos": report.get("voice_quality", {}).get("mean_mos", ""),
                    "mean_similarity": report.get("speaker_similarity", {}).get("mean_similarity", ""),
                    "mean_delay": report.get("sync_accuracy", {}).get("mean_delay", ""),
                    "timestamp": report.get("timestamp", ""),
                }
                writer.writerow(row)

        logger.info(f"Summary CSV saved: {csv_path} ({len(reports)} entries)")
        return csv_path

    def load_reports(self, pattern: str = "*_report.json") -> List[Dict[str, Any]]:
        """Load all report JSON files from the output directory."""
        reports = []
        for json_file in sorted(self.output_dir.glob(pattern)):
            with open(json_file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        return reports
