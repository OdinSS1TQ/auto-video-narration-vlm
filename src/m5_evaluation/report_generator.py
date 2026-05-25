"""
Report Generator — Aggregate evaluation metrics and export to JSON, Markdown, CSV.
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        video_name: str,
        translation_metrics: Optional[Dict] = None,
        mos_metrics: Optional[Dict] = None,
        similarity_metrics: Optional[Dict] = None,
        sync_metrics: Optional[Dict] = None,
        caption_drift_metrics: Optional[Dict] = None,
        pipeline_mode: Optional[str] = None,
        pipeline_elapsed_sec: Optional[float] = None,
        pipeline_config_snapshot: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        report = {
            "video_name": video_name,
            "timestamp": datetime.now().isoformat(),
            "pipeline_mode": pipeline_mode or "unknown",
            "pipeline_elapsed_sec": pipeline_elapsed_sec,
            "translation": translation_metrics or {},
            "voice_quality": mos_metrics or {},
            "speaker_similarity": similarity_metrics or {},
            "sync_accuracy": sync_metrics or {},
            "caption_drift": caption_drift_metrics or {},
            "pipeline_config": pipeline_config_snapshot or {},
        }

        json_path = self.output_dir / f"{video_name}_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved: {json_path}")
        return report

    def generate_markdown(self, report: Dict[str, Any]) -> Path:
        video_name = report.get("video_name", "unknown")
        lines = [
            f"# Evaluation Report — {video_name}",
            "",
            f"**Generated:** {report.get('timestamp', '')}",
            f"**Pipeline mode:** {report.get('pipeline_mode', 'unknown')}",
        ]
        if report.get("pipeline_elapsed_sec") is not None:
            lines.append(f"**Pipeline duration:** {report['pipeline_elapsed_sec']:.1f} s")
        lines.append("")

        # Summary table
        summary_rows = []
        trans = report.get("translation", {})
        sim = report.get("speaker_similarity", {})
        sync = report.get("sync_accuracy", {})
        drift = report.get("caption_drift", {})

        if trans.get("bleu4") is not None:
            summary_rows.append(("BLEU-4", f"{trans['bleu4']:.1f}"))
        if trans.get("chrf_plus_plus") is not None:
            summary_rows.append(("chrF++", f"{trans['chrf_plus_plus']:.1f}"))
        merged_sim = (sim.get("merged") or {}).get("similarity")
        if merged_sim is not None:
            summary_rows.append(("Speaker similarity (merged)", f"{merged_sim:.2f}"))
        if sync.get("p95") is not None:
            summary_rows.append(("Sync onset p95", f"{sync['p95']:.2f} s"))
        if drift.get("start_drift_p95") is not None:
            summary_rows.append(("Caption drift p95", f"{drift['start_drift_p95']:.2f} s"))

        if summary_rows:
            lines.extend(["## Summary", "", "| Metric | Value |", "|---|---|"])
            for label, val in summary_rows:
                lines.append(f"| {label} | {val} |")
            lines.append("")

        # Translation section
        if trans:
            lines.extend(["## Translation", ""])
            lines.append(f"- **BLEU-4:** {trans.get('bleu4', 'N/A')}")
            lines.append(f"- **chrF++:** {trans.get('chrf_plus_plus', 'N/A')}")
            lines.append(f"- **Entries evaluated:** {trans.get('n_entries', 'N/A')}")
            worst = trans.get("worst5", [])
            if worst:
                lines.extend(["", "**Worst 5 sentences by chrF++:**", ""])
                lines.append("| # | chrF++ | Hypothesis | Reference |")
                lines.append("|---|---|---|---|")
                for w in worst:
                    lines.append(f"| {w['index']} | {w['chrf']:.1f} | {w['hyp'][:60]} | {w['ref'][:60]} |")
            lines.append("")

        # Voice clone section
        if sim:
            lines.extend(["## Voice Clone", ""])
            for stage in ["pre_align", "post_align"]:
                stage_data = sim.get(stage)
                if stage_data:
                    label = stage.replace("_", " ").title()
                    lines.append(f"- **{label}:** mean={stage_data['mean']:.3f}, std={stage_data['std']:.3f}, p05={stage_data['p05']:.3f} (n={stage_data['n_chunks']})")
            if sim.get("merged"):
                lines.append(f"- **Merged audio:** similarity={sim['merged']['similarity']:.3f}")
            lines.append("")

        # Sync section
        if sync:
            lines.extend(["## Sync Accuracy", ""])
            lines.append(f"- **Mean delay:** {sync.get('mean_delay', 'N/A')}")
            lines.append(f"- **p50:** {sync.get('p50', 'N/A')}")
            lines.append(f"- **p95:** {sync.get('p95', 'N/A')}")
            lines.append(f"- **Missed onsets:** {sync.get('n_missed', 'N/A')}")
            lines.append(f"- **Count:** {sync.get('count', 'N/A')}")
            lines.append("")

        # Caption drift section
        if drift:
            lines.extend(["## Caption Drift (OCR-mode)", ""])
            lines.append(f"- **Matched:** {drift.get('n_matched', 'N/A')}")
            lines.append(f"- **Missed in pred:** {drift.get('n_missed_in_pred', 'N/A')}")
            lines.append(f"- **Extra in pred:** {drift.get('n_extra_in_pred', 'N/A')}")
            lines.append(f"- **Start drift mean:** {drift.get('start_drift_mean', 'N/A')}")
            lines.append(f"- **Start drift p50:** {drift.get('start_drift_p50', 'N/A')}")
            lines.append(f"- **Start drift p95:** {drift.get('start_drift_p95', 'N/A')}")
            lines.append(f"- **End drift mean:** {drift.get('end_drift_mean', 'N/A')}")
            lines.append(f"- **Duration Jaccard mean:** {drift.get('duration_jaccard_mean', 'N/A')}")
            lines.append("")

        # Config snapshot
        config = report.get("pipeline_config", {})
        if config:
            lines.extend(["## Config Snapshot", "", "```yaml"])
            for k, v in sorted(config.items()):
                lines.append(f"{k}: {v}")
            lines.extend(["```", ""])

        md_path = self.output_dir / f"{video_name}_report.md"
        md_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"Markdown report saved: {md_path}")
        return md_path

    def generate_summary_csv(
        self,
        reports: List[Dict[str, Any]],
        filename: str = "evaluation_summary.csv",
    ) -> Path:
        csv_path = self.output_dir / filename

        fieldnames = [
            "video_name",
            "pipeline_mode",
            "bleu4",
            "chrf_plus_plus",
            "mean_mos",
            "mean_similarity",
            "speaker_sim_merged",
            "mean_delay",
            "sync_delay_p95",
            "caption_drift_p95",
            "timestamp",
        ]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for report in reports:
                sim = report.get("speaker_similarity", {})
                merged_sim = (sim.get("merged") or {}).get("similarity", "")
                row = {
                    "video_name": report.get("video_name", ""),
                    "pipeline_mode": report.get("pipeline_mode", ""),
                    "bleu4": report.get("translation", {}).get("bleu4", ""),
                    "chrf_plus_plus": report.get("translation", {}).get("chrf_plus_plus", ""),
                    "mean_mos": report.get("voice_quality", {}).get("mean_mos", ""),
                    "mean_similarity": sim.get("mean_similarity", sim.get("pre_align", {}).get("mean", "") if isinstance(sim.get("pre_align"), dict) else ""),
                    "speaker_sim_merged": merged_sim,
                    "mean_delay": report.get("sync_accuracy", {}).get("mean_delay", ""),
                    "sync_delay_p95": report.get("sync_accuracy", {}).get("p95", ""),
                    "caption_drift_p95": report.get("caption_drift", {}).get("start_drift_p95", ""),
                    "timestamp": report.get("timestamp", ""),
                }
                writer.writerow(row)

        logger.info(f"Summary CSV saved: {csv_path} ({len(reports)} entries)")
        return csv_path

    def load_reports(self, pattern: str = "*_report.json") -> List[Dict[str, Any]]:
        reports = []
        for json_file in sorted(self.output_dir.glob(pattern)):
            with open(json_file, "r", encoding="utf-8") as f:
                reports.append(json.load(f))
        return reports
