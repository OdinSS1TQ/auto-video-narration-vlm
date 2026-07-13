"""CLI: Evaluate a dubbed video against ground truth + reference audio."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.m5_evaluation import (
    ReportGenerator,
    compare_caption_tracks,
    evaluate_translation_from_srts,
    evaluate_voice_clone,
)
from src.m5_evaluation.caption_drift import (
    load_caption_entries_from_ocr_json,
    load_caption_entries_from_srt,
)
from src.m5_evaluation.sync_accuracy import SyncAccuracy
from src.m1_vlm.srt_builder import SRTBuilder
from src.m4_pipeline.config import PipelineConfig


def build_config_snapshot(cfg: PipelineConfig) -> dict:
    return {
        "PIPELINE_MODE": cfg.pipeline_mode,
        "VLM_MODE": cfg.vlm_mode,
        "VLM_MODEL_NAME": cfg.vlm_model_name,
        "M3_MAX_SPEEDUP": cfg.m3_max_speedup,
        "M3_MIN_GAP_SEC": cfg.m3_min_gap_sec,
        "M1_VI_CHARS_PER_SEC": cfg.vi_chars_per_sec,
        "M1_CHUNK_FILL_RATIO": cfg.chunk_fill_ratio,
        "CAPTION_BAND_RATIO": cfg.caption_band_ratio,
        "OCR_SAMPLE_FPS": cfg.ocr_sample_fps,
        "CAPTION_MERGE_MAX_GAP_SEC": cfg.caption_merge_max_gap_sec,
        "CAPTION_MERGE_MAX_CHARS": cfg.caption_merge_max_chars,
        "CAPTION_EXTEND_END_SEC": cfg.caption_extend_end_sec,
        "CAPTION_DEDUP_RATIO": cfg.caption_dedup_ratio,
        "CAPTION_MIN_DURATION_SEC": cfg.caption_min_duration_sec,
        "SCENE_THRESHOLD": cfg.scene_threshold,
        "SSIM_THRESHOLD": cfg.ssim_threshold,
        "CHUNK_DURATION_SEC": cfg.chunk_duration,
        "TTS_ENGINE": cfg.tts_engine,
        "TTS_SAMPLE_RATE": cfg.tts_sample_rate,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a dubbed video against ground truth + reference audio."
    )
    p.add_argument("--video", required=True, help="Path to original input video")
    p.add_argument("--srt", required=True, help="Path to generated Vietnamese SRT")
    p.add_argument("--dubbed", required=True, help="Path to dubbed output video")
    p.add_argument("--ref-audio", required=True, help="Path to speaker reference audio")
    p.add_argument("--audio-chunks", help="Dir of raw TTS audio chunks")
    p.add_argument("--aligned-audio", help="Dir of time-stretched aligned audio chunks")
    p.add_argument("--merged-audio", help="Path to merged audio WAV")
    p.add_argument("--ground-truth-srt", help="Path to hand-curated Vietnamese reference SRT")
    p.add_argument("--ground-truth-captions", help="Path to hand-curated English caption SRT (for drift)")
    p.add_argument("--ocr-segments-json", help="Path to OCR segments_final.json (for drift)")
    p.add_argument(
        "--pipeline-mode",
        default="unknown",
        choices=["vlm", "ocr", "unknown"],
    )
    p.add_argument("--output-name", required=True, help="Base name for report files")
    args = p.parse_args(argv)

    cfg = PipelineConfig()
    config_snap = build_config_snapshot(cfg)

    translation = sync = voice = drift = None
    ran_any = False

    # --- Translation ---
    if args.ground_truth_srt:
        logger.info("Evaluating translation quality...")
        translation = evaluate_translation_from_srts(
            Path(args.srt), Path(args.ground_truth_srt)
        )
        logger.info(
            f"  BLEU-4={translation.get('bleu4', 'N/A')}, "
            f"chrF++={translation.get('chrf_plus_plus', 'N/A')}"
        )
        ran_any = True

    # --- Voice clone similarity ---
    if args.audio_chunks or args.aligned_audio or args.merged_audio:
        logger.info("Evaluating voice clone similarity...")
        voice = evaluate_voice_clone(
            Path(args.ref_audio),
            Path(args.audio_chunks) if args.audio_chunks else None,
            Path(args.aligned_audio) if args.aligned_audio else None,
            Path(args.merged_audio) if args.merged_audio else None,
        )
        merged_sim = (voice.get("merged") or {}).get("similarity")
        if merged_sim is not None:
            logger.info(f"  Merged speaker similarity={merged_sim:.3f}")
        ran_any = True

    # --- Sync accuracy ---
    if args.merged_audio:
        logger.info("Evaluating sync accuracy against merged audio...")
        sa = SyncAccuracy()
        srt_entries = SRTBuilder.load_srt(Path(args.srt))
        sync = sa.evaluate_against_merged_audio(srt_entries, Path(args.merged_audio))
        logger.info(f"  Sync p95={sync.get('p95', 'N/A')}, missed={sync.get('n_missed', 0)}")
        ran_any = True

    # --- Caption drift ---
    if args.ground_truth_captions:
        logger.info("Evaluating caption drift...")
        truth = load_caption_entries_from_srt(Path(args.ground_truth_captions))
        if args.ocr_segments_json:
            pred = load_caption_entries_from_ocr_json(Path(args.ocr_segments_json))
        else:
            pred = load_caption_entries_from_srt(Path(args.srt))
        drift_stats = compare_caption_tracks(pred, truth)
        drift = drift_stats.__dict__
        logger.info(
            f"  Matched={drift['n_matched']}, drift p95={drift['start_drift_p95']:.3f}s"
        )
        ran_any = True

    if not ran_any:
        print(
            "ERROR: No evaluation produced. Provide at least one "
            "ground-truth or audio input.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Generate reports ---
    rep = ReportGenerator()
    report = rep.generate_report(
        video_name=args.output_name,
        translation_metrics=translation,
        similarity_metrics=voice,
        sync_metrics=sync,
        caption_drift_metrics=drift,
        pipeline_mode=args.pipeline_mode,
        pipeline_config_snapshot=config_snap,
    )
    md_path = rep.generate_markdown(report)
    rep.generate_summary_csv(rep.load_reports())

    json_path = rep.output_dir / f"{args.output_name}_report.json"
    print(f"\nJSON:     {json_path}")
    print(f"Markdown: {md_path}")
    csv_path = rep.output_dir / "evaluation_summary.csv"
    print(f"CSV:      {csv_path}")


if __name__ == "__main__":
    main()
