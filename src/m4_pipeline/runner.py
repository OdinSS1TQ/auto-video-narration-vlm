"""
Pipeline Runner — End-to-end dubbing pipeline orchestrator.

Coordinates VLM → TTS → Sync modules for complete video dubbing.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.m4_pipeline.config import PipelineConfig
from src.m4_pipeline.exceptions import PipelineError
from src.m1_vlm.scene_detector import SceneDetector
from src.m1_vlm.frame_extractor import FrameExtractor
from src.m1_vlm.frame_dedup import FrameDeduplicator
from src.m1_vlm.prompt_chain import PromptChain
from src.m1_vlm.vlm_client import VLMClient
from src.m1_vlm.validator import SubtitleValidator
from src.m1_vlm.srt_builder import SRTBuilder
from src.m1_vlm.context_window import ContextWindow
from src.m1_vlm.entry_retimer import EntryRetimer
from src.m2_tts.tts_client import TTSClient
from src.m2_tts.speaker_encoder import SpeakerEncoder  # reserved: m5_evaluation MOS scoring
from src.m2_tts.batch_inference import BatchInference
from src.m3_sync.audio_aligner import AudioAligner
from src.m3_sync.ffmpeg_renderer import FFmpegRenderer


class PipelineRunner:
    """Orchestrate the complete video dubbing pipeline."""

    # Pipeline steps for progress tracking
    STEPS = [
        "scene_detection",
        "frame_extraction",
        "ocr_extraction",
        "vlm_translation",
        "srt_generation",
        "voice_cloning",
        "audio_alignment",
        "video_rendering",
    ]

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Args:
            config: Pipeline configuration. Uses defaults if not provided.
        """
        self.config = config or PipelineConfig()
        self._progress_callback: Optional[Callable] = None
        self._current_step: int = 0

    def set_progress_callback(self, callback: Callable[[str, int, int], None]):
        """
        Set a callback for progress updates.

        Args:
            callback: Function(step_name, current_step, total_steps).
        """
        self._progress_callback = callback

    def _update_progress(self, step_name: str):
        """Notify progress callback."""
        self._current_step += 1
        if self._progress_callback:
            self._progress_callback(step_name, self._current_step, len(self.STEPS))
        logger.info(f"[{self._current_step}/{len(self.STEPS)}] {step_name}")

    async def run(
        self,
        video_path: str | Path,
        reference_audio_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete dubbing pipeline.

        Args:
            video_path: Path to input video.
            reference_audio_path: Path to reference audio for voice cloning.
            output_path: Path for output video (auto-generated if None).

        Returns:
            Dict with results including output_path, srt_path, metrics.
        """
        start_time = time.time()
        self._current_step = 0

        video_path = Path(video_path)
        reference_audio_path = Path(reference_audio_path)

        if not video_path.exists():
            raise PipelineError(f"Video not found: {video_path}")
        if not reference_audio_path.exists():
            raise PipelineError(f"Reference audio not found: {reference_audio_path}")

        # Setup output directory.
        # Sanitize stem to ASCII-safe — Windows passes paths to subprocesses
        # like rubberband via cp1252 which mangles characters like 【】 to '?'.
        import re as _re
        safe_stem = _re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem).strip("_") or "video"
        if output_path is None:
            output_dir = Path(self.config.output_dir) / safe_stem
        else:
            output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        work_dir = output_dir / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "video_path": str(video_path),
            "reference_audio": str(reference_audio_path),
        }

        try:
            # === Step 1: Scene Detection ===
            self._update_progress("scene_detection")
            scene_detector = SceneDetector(threshold=self.config.scene_threshold)
            chunks = scene_detector.split_into_chunks(
                video_path, chunk_duration=self.config.chunk_duration
            )
            results["chunks"] = len(chunks)
            logger.info(f"Detected {len(chunks)} chunks")

            # === Step 2: Frame Extraction (adaptive + SSIM dedup) ===
            # Mirrors scripts/run_vlm_extract.py — fixed-interval sampling
            # generates 30+ frames for a 60s chunk and OOMs the VLM.
            self._update_progress("frame_extraction")
            frame_extractor = FrameExtractor(max_width=768)
            dedup = FrameDeduplicator(ssim_threshold=0.85)
            max_frames = 8

            all_frames = {}
            total_before_dedup = 0
            total_after_dedup = 0
            for i, (start, end) in enumerate(chunks):
                timestamps = scene_detector.get_adaptive_timestamps(
                    video_path, start, end,
                    min_frames=3, max_frames=max_frames * 2,
                )
                chunk_frames = frame_extractor.extract_frames_at_timestamps(
                    video_path, timestamps
                )
                total_before_dedup += len(chunk_frames)
                if len(chunk_frames) > 1:
                    chunk_frames = dedup.deduplicate(chunk_frames)
                if len(chunk_frames) > max_frames:
                    step = len(chunk_frames) / max_frames
                    chunk_frames = [chunk_frames[int(j * step)] for j in range(max_frames)]
                total_after_dedup += len(chunk_frames)
                all_frames[i] = chunk_frames

            logger.info(
                f"Extracted frames for {len(chunks)} chunks: "
                f"{total_before_dedup} → {total_after_dedup} after SSIM dedup "
                f"(max_frames={max_frames})"
            )

            # === Step 3: OCR (optional) ===
            self._update_progress("ocr_extraction")
            # OCR step is optional — skip if model not available
            logger.info("OCR step: skipped (implement when GLM-OCR model is ready)")

            # === Step 4: VLM Translation ===
            self._update_progress("vlm_translation")
            vlm_client = VLMClient(
                mode=self.config.vlm_mode,
                model_name=self.config.vlm_model_name,
                api_key=self.config.gemini_api_key,
                local_model_path=self.config.qwen_model_path,
                temperature=self.config.vlm_temperature,
                max_tokens=self.config.vlm_max_tokens,
            )
            prompt_chain = PromptChain()
            context_window = ContextWindow()
            validator = SubtitleValidator()
            srt_builder = SRTBuilder()
            entry_retimer = EntryRetimer(
                chars_per_sec=self.config.vi_chars_per_sec,
                fill_ratio=self.config.chunk_fill_ratio,
                min_gap_sec=self.config.m3_min_gap_sec,
            )

            import json as _json

            # --- Global Summary Pass (Pass 0) ---
            # Without this, the per-chunk prompt has no `global_context`, and
            # Qwen3.5 tends to return a single-dict entry instead of the JSON
            # array the prompt asks for — yielding a 1-entry SRT.
            # Mirrors scripts/run_vlm_extract.py:215-307.
            global_frames_n = int(getattr(self.config, "global_summary_frames", 15) or 15)
            try:
                global_frames = frame_extractor.extract_frames_evenly(
                    video_path, n=global_frames_n
                )
            except Exception as exc:
                logger.warning(f"Global summary frame extraction failed: {exc}")
                global_frames = []

            if global_frames:
                global_b64 = frame_extractor.frames_to_base64_batch(global_frames)
                # Cap to max_frames so we don't OOM the VLM context.
                if len(global_b64) > max_frames:
                    step = len(global_b64) / max_frames
                    global_b64 = [global_b64[int(j * step)] for j in range(max_frames)]

                global_prompt = prompt_chain.build_global_summary_prompt()
                raw_global = await vlm_client.generate(
                    prompt=global_prompt, images_base64=global_b64
                )

                (work_dir / "global_summary_raw.txt").write_text(
                    raw_global, encoding="utf-8"
                )

                cleaned_g = raw_global.strip()
                if cleaned_g.startswith("```json"):
                    cleaned_g = cleaned_g[7:]
                if cleaned_g.startswith("```"):
                    cleaned_g = cleaned_g[3:]
                if cleaned_g.endswith("```"):
                    cleaned_g = cleaned_g[:-3]

                try:
                    parsed_g = _json.loads(cleaned_g.strip())
                    pretty_g = _json.dumps(parsed_g, indent=2, ensure_ascii=False)
                    (work_dir / "global_summary.json").write_text(
                        pretty_g, encoding="utf-8"
                    )
                    context_window.set_global_summary(pretty_g)
                    logger.info(
                        f"Global summary: topic='{parsed_g.get('topic', 'N/A')}', "
                        f"sections={len(parsed_g.get('sections', []))}"
                    )
                except _json.JSONDecodeError as exc:
                    logger.warning(
                        f"Global summary JSON parse failed ({exc}); using raw text"
                    )
                    context_window.set_global_summary(raw_global)
            else:
                logger.warning("Global summary skipped: no frames extracted")

            for chunk_idx, (start, end) in enumerate(chunks):
                frames = all_frames.get(chunk_idx, [])
                if not frames:
                    logger.warning(f"Chunk {chunk_idx}: no frames, skipping VLM call")
                    continue

                frame_images = [
                    frame_extractor.frame_to_base64(f) for _, f in frames
                ]
                timestamps = [t for t, _ in frames]

                chunk_info = (
                    f"Chunk {chunk_idx}: {start:.1f}s — {end:.1f}s\n"
                    f"Video: {video_path.name}\n"
                    f"Frames at timestamps: {[f'{t:.1f}s' for t in timestamps]}"
                )

                prompt = prompt_chain.build_single_prompt(
                    chunk_info=chunk_info,
                    context_summary=context_window.get_context_summary() or None,
                    previous_translations=context_window.get_previous_translations(limit=10) or None,
                    global_context=context_window.get_global_summary(),
                    frame_timestamps=timestamps,
                    chunk_start=start,
                    chunk_end=end,
                    chunk_index=chunk_idx,
                    total_chunks=len(chunks),
                )

                # Use generate() + manual JSON parse so an invalid response
                # from one chunk doesn't kill the whole pipeline.
                raw_response = await vlm_client.generate(
                    prompt=prompt,
                    images_base64=frame_images,
                )

                cleaned = raw_response.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.startswith("```"):
                    cleaned = cleaned[3:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]

                # Save raw response for diagnostics (every chunk, every run).
                raw_dump = work_dir / f"chunk_{chunk_idx:02d}_raw_response.txt"
                raw_dump.write_text(raw_response, encoding="utf-8")

                try:
                    parsed = _json.loads(cleaned.strip())
                except _json.JSONDecodeError as exc:
                    logger.warning(
                        f"Chunk {chunk_idx}: VLM response not valid JSON ({exc}); "
                        f"raw saved to {raw_dump}"
                    )
                    continue

                # Accept either:
                #   - JSON array of entries (preferred per prompt)
                #   - JSON object wrapping the array under any key
                #   - Single JSON object that IS one entry (wrap in list)
                if isinstance(parsed, list):
                    entries = parsed
                elif isinstance(parsed, dict):
                    list_values = [v for v in parsed.values() if isinstance(v, list)]
                    if list_values:
                        entries = list_values[0]
                        logger.info(
                            f"Chunk {chunk_idx}: VLM returned dict; extracted list "
                            f"from key value (raw saved to {raw_dump})"
                        )
                    elif {"start_time", "end_time", "translated_text"} <= set(parsed.keys()):
                        entries = [parsed]
                        logger.info(
                            f"Chunk {chunk_idx}: VLM returned single entry as dict; "
                            f"wrapped in list (raw saved to {raw_dump})"
                        )
                    else:
                        logger.warning(
                            f"Chunk {chunk_idx}: VLM returned dict with no usable list "
                            f"or entry shape; raw saved to {raw_dump}"
                        )
                        continue
                else:
                    logger.warning(
                        f"Chunk {chunk_idx}: VLM returned {type(parsed).__name__}; "
                        f"raw saved to {raw_dump}"
                    )
                    continue

                retimed = entry_retimer.retime(
                    entries=entries,
                    chunk_start=start,
                    chunk_end=end,
                    frame_timestamps=timestamps,
                )
                logger.info(
                    f"Chunk {chunk_idx}: parsed {len(entries)} entries; "
                    f"retimer rewrote starts (first={retimed[0]['start_time']}, "
                    f"last_end={retimed[-1]['end_time']})"
                )

                valid, issues = validator.validate_sequence(retimed)
                if not valid:
                    logger.debug(f"Chunk {chunk_idx}: validation issues fixed: {issues}")
                    retimed = validator.fix_overlaps(retimed)
                    retimed = validator.reindex(retimed)

                srt_builder.add_entries(retimed, chunk_offset=0.0)
                context_window.add_chunk_result(chunk_idx, retimed)

            # === Step 5: SRT Generation ===
            self._update_progress("srt_generation")
            srt_path = output_dir / f"{safe_stem}_vi.srt"
            srt_builder.save(srt_path)
            results["srt_path"] = str(srt_path)

            # Free VLM VRAM before loading TTS — both don't fit on 8 GB GPUs.
            try:
                vlm_client.unload_model()
            except Exception as exc:
                logger.warning(f"VLM unload failed (continuing): {exc}")
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

            # === Step 6: Voice Cloning (VieNeu-TTS v2 Turbo) ===
            self._update_progress("voice_cloning")

            # Initialize TTSClient with VieNeu-TTS v2 Turbo
            # Models are auto-downloaded from HuggingFace on first run.
            tts_client = TTSClient(
                engine=self.config.tts_engine,
                backbone_repo=self.config.tts_backbone_repo,
                backbone_device=self.config.tts_backbone_device,
                codec_device=self.config.tts_codec_device,
                vieneu_mode=self.config.tts_vieneu_mode,
                hf_token=self.config.tts_hf_token,
                sample_rate=self.config.tts_sample_rate,
            )

            # Encode reference audio ONCE — ref_codes are reused for all segments.
            # VieNeu v2 Turbo: no ref_text required (truly zero-shot).
            logger.info(f"Encoding reference voice: {reference_audio_path.name}")
            ref_codes = tts_client.encode_reference(reference_audio_path, use_cache=True)

            batch_inference = BatchInference(tts_client)

            # Load SRT entries for TTS
            srt_entries = SRTBuilder.load_srt(srt_path)
            audio_segments = await batch_inference.process_all(
                segments=srt_entries,
                output_dir=work_dir / "audio_chunks",
                ref_codes=ref_codes,   # pre-encoded, reused across all segments
                ref_text=None,         # v2 Turbo: zero-shot, no transcript needed
            )

            # === Step 7: Audio Alignment ===
            self._update_progress("audio_alignment")
            aligner = AudioAligner(
                max_speedup=self.config.m3_max_speedup,
                min_gap_sec=self.config.m3_min_gap_sec,
            )
            segments_with_deltas = aligner.calculate_deltas(audio_segments)
            aligned_segments = aligner.align_all(
                segments_with_deltas,
                output_dir=work_dir / "aligned_audio",
            )

            # === Step 8: Video Rendering ===
            self._update_progress("video_rendering")
            renderer = FFmpegRenderer()

            # Get video duration
            video_info = FFmpegRenderer.get_video_info(video_path)
            duration = float(video_info["format"]["duration"])

            # Merge audio
            merged_audio_path = work_dir / "merged_audio.wav"
            renderer.merge_audio_segments(
                aligned_segments,
                duration,
                merged_audio_path,
                sample_rate=24000,
            )

            # Final render
            final_output = output_path or output_dir / f"{safe_stem}_dubbed.mp4"
            renderer.render_final_video(
                video_path=video_path,
                dubbed_audio_path=merged_audio_path,
                output_path=final_output,
            )
            results["output_path"] = str(final_output)

        except Exception as e:
            logger.error(f"Pipeline failed at step {self._current_step}: {e}")
            results["error"] = str(e)
            raise PipelineError(f"Pipeline failed: {e}") from e

        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        logger.info(f"Pipeline completed in {elapsed:.1f}s")

        return results
