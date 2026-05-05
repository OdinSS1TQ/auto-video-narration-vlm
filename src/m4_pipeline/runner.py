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
from src.m1_vlm.prompt_chain import PromptChain
from src.m1_vlm.vlm_client import VLMClient
from src.m1_vlm.validator import SubtitleValidator
from src.m1_vlm.srt_builder import SRTBuilder
from src.m1_vlm.context_window import ContextWindow
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

        # Setup output directory
        if output_path is None:
            output_dir = Path(self.config.output_dir) / video_path.stem
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

            # === Step 2: Frame Extraction ===
            self._update_progress("frame_extraction")
            frame_extractor = FrameExtractor()
            all_frames = {}
            for i, (start, end) in enumerate(chunks):
                frames = frame_extractor.extract_frames_in_range(
                    video_path, start, end, interval=2.0
                )
                all_frames[i] = frames
            logger.info(f"Extracted frames for {len(chunks)} chunks")

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

            for chunk_idx, (start, end) in enumerate(chunks):
                frames = all_frames.get(chunk_idx, [])
                frame_images = [
                    frame_extractor.frame_to_base64(f) for _, f in frames
                ]

                prompt = prompt_chain.build_single_prompt(
                    chunk_info=f"Chunk {chunk_idx}: {start:.1f}s — {end:.1f}s",
                    context_summary=context_window.get_context_summary(),
                )

                entries = await vlm_client.generate_json(
                    prompt=prompt,
                    images_base64=frame_images,
                )

                # Validate and fix
                if isinstance(entries, list):
                    valid, issues = validator.validate_sequence(entries)
                    if not valid:
                        entries = validator.fix_overlaps(entries)
                        entries = validator.reindex(entries)

                    srt_builder.add_entries(entries, chunk_offset=0.0)
                    context_window.add_chunk_result(chunk_idx, entries)

            # === Step 5: SRT Generation ===
            self._update_progress("srt_generation")
            srt_path = output_dir / f"{video_path.stem}_vi.srt"
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
            aligner = AudioAligner()
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
            final_output = output_path or output_dir / f"{video_path.stem}_dubbed.mp4"
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
