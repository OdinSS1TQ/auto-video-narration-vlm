"""
Gradio UI — Interactive interface for video dubbing pipeline.

Flow: Upload video → Upload reference audio → Select VLM mode → Process → Preview → Download
"""

import gradio as gr
from pathlib import Path
from typing import Optional


def create_ui() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="Video Dubbing Vietnamese",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 🎬 Video Dubbing Vietnamese
            ### Hệ thống tự động lồng tiếng Việt cho video

            **Pipeline**: Video Input → VLM Text Extraction → Translation → Voice Cloning → Audio Sync → Output
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### 📤 Input")

                video_input = gr.Video(
                    label="Video gốc (tiếng Anh)",
                    interactive=True,
                )

                audio_ref = gr.Audio(
                    label="Audio mẫu giọng (3-10 giây)",
                    type="filepath",
                )

                with gr.Row():
                    vlm_mode = gr.Radio(
                        choices=["API (Gemini)", "Local (Qwen2-VL)"],
                        value="API (Gemini)",
                        label="VLM Mode",
                    )

                    tts_engine = gr.Radio(
                        choices=["F5-TTS", "viXTTS"],
                        value="F5-TTS",
                        label="TTS Engine",
                    )

                process_btn = gr.Button(
                    "🚀 Bắt đầu lồng tiếng",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### 📥 Output")

                # Progress
                progress_text = gr.Textbox(
                    label="Tiến trình",
                    interactive=False,
                    lines=2,
                )

                progress_bar = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=0,
                    label="Progress",
                    interactive=False,
                )

                # Output video
                video_output = gr.Video(
                    label="Video lồng tiếng",
                    interactive=False,
                )

                # SRT download
                srt_output = gr.File(
                    label="Download SRT",
                )

        # Metrics section
        with gr.Accordion("📊 Metrics", open=False):
            with gr.Row():
                bleu_score = gr.Number(label="BLEU-4", precision=2)
                mos_score = gr.Number(label="MOS", precision=2)
                similarity_score = gr.Number(label="Speaker Similarity", precision=4)
                sync_delay = gr.Number(label="Avg Sync Delay (s)", precision=4)

        # Processing logic
        async def process_video(video, audio, vlm, tts):
            """Run the dubbing pipeline."""
            if not video:
                return None, None, "❌ Vui lòng upload video", 0

            yield None, None, "🔄 Đang xử lý...", 10

            try:
                from src.m4_pipeline.runner import PipelineRunner
                from src.m4_pipeline.config import PipelineConfig

                config = PipelineConfig()
                runner = PipelineRunner(config)

                # TODO: Connect progress callback
                # runner.set_progress_callback(...)

                result = await runner.run(
                    video_path=video,
                    reference_audio_path=audio,
                )

                yield (
                    result.get("output_path"),
                    result.get("srt_path"),
                    f"✅ Hoàn thành trong {result.get('elapsed_seconds', 0):.1f}s",
                    100,
                )

            except Exception as e:
                yield None, None, f"❌ Lỗi: {str(e)}", 0

        process_btn.click(
            fn=process_video,
            inputs=[video_input, audio_ref, vlm_mode, tts_engine],
            outputs=[video_output, srt_output, progress_text, progress_bar],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
