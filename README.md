# Video Dubbing Vietnamese

> Hệ thống tự động lồng tiếng Việt cho video tiếng Anh sử dụng Vision-Language Model + Zero-shot Voice Cloning. Đề tài đồ án tốt nghiệp (DATN).

## Tổng quan

Pipeline 5 module tự động chuyển video tiếng Anh sang tiếng Việt với giọng clone từ audio mẫu:

1. **Module 1 (VLM)** — Trích xuất phụ đề tiếng Việt từ video tiếng Anh (Gemini API hoặc Qwen3.5-2B local)
2. **Module 2 (TTS)** — Zero-shot voice cloning bằng VieNeu-TTS v2 Turbo
3. **Module 3 (Sync)** — Time-stretch + align + render audio vào video gốc
4. **Module 4 (Pipeline)** — Orchestrator chạy 8 step end-to-end
5. **Module 5 (Evaluation)** — BLEU/chrF++, speaker similarity, sync accuracy
6. **Web UI** — FastAPI + Gradio giao diện demo (`/ui`)

## Kiến trúc pipeline

```
Video tiếng Anh
   │
   ▼
Scene Detection (PySceneDetect, threshold 27)
   │
   ▼
Frame Extraction (adaptive sampling, SSIM dedup 40-70%)
   │
   ▼
VLM Inference (Qwen3.5-2B local hoặc Gemini API)
   │  • 2-layer context: global summary + sliding window
   │  • Anti-hallucination prompts, absolute timestamps
   ▼
Phụ đề tiếng Việt (.srt)
   │
   ▼  Reference audio (3-10s) ────► VieNeu-TTS Voice Cloning
   │                                        │
   │                                        ▼
   │                               Audio chunks (24kHz mono)
   │                                        │
   ▼                                        ▼
   AudioAligner.calculate_deltas + TimeStretcher (rubberband, ratio 0.5-2.0)
                                            │
                                            ▼
                          FFmpegRenderer.merge (apad to full duration)
                                            │
                                            ▼
                          Mux với video gốc → Video lồng tiếng Việt (.mp4)
```

## Yêu cầu hệ thống

| Tool | Version | Ghi chú |
|------|---------|---------|
| Python | ≥ 3.10 | |
| CUDA | ≥ 12.1 | GPU ≥ 4 GB VRAM (Qwen3.5-2B). Bỏ qua nếu chỉ dùng Gemini API. |
| FFmpeg | ≥ 4.x | Linux: `apt install ffmpeg`. Windows: tải tại https://ffmpeg.org/, thêm `bin\` vào PATH. |
| Rubberband CLI | ≥ 3.x (4.0 đã test) | Linux: `apt install rubberband-cli`. Windows: tải tại https://breakfastquay.com/rubberband/, thêm folder chứa `rubberband.exe` vào PATH. |
| HuggingFace token | | Bắt buộc cho VieNeu-TTS. Lấy tại https://huggingface.co/settings/tokens |

## Cài đặt

### Linux / macOS

```bash
git clone <repo-url>
cd auto-video-narration-vlm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Chỉnh sửa với HF_TOKEN, GEMINI_API_KEY (nếu dùng API mode)
python scripts/download_models.py
```

### Windows (PowerShell)

```powershell
git clone <repo-url>
cd auto-video-narration-vlm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env  # Chỉnh sửa với HF_TOKEN, GEMINI_API_KEY (nếu dùng API mode)
python scripts/download_models.py
```

### Cài rubberband trên Windows

Tải `rubberband-X.X.X-gpl-executable-windows.zip` tại https://breakfastquay.com/rubberband/, giải nén vào ví dụ `C:\Tools\rubberband\`, thêm vào user PATH:

```powershell
$dest = "C:\Tools\rubberband"
$current = [Environment]::GetEnvironmentVariable("Path", "User")
if ($current -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable("Path", "$current;$dest", "User")
}
```

Mở terminal mới rồi verify: `rubberband --version`. Tuỳ chọn: thay vì sửa PATH, có thể đặt `RUBBERBAND_PATH=C:\Tools\rubberband\rubberband.exe` trong `.env`.

### Lưu ý Windows về Unicode

Các script in ký tự `═`, dấu tiếng Việt v.v. ra console. Nếu gặp `UnicodeEncodeError` (cp1252), set:

```powershell
$env:PYTHONIOENCODING = "utf-8"
```

trước khi chạy. Hoặc thêm `chcp 65001` vào profile.

## Sử dụng

### CLI — pipeline đầy đủ (khuyến nghị)

```powershell
python scripts/run_pipeline.py `
    --video "data/raw/sample.mp4" `
    --ref-audio "data/reference_audio/speaker.wav"
```

Output: `output/<stem>/<stem>_dubbed.mp4` + SRT + intermediate artifacts ở `output/<stem>/work/`.

### CLI — chỉ trích xuất SRT (không TTS)

```powershell
python scripts/run_vlm_extract.py --video "data/raw/sample.mp4"
# Mặc định: Qwen3.5-2B local. Thêm --api để dùng Gemini.
```

### CLI — debug từng module

```powershell
# M1+M2 (SRT → audio chunks)
python scripts/test_m1_m2_integration.py `
    --srt "output/sample/sample.srt" `
    --ref-audio "data/reference_audio/speaker.wav" `
    --output-dir "debug_output/m1_m2_integration"

# M3 (audio chunks + SRT → dubbed video)
python scripts/test_m3_sync.py `
    --srt "output/sample/sample.srt" `
    --audio-dir "debug_output/m1_m2_integration/audio_chunks" `
    --video "data/raw/sample.mp4" `
    --output-dir "debug_output/m3_test_run"
```

### Web UI

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
# http://localhost:8000  (Gradio mounted tại /ui)
```

Xem thêm [`docs/RUNNING.md`](docs/RUNNING.md) để có hướng dẫn chạy lần đầu chi tiết.

## Cấu trúc thư mục

```
auto-video-narration-vlm/
├── src/
│   ├── m1_vlm/              # Module 1: scene detect, frame extract, VLM, SRT
│   ├── m2_tts/              # Module 2: VieNeu-TTS voice cloning
│   ├── m3_sync/             # Module 3: align, time-stretch, FFmpeg render
│   ├── m4_pipeline/         # Module 4: orchestrator (8-step async)
│   ├── m5_evaluation/       # Module 5: metrics
│   └── utils/               # async / audio / file / video helpers
├── app/                     # FastAPI + Gradio Web UI
├── configs/                 # YAML defaults (default, gemini, qwen_local, tts)
├── data/                    # Input/output (gitignored)
│   ├── raw/                 # Video gốc
│   └── reference_audio/     # Audio reference cho voice cloning
├── debug_output/            # Intermediate artifacts (gitignored)
├── docs/                    # Documentation, progress reports, specs/plans
├── models/                  # Model weights (gitignored)
├── output/                  # Final SRT + dubbed video
├── scripts/                 # CLI entry points
└── tests/                   # pytest
```

## Trạng thái module

| Module | Trạng thái | Ghi chú |
|--------|-----------|---------|
| **M1: VLM Extraction** | ✅ Done | 8 components, 8 pytest files. Default: Qwen3.5-2B local. |
| **M2: TTS Voice Cloning** | ✅ Done | VieNeu-TTS v2 Turbo, 24 test case, bridge SRT→TTS. |
| **M3: Audio Sync & Render** | ✅ Done (Phase 2.A) | AudioAligner + TimeStretcher (rubberband 4.0) + FFmpegRenderer. End-to-end tested. |
| **M4: Pipeline Orchestrator** | ✅ Done (Phase 2.A) | `PipelineRunner` chạy 8 step, đã fix chunk_offset + sample_rate. |
| **M5: Evaluation** | ⚠️ Partial | BLEU/chrF++, Speaker Similarity, Sync Accuracy hoạt động. MOS chưa implement. |
| **Web App** | ⚠️ Scaffolding | FastAPI + Gradio — chưa test thực tế. |

## Evaluation metrics

| Metric | Tool | Mục đích |
|--------|------|----------|
| BLEU-4, chrF++ | sacrebleu | Chất lượng dịch thuật |
| MOS | UTMOS / MOSNet | Chất lượng giọng nói (chưa implement) |
| Speaker Similarity | resemblyzer | Độ giống giọng gốc |
| Sync Accuracy | librosa onset | Độ chính xác đồng bộ |

## Công nghệ

- **VLM**: Qwen3.5-2B (default local, ~4 GB VRAM, 262K context), Qwen2.5-VL-3B (legacy), Gemini 1.5 Flash (API)
- **TTS**: VieNeu-TTS v2 Turbo (GGUF + ONNX, không cần PyTorch lúc runtime)
- **OCR**: GLM-OCR 0.9B (đã code, chưa tích hợp pipeline)
- **Audio**: FFmpeg, Rubberband CLI, soundfile, pydub
- **Web**: FastAPI, Gradio
- **Evaluation**: sacrebleu, resemblyzer, librosa

## Tài liệu liên quan

- [`docs/RUNNING.md`](docs/RUNNING.md) — Hướng dẫn chạy lần đầu chi tiết
- [`docs/architecture.md`](docs/architecture.md) — Chi tiết kiến trúc
- [`docs/api_reference.md`](docs/api_reference.md) — API reference
- [`docs/prompt_catalog.md`](docs/prompt_catalog.md) — Catalog prompt VLM
- [`docs/Phase1_Progress_Report.md`](docs/Phase1_Progress_Report.md) — Progress report Phase 1 (English)
- [`docs/BaoCao_TienDo_Phase1.md`](docs/BaoCao_TienDo_Phase1.md) — Báo cáo tiến độ Phase 1 (Vietnamese)

## License

MIT
