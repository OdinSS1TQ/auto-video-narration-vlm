# Video Dubbing Vietnamese

> Hệ thống tự động lồng tiếng Việt cho video tiếng Anh sử dụng Vision-Language Model + Zero-shot Voice Cloning. Đề tài đồ án tốt nghiệp (DATN).

## Tổng quan

Pipeline 5 module tự động chuyển video tiếng Anh sang tiếng Việt với giọng clone từ audio mẫu:

1. **Module 1 (VLM)** — Trích xuất phụ đề tiếng Việt từ video tiếng Anh (Gemini API hoặc Qwen3.5-2B local)
2. **Module 2 (TTS)** — Zero-shot voice cloning bằng VieNeu-TTS v2 Turbo
3. **Module 3 (Sync)** — Retime + budgeted time-stretch + align + render audio vào video gốc
4. **Module 4 (Pipeline)** — Orchestrator chạy 8 step end-to-end, 2 chế độ `vlm` và `ocr`
5. **Module 5 (Evaluation)** — BLEU/chrF++, speaker similarity, sync accuracy, caption drift
6. **Web App** — FastAPI REST API + React (Vite) frontend (`/studio`) + Gradio demo (`/ui`)

### Hai chế độ pipeline

- **`--mode vlm`** (mặc định) — VLM đọc frame và tự sinh phụ đề + timestamp. Phù hợp video không có caption cháy sẵn.
- **`--mode ocr`** — GLM-OCR đọc caption tiếng Anh cháy sẵn để lấy timestamp chính xác, sau đó classifier VLM lọc narration vs on-screen text, rồi dịch. Phù hợp video có phụ đề/caption burned-in.

## Kiến trúc pipeline

```
Video tiếng Anh
   │
   ├─── mode vlm ─────────────────────┐        ├─── mode ocr ──────────────────────┐
   │                                  │        │                                    │
   ▼                                  │        ▼                                    │
Scene Detection (PySceneDetect, 27)   │   CaptionTimeline (sample 3fps, crop band) │
   │                                  │        │  • GLM-OCR đọc caption tiếng Anh   │
   ▼                                  │        │  • gộp/khử trùng lặp segment      │
Frame Extraction (adaptive, SSIM      │        ▼                                    │
   dedup 40-70%)                      │   Narration classifier (VLM lọc screen)    │
   │                                  │        │                                    │
   ▼                                  │        ▼                                    │
VLM Inference (Qwen3.5-2B / Gemini)   │   VLM translation (EN → VI)                │
   │  • 2-layer context               │        │                                    │
   │  • anti-hallucination            │        │                                    │
   └──────────────┬───────────────────┘        └────────────────┬───────────────────┘
                  ▼                                              ▼
       EntryRetimer (reconstruct timing từ chunk duration + CPS model)
                  │
                  ▼
       Phụ đề tiếng Việt (.srt)
                  │
                  ▼  Reference audio (3-10s) ────► VieNeu-TTS Voice Cloning
                  │                                        │
                  │                                        ▼
                  │                               Audio chunks (24kHz mono)
                  │                                        │
                  ▼                                        ▼
   AudioAligner (budgeted stretch + slip cascade) + TimeStretcher (rubberband, ≤1.25x)
                                            │
                                            ▼
                          FFmpegRenderer.merge (amix normalize=0, apad full duration)
                                            │
                                            ▼
                          Mux với video gốc → Video lồng tiếng Việt (.mp4)
```

## Yêu cầu hệ thống

| Tool | Version | Ghi chú |
|------|---------|---------|
| Python | ≥ 3.10 | |
| Node.js | ≥ 18 | Chỉ cần nếu build React frontend (`/studio`). Bỏ qua nếu chỉ dùng CLI hoặc Gradio UI. |
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
    --ref-audio "data/reference_audio/speaker.wav" `
    --mode vlm            # hoặc: ocr (video có caption cháy sẵn)
```

Các cờ hữu ích: `--mode {vlm,ocr}`, `--vlm-mode {api,local}`, `--tts-engine {f5-tts,vixtts}`, `--output <path>`, `--config <yaml>`, `--log-level`.

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

### Web App (FastAPI + React + Gradio)

```powershell
# 1. (Tuỳ chọn) build React frontend — assets vào frontend/dist, phục vụ tại /studio
cd frontend
npm install
npm run build
cd ..

# 2. Chạy backend (phục vụ cả API, React, Gradio)
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `http://localhost:8000/` — API info + links
- `http://localhost:8000/studio` — React frontend (AI Dub Studio)
- `http://localhost:8000/ui` — Gradio demo
- `http://localhost:8000/docs` — Swagger / OpenAPI docs

Dev frontend với hot-reload (proxy `/api` → `:8000`):

```powershell
cd frontend
npm run dev   # http://localhost:3000
```

#### REST API endpoints (prefix `/api`)

| Method | Path | Mô tả |
|--------|------|-------|
| POST | `/api/upload/video` | Upload video |
| POST | `/api/upload/audio` | Upload reference audio |
| POST | `/api/process` | Bắt đầu job dubbing (background task) |
| DELETE | `/api/process/{job_id}` | Huỷ job |
| GET | `/api/status/{job_id}` | Trạng thái + progress của job |
| GET | `/api/jobs` | Liệt kê job (filter + pagination) |
| GET | `/api/result/{job_id}` | Tải video kết quả |
| GET | `/api/result/{job_id}/srt` | Tải SRT |
| GET | `/api/result/{job_id}/report` | Metadata report của job |
| GET | `/api/tts/voices` | Danh sách giọng preset VieNeu |
| GET | `/api/health` | Health check (ffmpeg/rubberband/GPU/disk) |
| GET | `/api/config` | Cấu hình pipeline hiện tại |

### Docker

```bash
docker build -t video-dubbing-vi .
docker run --gpus all -p 8000:8000 \
    -e HF_TOKEN=<token> -e GEMINI_API_KEY=<key> \
    video-dubbing-vi
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
├── app/                     # Backend web
│   ├── routers/             #   REST API (upload, process, status, download, health, config, tts)
│   ├── schemas/             #   Pydantic request/response models
│   ├── gradio_ui.py         #   Gradio demo (/ui)
│   └── main.py              #   FastAPI entrypoint (mount API + React /studio + Gradio /ui)
├── frontend/                # React 19 + Vite + Tailwind SPA (build → frontend/dist, phục vụ tại /studio)
├── Dockerfile               # CUDA 12.1 runtime + ffmpeg + rubberband
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
| **M1: VLM Extraction** | ✅ Done | 8 components, VLM mode + OCR mode (GLM-OCR CaptionTimeline + narration classifier). Default: Qwen3.5-2B local. |
| **M2: TTS Voice Cloning** | ✅ Done | VieNeu-TTS v2 Turbo, 24 test case, bridge SRT→TTS. |
| **M3: Audio Sync & Render** | ✅ Done | EntryRetimer + AudioAligner (budgeted stretch + slip cascade) + TimeStretcher (rubberband ≤1.25x) + FFmpegRenderer. |
| **M4: Pipeline Orchestrator** | ✅ Done | `PipelineRunner` chạy 8 step, 2 mode (`vlm`/`ocr`), đã fix chunk_offset + sample_rate. |
| **M5: Evaluation** | ✅ Done | BLEU/chrF++, Speaker Similarity, Sync Accuracy, Caption Drift + report JSON/Markdown/CSV. MOS đánh giá chủ quan (survey), không dùng ước lượng tự động. |
| **Web App** | ✅ Done | FastAPI REST API + React (Vite) frontend `/studio` + Gradio `/ui`. |

## Evaluation metrics

| Metric | Tool | Mục đích |
|--------|------|----------|
| BLEU-4, chrF++ | sacrebleu | Chất lượng dịch thuật |
| MOS | Survey chủ quan | Chất lượng giọng nói (đánh giá thủ công, không tự động) |
| Speaker Similarity | resemblyzer | Độ giống giọng gốc (pre/post-align + merged) |
| Sync Accuracy | librosa onset | Độ chính xác đồng bộ |
| Caption Drift | fuzzy match | Sai lệch timestamp OCR vs ground-truth |

Chạy: `python scripts/run_evaluation.py` (xuất JSON + Markdown + CSV).

## Công nghệ

- **VLM**: Qwen3.5-2B (default local, ~4 GB VRAM, 262K context), Qwen2.5-VL-3B (legacy), Gemini 1.5 Flash (API)
- **TTS**: VieNeu-TTS v2 Turbo (GGUF + ONNX, không cần PyTorch lúc runtime)
- **OCR**: GLM-OCR 0.9B (đã tích hợp trong `--mode ocr`: CaptionTimeline + narration classifier)
- **Audio**: FFmpeg, Rubberband CLI, soundfile, pydub
- **Backend**: FastAPI, Gradio, uvicorn
- **Frontend**: React 19, Vite 6, Tailwind CSS 4, TypeScript
- **Evaluation**: sacrebleu, resemblyzer, librosa

## Tài liệu liên quan

- [`docs/RUNNING.md`](docs/RUNNING.md) — Hướng dẫn chạy lần đầu chi tiết
- [`docs/architecture.md`](docs/architecture.md) — Chi tiết kiến trúc
- [`docs/api_reference.md`](docs/api_reference.md) — API reference
- [`docs/api_spec.md`](docs/api_spec.md) — Spec REST API (endpoints, schema)
- [`docs/prompt_catalog.md`](docs/prompt_catalog.md) — Catalog prompt VLM
- [`docs/XuLy_Timestamp_GLM_OCR_CaptionTimeline.md`](docs/XuLy_Timestamp_GLM_OCR_CaptionTimeline.md) — Xử lý timestamp OCR
- [`docs/Phase1_Progress_Report.md`](docs/Phase1_Progress_Report.md) — Progress report Phase 1 (English)
- [`docs/BaoCao_TienDo_Phase1.md`](docs/BaoCao_TienDo_Phase1.md) — Báo cáo tiến độ Phase 1 (Vietnamese)
- [`docs/Phase2_Progress_Report.md`](docs/Phase2_Progress_Report.md) — Progress report Phase 2 (English)
- [`docs/BaoCao_TienDo_Phase2.md`](docs/BaoCao_TienDo_Phase2.md) — Báo cáo tiến độ Phase 2 (Vietnamese)

## License

MIT
