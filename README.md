# 🎬 Video Dubbing Vietnamese

> Hệ thống tự động lồng tiếng Việt cho video sử dụng Vision Language Model + Voice Cloning

## Tổng quan

Pipeline tự động chuyển đổi video tiếng Anh sang tiếng Việt với giọng clone chất lượng cao:

1. **Module VLM** — Trích xuất text từ video → dịch sang tiếng Việt → tạo file `.srt`
2. **Module Voice Cloning** — Clone giọng nói từ audio mẫu → sinh audio tiếng Việt
3. **Module Audio Sync** — Đồng bộ audio vào video theo timestamp
4. **Web UI** — Giao diện web demo toàn bộ pipeline

## Kiến trúc Pipeline

```
Video Input ──► Scene Detection ──► Frame Extraction ──► GLM-OCR
                                                            │
                                                    VLM (Gemini/Qwen2-VL)
                                                            │
                                                     Vietnamese SRT
                                                            │
                            Audio Reference ──► Voice Cloning (F5-TTS)
                                                            │
                                                     Audio Chunks
                                                            │
                                                  Time Stretch + Align
                                                            │
                                                     FFmpeg Merge
                                                            │
                                                    Dubbed Video Output
```

## Yêu cầu hệ thống

- Python >= 3.10
- CUDA >= 12.1 (khuyến nghị GPU 8GB+ VRAM)
- FFmpeg
- Rubberband CLI

## Cài đặt

```bash
# Clone project
git clone <repo-url>
cd video-dubbing-vi

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Cài đặt dependencies
pip install -r requirements.txt

# Copy env config
cp .env.example .env
# Chỉnh sửa .env với API key và cấu hình

# Download model weights
python scripts/download_models.py
```

## Sử dụng

### CLI

```bash
# Process single video
python scripts/run_pipeline.py --video data/raw/sample.mp4 --ref-audio data/reference_audio/speaker.wav

# Run evaluation
python scripts/batch_eval.py --eval-dir data/eval_set --output docs/eval_results/
```

### Web UI

```bash
# Start server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Truy cập http://localhost:8000
```

### Docker

```bash
docker build -t video-dubbing-vi .
docker run --gpus all -p 8000:8000 video-dubbing-vi
```

## Cấu trúc thư mục

```
video-dubbing-vi/
├── src/                    # Core source code
│   ├── vlm/                # Module 1: VLM text extraction + translation
│   ├── tts/                # Module 2: Voice cloning
│   ├── sync/               # Module 3: Audio synchronization
│   ├── pipeline/           # Pipeline orchestrator
│   ├── evaluation/         # Evaluation metrics
│   └── utils/              # Shared utilities
├── app/                    # Module 4: Web UI (FastAPI + Gradio)
├── models/                 # Model weights (gitignored)
├── data/                   # Input/output data (gitignored)
├── tests/                  # Unit + integration tests
├── notebooks/              # Jupyter experiments
├── scripts/                # CLI helpers
├── configs/                # YAML configurations
└── docs/                   # Documentation
```

## Evaluation Metrics

| Metric | Tool | Mục đích |
|--------|------|----------|
| BLEU-4, chrF++ | sacrebleu | Đánh giá chất lượng dịch thuật |
| MOS | UTMOS / MOSNet | Đánh giá chất lượng giọng nói |
| Speaker Similarity | resemblyzer | Đánh giá độ giống giọng gốc |
| Sync Accuracy | librosa onset | Đánh giá độ chính xác đồng bộ |

## Công nghệ sử dụng

- **VLM**: Gemini 1.5 Flash/Pro (API), Qwen2-VL-7B (local)
- **OCR**: GLM-OCR 0.9B
- **TTS**: F5-TTS-Vietnamese, viXTTS-v2
- **Audio Processing**: FFmpeg, Rubberband, PyDub
- **Web**: FastAPI, Gradio
- **Evaluation**: sacrebleu, UTMOS, resemblyzer

## License

MIT
