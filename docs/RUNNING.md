# Hướng dẫn chạy lần đầu

Tài liệu này hướng dẫn chi tiết cách chạy pipeline `auto-video-narration-vlm` từ checkout sạch trên máy Windows. Linux/macOS tương tự — chỉ khác bước cài tools native.

## 1. Prerequisites

| Tool | Tối thiểu | Kiểm tra |
|------|-----------|----------|
| Python | 3.10 | `python --version` |
| CUDA Toolkit | 12.1 | `nvidia-smi` |
| FFmpeg | 4.x | `ffmpeg -version` |
| Rubberband | 3.x (4.0 đã test OK) | `rubberband --version` |
| Git | 2.x | `git --version` |

GPU ≥ 4 GB VRAM. Test thực tế trên RTX 3050 4GB chạy được Qwen3.5-2B local.

## 2. Clone và setup môi trường

```powershell
git clone <repo-url>
cd auto-video-narration-vlm

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **Lưu ý:** Mọi lệnh Python trong tài liệu này giả định venv đã được activate (`.\.venv\Scripts\Activate.ps1`). Đừng chạy bằng system Python — sẽ thiếu dependencies.

## 3. Cấu hình `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Cần điền:

| Biến | Bắt buộc | Mô tả |
|------|----------|-------|
| `HF_TOKEN` | Có | Token HuggingFace cho VieNeu-TTS. Lấy tại https://huggingface.co/settings/tokens |
| `GEMINI_API_KEY` | Không (chỉ cần khi `--api`) | API key Gemini |
| `VLM_MODE` | Không | `local` (default) hoặc `api`. CLI flag `--api` overrides. |
| `QWEN_MODEL_PATH` | Không | Đường dẫn weights, mặc định `./models/qwen3.5-2b` |
| `RUBBERBAND_PATH` | Không | Đường dẫn `rubberband.exe` nếu không có trên PATH |

## 4. Tải model weights

```powershell
python scripts/download_models.py
```

Tải khoảng 5 GB:
- Qwen3.5-2B (~4 GB) → `./models/qwen3.5-2b/`
- VieNeu-TTS v2 Turbo (~700 MB GGUF + ~200 MB ONNX) → tải lazy lúc TTSClient khởi tạo lần đầu
- GLM-OCR 0.9B (optional, hiện chưa được tích hợp pipeline)

## 5. Cài tools native (Windows)

### FFmpeg

Tải tại https://www.gyan.dev/ffmpeg/builds/ (essentials build), giải nén ví dụ vào `C:\ffmpeg\`, thêm `C:\ffmpeg\bin\` vào user PATH:

```powershell
$dest = "C:\ffmpeg\bin"
$current = [Environment]::GetEnvironmentVariable("Path", "User")
if ($current -notlike "*$dest*") {
    [Environment]::SetEnvironmentVariable("Path", "$current;$dest", "User")
}
```

### Rubberband

Tải `rubberband-X.X.X-gpl-executable-windows.zip` tại https://breakfastquay.com/rubberband/, giải nén vào ví dụ `C:\Tools\rubberband\`, thêm vào PATH (giống FFmpeg). Hoặc đặt `RUBBERBAND_PATH` trong `.env`.

Mở terminal mới rồi verify cả hai:

```powershell
ffmpeg -version
rubberband --version
```

## 6. Chuẩn bị input

- 1 video tiếng Anh ngắn (1–2 phút lý tưởng để iterate nhanh) đặt vào `data/raw/sample.mp4`
- 1 file audio reference (3–10 s, mono, WAV bất kỳ sample rate) đặt vào `data/reference_audio/speaker.wav`

## 7. Chạy pipeline đầy đủ

```powershell
$env:PYTHONIOENCODING = "utf-8"  # Tránh UnicodeEncodeError trên Windows console

python scripts/run_pipeline.py `
    --video "data/raw/sample.mp4" `
    --ref-audio "data/reference_audio/speaker.wav"
```

Output mong đợi tại `output/sample/`:

```
output/sample/
├── sample_vi.srt              # Phụ đề tiếng Việt
├── sample_dubbed.mp4          # Video lồng tiếng Việt
└── work/                       # Intermediate artifacts
    ├── audio_chunks/
    │   ├── chunk_0000.wav     # TTS output từng segment
    │   └── ...
    ├── aligned_audio/
    │   ├── chunk_0000_stretched.wav   # Sau time-stretch
    │   └── ...
    └── merged_audio.wav       # Track audio cuối (đã pad silence tới full duration)
```

Thời gian ước tính trên GTX 3050 4GB cho video 60s:
- M1 (VLM extract): 1–2 phút (chủ yếu do Qwen3.5-2B inference + global summary pass)
- M2 (TTS): 20–40 giây (tuỳ số segment)
- M3 (sync + render): 1–2 giây

## 8. Kiểm tra từng module riêng lẻ (debug)

Pipeline đầy đủ fail thì debug theo thứ tự:

### Bước 1: M1 (Video → SRT)

```powershell
python scripts/run_vlm_extract.py --video "data/raw/sample.mp4" --output "output/sample"
```

Output: `output/sample/<video_stem>.srt`. Mở bằng Notepad, kiểm tra:
- Có entries với timestamp `HH:MM:SS,mmm`?
- Translated text là tiếng Việt, không phải tiếng Anh?
- Last entry's end timestamp ≤ video duration?

### Bước 2: M2 (SRT → audio chunks)

```powershell
python scripts/test_m1_m2_integration.py `
    --srt "output/sample/<video_stem>.srt" `
    --ref-audio "data/reference_audio/speaker.wav" `
    --output-dir "debug_output/m1_m2_integration"
```

Output: `debug_output/m1_m2_integration/audio_chunks/chunk_NNNN.wav` — mỗi SRT entry tạo 1 file. Check:
- Số file == số SRT entries?
- Mở 1 file random, có audio tiếng Việt với giọng giống `speaker.wav`?

### Bước 3: M3 (chunks + SRT → dubbed video)

```powershell
python scripts/test_m3_sync.py `
    --srt "output/sample/<video_stem>.srt" `
    --audio-dir "debug_output/m1_m2_integration/audio_chunks" `
    --video "data/raw/sample.mp4" `
    --output-dir "debug_output/m3_test_run"
```

Output: `debug_output/m3_test_run/<video_stem>_vi.mp4`. Check:
- Duration == video gốc?
- Audio là tiếng Việt, video là frame gốc?

## 9. Troubleshooting

### `UnicodeEncodeError: 'charmap' codec can't encode character '═'`
Console Windows mặc định cp1252. Fix:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### `ModuleNotFoundError: No module named 'numpy'` (hoặc lib khác)
Chạy bằng system Python thay vì venv. Activate trước:
```powershell
.\.venv\Scripts\Activate.ps1
```

### `FileNotFoundError: [WinError 2]` khi gọi rubberband
Rubberband không có trên PATH. Hoặc:
- Set `RUBBERBAND_PATH=C:\Tools\rubberband\rubberband.exe` trong `.env`
- Hoặc thêm folder vào user PATH (xem mục 5), mở terminal mới

### `CUDA out of memory`
- Đóng tab browser / app khác chiếm GPU
- Đổi sang model nhỏ hơn: trong `.env`, set `QWEN_MODEL_PATH=./models/qwen3.5-0.8b` (cần download trước)
- Hoặc dùng API mode: `python scripts/run_pipeline.py --video ... --api`

### Model download stall hoặc timeout
HuggingFace rate limit hoặc mạng yếu. Chạy lại `python scripts/download_models.py` (sẽ resume từ chỗ dở).

### Video output không phát được / im lặng hoàn toàn
Trace từng bước:
1. `output/sample/work/audio_chunks/chunk_0000.wav` mở được không? Nếu không → M2 fail.
2. `output/sample/work/aligned_audio/chunk_0000_stretched.wav` có không? Nếu không → AudioAligner fail (kiểm tra rubberband).
3. `output/sample/work/merged_audio.wav` có duration ≈ video gốc không? Nếu ngắn hơn → bug merge_audio_segments.
4. Final video có duration ≈ audio không? Nếu lệch → bug render_final_video.

### Phụ đề SRT chỉ phủ phần đầu video, phần sau im lặng
Đây là known issue của Phase 2.A — VLM (đặc biệt Qwen3.5-2B local) đôi khi sinh ít entries hơn nội dung thực, hoặc compress hết narration vào timeline ngắn. Sẽ cải thiện ở Phase 2.C (timestamp accuracy). Phần video không có narration sẽ giữ silence trong audio dub.

### Audio TTS dài hơn slot SRT, bị truncate
Tiếng Việt thường dài hơn tiếng Anh ~30%. Nếu rate stretch_to_fit < 0.5 (audio TTS dài hơn 2× slot), TimeStretcher fallback truncate (cắt cuối). Cải thiện bằng cách:
- VLM tạo SRT slot dài hơn (sửa prompt)
- Hoặc giảm độ dài translation (sửa prompt)

## 10. Cấu trúc artifacts tham khảo

```
output/sample/
├── sample.srt                                     # M1 output (Vietnamese subtitles)
├── sample_dubbed.mp4                              # FINAL: dubbed video
├── chunk_00_raw_response.txt                      # VLM raw response (debug)
├── extraction_results.json                        # M1 metadata
├── frames/                                         # Saved frames per chunk
│   ├── chunk_00_global_00.jpg
│   └── ...
├── global_summary.json                            # Pass 0 output
└── work/
    ├── audio_chunks/
    │   ├── chunk_0000.wav                         # M2 raw TTS output (24kHz)
    │   └── ...
    ├── aligned_audio/
    │   ├── chunk_0000_stretched.wav               # After rubberband stretch
    │   └── ...
    └── merged_audio.wav                            # Final audio track (full duration, padded with silence)
```
