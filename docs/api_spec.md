# API Specification — Video Dubbing Vietnamese

> Phiên bản: 1.0  
> Cập nhật: 2026-06-05  
> Base URL: `http://localhost:8000/api`

---

## 1. Tổng Quan

Hệ thống cung cấp **11 REST API endpoints** chia thành **6 nhóm** phục vụ luồng: Upload file → Khởi chạy pipeline → Theo dõi tiến trình → Tải kết quả.

### Quy Ước Chung

| Thuộc tính | Giá trị |
|---|---|
| **Protocol** | HTTP/HTTPS |
| **Content-Type (request)** | `application/json` hoặc `multipart/form-data` (upload) |
| **Content-Type (response)** | `application/json` (trừ endpoint download trả binary) |
| **Encoding** | UTF-8 |
| **Authentication** | Không yêu cầu (hệ thống demo nội bộ) |
| **Job Storage** | In-memory dict (phù hợp demo, restart server mất dữ liệu job) |
| **Concurrent Jobs** | Tối đa **5 jobs** đồng thời |

### Giới Hạn Upload

| Loại file | Độ dài tối đa | Dung lượng tối đa | Ghi chú |
|---|---|---|---|
| Video | ~5 phút | **200 MB** | Ước tính: 1080p H.264 ~5MB/phút → ~25MB; 4K hoặc bitrate cao có thể đến ~200MB |
| Audio mẫu | 3–10 giây | **10 MB** | Mono/stereo, ≥16kHz |

### Defaults Quan Trọng

| Cấu hình | Giá trị mặc định | Ghi chú |
|---|---|---|
| `pipeline_mode` | `"ocr"` | OCR-driven pipeline (GLM-OCR đọc caption → VLM dịch) |
| `vlm_mode` | `"local"` | Qwen3.5-2B local (không cần Gemini API key) |
| `tts_engine` | `"vieneu"` | VieNeu-TTS v2 Turbo |

---

## 2. Danh Sách Endpoints

| # | Method | Endpoint | Mô tả | Trạng thái |
|---|---|---|---|---|
| 1 | `POST` | `/api/upload/video` | Upload video gốc (tiếng Anh) | Cải thiện |
| 2 | `POST` | `/api/upload/audio` | Upload audio mẫu cho voice cloning | Cải thiện |
| 3 | `POST` | `/api/process` | Khởi chạy pipeline dubbing (background) | Cải thiện |
| 4 | `DELETE` | `/api/process/{job_id}` | Hủy job đang chạy | **Mới** |
| 5 | `GET` | `/api/status/{job_id}` | Kiểm tra tiến trình job | Cải thiện |
| 6 | `GET` | `/api/jobs` | Liệt kê tất cả jobs | **Mới** |
| 7 | `GET` | `/api/result/{job_id}` | Download video đã lồng tiếng | Giữ nguyên |
| 8 | `GET` | `/api/result/{job_id}/srt` | Download file SRT phụ đề | Giữ nguyên |
| 9 | `GET` | `/api/result/{job_id}/report` | Lấy metadata kết quả (JSON) | **Mới** |
| 10 | `GET` | `/api/health` | Health check hệ thống | **Mới** |
| 11 | `GET` | `/api/config` | Xem cấu hình pipeline hiện tại | **Mới** |

> **Lưu ý**: Không có endpoint Evaluation (đánh giá chất lượng dịch BLEU/chrF++). Việc đánh giá SRT sẽ được thực hiện **thủ công** bởi người dùng dựa trên video.

---

## 3. Chi Tiết Từng Endpoint

---

### 3.1. `POST /api/upload/video`

Upload video gốc (tiếng Anh) lên server để xử lý.

**Request**:
- Content-Type: `multipart/form-data`
- Field: `file` (required) — File video

**Validation**:
- Định dạng: `.mp4`, `.avi`, `.mkv`, `.webm`, `.mov`
- Dung lượng: ≤ 200 MB
- Độ dài: khuyến nghị ≤ 5 phút

**Response** `200 OK`:
```json
{
  "file_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "sample_video.mp4",
  "path": "./data/uploads/a1b2c3d4-e5f6-7890-abcd-ef1234567890.mp4",
  "size_mb": 45.2
}
```

**Errors**:
| Code | Condition |
|---|---|
| `400` | Định dạng file không hỗ trợ |
| `413` | File vượt quá 200 MB |
| `500` | Lỗi hệ thống khi lưu file |

---

### 3.2. `POST /api/upload/audio`

Upload file audio mẫu giọng nói cho zero-shot voice cloning.

**Request**:
- Content-Type: `multipart/form-data`
- Field: `file` (required) — File audio

**Validation**:
- Định dạng: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`
- Dung lượng: ≤ 10 MB
- Khuyến nghị: 3–10 giây, mono/stereo, ≥ 16kHz sample rate

**Response** `200 OK`:
```json
{
  "file_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
  "filename": "speaker_sample.wav",
  "path": "./data/uploads/b2c3d4e5-f6a7-8901-bcde-f12345678901.wav",
  "size_mb": 0.5
}
```

**Errors**:
| Code | Condition |
|---|---|
| `400` | Định dạng file không hỗ trợ |
| `413` | File vượt quá 10 MB |

---

### 3.3. `POST /api/process`

Khởi chạy pipeline dubbing ở background. Trả về `job_id` ngay lập tức.

**Request**:
- Content-Type: `application/json`

**Request Body**:
```json
{
  "video_path": "./data/uploads/uuid.mp4",
  "audio_path": "./data/uploads/uuid.wav",
  "vlm_mode": "local",
  "tts_engine": "vieneu",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "keep_original_audio": false,
  "pipeline_mode": "ocr"
}
```

**Các field**:

| Field | Type | Required | Default | Mô tả |
|---|---|---|---|---|
| `video_path` | `string` | ✅ | — | Đường dẫn video (lấy từ `path` trong response upload) |
| `audio_path` | `string` | ✅ | — | Đường dẫn audio mẫu (lấy từ `path` trong response upload) |
| `vlm_mode` | `string` | ❌ | `"local"` | `"local"` = Qwen3.5-2B, `"api"` = Gemini API |
| `tts_engine` | `string` | ❌ | `"vieneu"` | Engine TTS: `"vieneu"` (VieNeu-TTS v2 Turbo) |
| `source_lang` | `string` | ❌ | `"English"` | Ngôn ngữ nguồn của video |
| `target_lang` | `string` | ❌ | `"Vietnamese"` | Ngôn ngữ đích (lồng tiếng) |
| `keep_original_audio` | `bool` | ❌ | `false` | `true` = giữ audio gốc ở âm lượng nhỏ nền |
| `pipeline_mode` | `string` | ❌ | `"ocr"` | `"ocr"` = OCR-driven, `"vlm"` = VLM-driven |

**Response** `202 Accepted`:
```json
{
  "job_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "status": "queued",
  "message": "Dubbing job queued. Use GET /api/status/{job_id} to track progress."
}
```

**8 Bước Pipeline** (theo thứ tự):

| Step | Tên | Mô tả |
|---|---|---|
| 1 | `caption_ocr_timeline` | GLM-OCR đọc caption burned-in từ video (mode OCR) |
| 2 | `vlm_classify_narration` | VLM phân loại narration vs screen-text |
| 3 | `vlm_global_summary` | VLM tạo tóm tắt tổng quan video |
| 4 | `vlm_translate_segments` | VLM dịch từng segment Anh → Việt |
| 5 | `srt_generation` | Xây dựng file SRT phụ đề tiếng Việt |
| 6 | `voice_cloning` | VieNeu-TTS zero-shot voice cloning |
| 7 | `audio_alignment` | Time-stretch + align audio với timeline |
| 8 | `video_rendering` | FFmpeg merge audio + video → output |

> Nếu `pipeline_mode = "vlm"`, 8 step sẽ là: `scene_detection` → `frame_extraction` → `ocr_extraction` → `vlm_translation` → `srt_generation` → `voice_cloning` → `audio_alignment` → `video_rendering`.

**Errors**:
| Code | Condition |
|---|---|
| `400` | Thiếu `video_path` hoặc `audio_path` |
| `404` | File video/audio không tồn tại trên server |
| `422` | Giá trị `vlm_mode` hoặc `tts_engine` không hợp lệ |
| `429` | Đã đạt giới hạn 5 jobs đồng thời |

---

### 3.4. `DELETE /api/process/{job_id}`

Hủy một job đang chạy hoặc đang queued.

**URL Params**: `job_id` (string, UUID)

**Response** `200 OK`:
```json
{
  "job_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "status": "cancelled",
  "message": "Job cancelled successfully."
}
```

**Errors**:
| Code | Condition |
|---|---|
| `404` | Job ID không tồn tại |
| `400` | Job đã hoàn thành hoặc đã bị hủy (không thể hủy nữa) |

---

### 3.5. `GET /api/status/{job_id}`

Kiểm tra trạng thái và tiến trình của một job.

**URL Params**: `job_id` (string, UUID)

**Response** `200 OK`:
```json
{
  "job_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "status": "processing",
  "progress": {
    "step": "vlm_translate_segments",
    "step_name_vi": "Dịch thuật bằng VLM",
    "current": 4,
    "total": 8,
    "percent": 50
  },
  "created_at": "2026-06-05T13:00:00+07:00",
  "error": null
}
```

**Giá trị `status`**:

| Status | Mô tả |
|---|---|
| `queued` | Đã tạo, chờ xử lý |
| `processing` | Đang chạy pipeline |
| `completed` | Hoàn thành thành công |
| `failed` | Lỗi xảy ra trong pipeline |
| `cancelled` | Đã bị hủy bởi user |

**`progress.step_name_vi`** — Tên bước bằng tiếng Việt:

| Step key | Tên tiếng Việt |
|---|---|
| `caption_ocr_timeline` | Trích xuất caption bằng OCR |
| `vlm_classify_narration` | Phân loại lời thoại |
| `vlm_global_summary` | Tóm tắt tổng quan video |
| `vlm_translate_segments` | Dịch thuật bằng VLM |
| `srt_generation` | Tạo file phụ đề SRT |
| `voice_cloning` | Nhân bản giọng nói |
| `audio_alignment` | Căn chỉnh audio |
| `video_rendering` | Render video cuối cùng |

**Errors**:
| Code | Condition |
|---|---|
| `404` | Job ID không tồn tại |

---

### 3.6. `GET /api/jobs`

Liệt kê tất cả jobs với filter và phân trang.

**Query Params**:

| Param | Type | Default | Mô tả |
|---|---|---|---|
| `status` | `string` | `null` (tất cả) | Filter theo status: `queued`, `processing`, `completed`, `failed`, `cancelled` |
| `limit` | `int` | `20` | Số lượng kết quả tối đa |
| `offset` | `int` | `0` | Vị trí bắt đầu (phân trang) |

**Response** `200 OK`:
```json
{
  "jobs": [
    {
      "job_id": "uuid-1",
      "status": "completed",
      "progress": {
        "step": "video_rendering",
        "current": 8,
        "total": 8,
        "percent": 100
      },
      "created_at": "2026-06-05T12:00:00+07:00"
    },
    {
      "job_id": "uuid-2",
      "status": "processing",
      "progress": {
        "step": "voice_cloning",
        "current": 6,
        "total": 8,
        "percent": 75
      },
      "created_at": "2026-06-05T13:00:00+07:00"
    }
  ],
  "total": 2,
  "limit": 20,
  "offset": 0
}
```

---

### 3.7. `GET /api/result/{job_id}`

Download file video đã lồng tiếng Việt.

**URL Params**: `job_id` (string, UUID)

**Response** `200 OK`:
- Content-Type: `video/mp4`
- Content-Disposition: `attachment; filename="<stem>_dubbed.mp4"`
- Body: Binary file

**Errors**:
| Code | Condition |
|---|---|
| `400` | Job chưa hoàn thành (`status ≠ completed`) |
| `404` | Job không tồn tại hoặc file output bị mất |

---

### 3.8. `GET /api/result/{job_id}/srt`

Download file SRT phụ đề tiếng Việt.

**URL Params**: `job_id` (string, UUID)

**Response** `200 OK`:
- Content-Type: `text/plain; charset=utf-8`
- Content-Disposition: `attachment; filename="<stem>_vi.srt"`
- Body: Nội dung SRT chuẩn

```
1
00:00:01,000 --> 00:00:04,500
Xin chào, hôm nay chúng ta sẽ tìm hiểu về...

2
00:00:05,000 --> 00:00:08,200
Đây là một chủ đề rất thú vị...
```

**Errors**:
| Code | Condition |
|---|---|
| `400` | Job chưa hoàn thành |
| `404` | Job không tồn tại hoặc file SRT bị mất |

---

### 3.9. `GET /api/result/{job_id}/report`

Lấy metadata chi tiết về kết quả xử lý của một job.

**URL Params**: `job_id` (string, UUID)

**Response** `200 OK`:
```json
{
  "job_id": "c3d4e5f6-a7b8-9012-cdef-123456789012",
  "video_path": "./data/uploads/uuid.mp4",
  "reference_audio": "./data/uploads/uuid.wav",
  "output_path": "./output/sample/sample_dubbed.mp4",
  "srt_path": "./output/sample/sample_vi.srt",
  "mode": "ocr",
  "elapsed_seconds": 142.3,
  "n_segments_raw": 28,
  "n_segments_after_classify": 25,
  "n_segments_merged": 22
}
```

**Errors**:
| Code | Condition |
|---|---|
| `400` | Job chưa hoàn thành |
| `404` | Job không tồn tại |

---

### 3.10. `GET /api/health`

Kiểm tra trạng thái hệ thống và các dependency.

**Response** `200 OK`:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-06-05T13:12:00+07:00",
  "dependencies": {
    "ffmpeg": {
      "available": true,
      "version": "6.1"
    },
    "rubberband": {
      "available": true,
      "version": "4.0.0"
    },
    "gpu": {
      "available": true,
      "name": "NVIDIA RTX 3060",
      "vram_mb": 12288
    }
  },
  "active_jobs": 1,
  "max_concurrent_jobs": 5,
  "disk_usage": {
    "uploads_mb": 234.5,
    "outputs_mb": 567.8
  }
}
```

**Trạng thái `status`**:
- `"healthy"` — Tất cả dependency OK
- `"degraded"` — Một số dependency thiếu (ví dụ: không có GPU → chỉ chạy được API mode)
- `"unhealthy"` — Dependency bắt buộc thiếu (ví dụ: không có FFmpeg)

---

### 3.11. `GET /api/config`

Xem cấu hình pipeline hiện tại. Ẩn thông tin nhạy cảm (API keys, HF tokens).

**Response** `200 OK`:
```json
{
  "vlm": {
    "mode": "local",
    "model_name": "qwen3.5-2b",
    "temperature": 0.2,
    "max_tokens": 4096
  },
  "tts": {
    "engine": "vieneu",
    "sample_rate": 24000,
    "vieneu_mode": "turbo",
    "backbone_device": "cuda",
    "codec_device": "cpu"
  },
  "pipeline": {
    "mode": "ocr",
    "chunk_duration_sec": 45.0,
    "scene_threshold": 27.0,
    "ssim_threshold": 0.85,
    "max_concurrent_jobs": 5
  },
  "sync": {
    "max_speedup": 1.25,
    "min_gap_sec": 0.1
  },
  "limits": {
    "max_video_size_mb": 200,
    "max_audio_size_mb": 10,
    "max_video_duration_min": 5,
    "max_concurrent_jobs": 5
  },
  "output": {
    "dir": "./data/outputs",
    "format": "mp4"
  }
}
```

---

## 4. Error Response Format

Tất cả error response tuân theo format chuẩn:

```json
{
  "detail": "Mô tả lỗi chi tiết bằng tiếng Anh"
}
```

Với FastAPI `HTTPException`, format tự động là:
```json
{
  "detail": "Job not found: <job_id>"
}
```

Các HTTP status code sử dụng:

| Code | Ý nghĩa | Khi nào dùng |
|---|---|---|
| `200` | OK | Request thành công |
| `202` | Accepted | Job được tạo, đang xử lý background |
| `400` | Bad Request | Input không hợp lệ, job chưa sẵn sàng |
| `404` | Not Found | Job hoặc file không tồn tại |
| `413` | Payload Too Large | File upload vượt giới hạn |
| `422` | Unprocessable Entity | Validation lỗi (Pydantic) |
| `429` | Too Many Requests | Đã đạt giới hạn 5 jobs đồng thời |
| `500` | Internal Server Error | Lỗi server không mong đợi |

---

## 5. Sequence Diagram — Luồng Sử Dụng Chính

```
Client                          API Server                    Pipeline (Background)
  │                                │                                │
  │── POST /api/upload/video ─────>│                                │
  │<── {file_id, path} ───────────│                                │
  │                                │                                │
  │── POST /api/upload/audio ─────>│                                │
  │<── {file_id, path} ───────────│                                │
  │                                │                                │
  │── POST /api/process ──────────>│                                │
  │<── {job_id, status:"queued"} ──│── run 8 steps ───────────────>│
  │                                │                                │
  │── GET /api/status/{job_id} ───>│   [step 1: OCR timeline]      │
  │<── {status:"processing", 1/8} │                                │
  │                                │   [step 2: classify]          │
  │── GET /api/status/{job_id} ───>│                                │
  │<── {status:"processing", 4/8} │   [step 4: translate]         │
  │                                │                                │
  │    ... polling ...             │   [step 6: voice cloning]     │
  │                                │                                │
  │── GET /api/status/{job_id} ───>│   [step 8: render]            │
  │<── {status:"completed", 8/8} ──│<── done ─────────────────────│
  │                                │                                │
  │── GET /api/result/{job_id} ───>│                                │
  │<── video/mp4 (dubbed video) ──│                                │
  │                                │                                │
  │── GET /api/result/{job_id}/srt>│                                │
  │<── text/plain (SRT file) ─────│                                │
```

---

## 6. Pydantic Schemas

### Request Schemas

```python
class ProcessRequest(BaseModel):
    video_path: str                           # required
    audio_path: str                           # required
    vlm_mode: str = "local"                   # "local" | "api"
    tts_engine: str = "vieneu"                # "vieneu"
    source_lang: str = "English"
    target_lang: str = "Vietnamese"
    keep_original_audio: bool = False
    pipeline_mode: str = "ocr"                # "ocr" | "vlm"
```

### Response Schemas

```python
class UploadResponse(BaseModel):
    file_id: str
    filename: str
    path: str
    size_mb: float

class ProcessResponse(BaseModel):
    job_id: str
    status: str
    message: str

class StatusResponse(BaseModel):
    job_id: str
    status: str                               # queued|processing|completed|failed|cancelled
    progress: ProgressInfo
    created_at: str
    error: Optional[str] = None

class ProgressInfo(BaseModel):
    step: str
    step_name_vi: str
    current: int
    total: int
    percent: int

class JobListResponse(BaseModel):
    jobs: List[JobSummary]
    total: int
    limit: int
    offset: int

class ResultReportResponse(BaseModel):
    job_id: str
    video_path: str
    reference_audio: str
    output_path: str
    srt_path: Optional[str] = None
    mode: str
    elapsed_seconds: float
    n_segments_raw: Optional[int] = None
    n_segments_after_classify: Optional[int] = None
    n_segments_merged: Optional[int] = None

class HealthResponse(BaseModel):
    status: str                               # healthy|degraded|unhealthy
    version: str
    timestamp: str
    dependencies: Dict[str, Any]
    active_jobs: int
    max_concurrent_jobs: int
    disk_usage: Dict[str, float]

class ConfigResponse(BaseModel):
    vlm: Dict[str, Any]
    tts: Dict[str, Any]
    pipeline: Dict[str, Any]
    sync: Dict[str, Any]
    limits: Dict[str, Any]
    output: Dict[str, Any]
```

---

## 7. File Structure

```
app/
├── __init__.py
├── main.py                    # FastAPI app, mount routers, CORS, exception handlers
├── gradio_ui.py               # Gradio Web UI (mounted at /ui)
├── routers/
│   ├── upload.py              # POST /upload/video, /upload/audio
│   ├── process.py             # POST /process, DELETE /process/{job_id}
│   ├── status.py              # GET /status/{job_id}, GET /jobs
│   ├── download.py            # GET /result/{job_id}, /result/{job_id}/srt, /result/{job_id}/report
│   ├── health.py              # GET /health                        [MỚI]
│   └── config.py              # GET /config                        [MỚI]
└── schemas/
    ├── request.py             # ProcessRequest (updated defaults)
    └── response.py            # All response models (expanded)
```

---

## 8. Ghi Chú Thiết Kế

1. **Không có Evaluation API**: Đánh giá chất lượng SRT/dịch thuật được thực hiện **thủ công** bởi người dùng xem video. Module M5 (Evaluation) vẫn được giữ trong source code cho CLI evaluation nhưng không expose qua REST API.

2. **Pipeline Mode mặc định = OCR**: Pipeline OCR-driven phù hợp hơn cho video có caption burned-in. Nếu video không có caption, user có thể chuyển sang `pipeline_mode: "vlm"`.

3. **VLM Mode mặc định = Local**: Sử dụng Qwen3.5-2B local, không cần Gemini API key. Yêu cầu GPU ≥ 4GB VRAM.

4. **Rate Limiting đơn giản**: Giới hạn 5 jobs đồng thời bằng đếm in-memory. Khi đạt limit, trả HTTP `429 Too Many Requests`.

5. **Swagger UI**: Tự động có tại `http://localhost:8000/docs` (FastAPI tích hợp sẵn). Không cần implement thêm.

6. **Gradio UI**: Web interface cho demo tại `http://localhost:8000/ui`. Sử dụng cùng backend API.
