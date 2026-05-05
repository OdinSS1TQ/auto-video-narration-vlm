# Báo cáo tiến độ Phase 1

**Hệ thống lồng tiếng video tự động bằng tiếng Việt**
Sử dụng Vision-Language Model (VLM) và Zero-shot Voice Cloning

| | |
|---|---|
| **Sinh viên** | Ngô Nguyễn Tấn Quân |
| **Ngày** | 17/04/2026 |
| **Giai đoạn** | Phase 1 — Trích xuất VLM và Tổng tạo giọng nói |

---

## 1. Tổng quan

Trong giai đoạn Phase 1, đề tài tập trung hoàn thành hai module cốt lõi:

- **Module 1 (VLM):** Trích xuất phụ đề tiếng Việt từ video tiếng Anh thông qua mô hình Vision-Language, bao gồm phát hiện cảnh, trích xuất khung hình, loại bỏ trùng lặp, suy diễn VLM và sinh file SRT.
- **Module 2 (TTS):** Tổng tạo giọng nói tiếng Việt với khả năng sao chép giọng nói zero-shot, sử dụng VieNeu-TTS v2 Turbo.

Hai module đã được phát triển, kiểm thử độc lập và **tích hợp thành công** (21/04/2026). Module 1 trích xuất phụ đề SRT từ video, Module 2 tổng tạo giọng nói từ văn bản với giọng tham chiếu, bridge SRT→TTS hoạt động end-to-end.

---

## 2. Quyết định kỹ thuật

### 2.1 Lựa chọn mô hình VLM

Hệ thống đánh giá ba mô hình VLM trước khi chốt kiến trúc hiện tại:

| Mô hình | VRAM | Context | Ưu điểm | Nhược điểm |
|---------|------|---------|---------|------------|
| **Gemini 1.5 Flash** | Cloud | 1M tokens | Chất lượng tốt nhất, xử lý video gốc | Cần API key, rate limit, độ trễ mạng |
| **Qwen2.5-VL-3B** | ~6 GB | 8K tokens | Mô hình local đầu tiên test được | VRAM cao, cần `qwen_vl_utils`, context nhỏ |
| **Qwen3.5-2B** | ~4 GB | 262K tokens | VRAM thấp, API đơn giản, context lớn | Mô hình nhỏ hơn, có thể hallucinate nhiều hơn Gemini |

**Lý do chọn Qwen3.5-2B làm mô hình local mặc định:**

1. **VRAM thấp** — 4GB so với 6GB của Qwen2.5-VL-3B, phù hợp GPU phổ thông (GTX 1660, RTX 3050)
2. **API đơn giản** — Dùng `AutoModelForImageTextToText` (Transformers chuẩn), không cần thư viện riêng `qwen_vl_utils`
3. **Context window lớn** — 262K tokens so với 8K tokens, quan trọng để xử lý các chunk video dài với đầy đủ frame context
4. **Tương thích về sau** — Dùng chat template API chuẩn, dễ đổi sang các biến thể Qwen3.5 khác (0.8B cho thiết bị yếu)

**Tham số sinh (theo model card Qwen3.5):**

```
top_p=0.8, top_k=20, repetition_penalty=1.1, max_new_tokens=4096
```

**Cơ chế tự động nhận diện model:** `VLMClient` kiểm tra đường dẫn model — nếu chứa "qwen2.5" hoặc "qwen2-vl", nạp qua `Qwen2_5_VLForConditionalGeneration` (legacy). Ngược lại, nạp qua `AutoModelForImageTextToText` (Qwen3.5).

### 2.2 Thiết kế Structured Prompt

Hệ thống prompt được thiết kế với nhiều kỹ thuật chống hallucination, yêu cầu output JSON có cấu trúc rõ ràng.

#### Hai chế độ prompt

**Chế độ 1 — Single-prompt (mặc định):** Trích xuất + dịch + gán timestamp trong một lần gọi VLM. Phù hợp cho Gemini và Qwen3.5 (context lớn). Hiệu quả hơn, giảm tích lũy sai số giữa các bước.

**Chế độ 2 — 3-step chain:** Gọi riêng biệt: extract → translate → format. Kích hoạt bằng `--mode 3step`. Phù hợp cho mô hình có context nhỏ.

#### Kỹ thuật chống hallucination

1. **Neo timestamp khung hình (Frame timestamp anchoring):** Prompt liệt kê chính xác timestamp của từng frame làm điểm neo. VLM được yêu cầu: *"Timestamp của phụ đề phải nằm trong khoảng [chunk_start, chunk_end]"* và *"start_time của mỗi phụ đề phải gần một frame timestamp."* Điều này giúp mô hình dựa trên bằng chứng hình ảnh thực tế.

2. **Timestamp tuyệt đối (Absolute timestamps):** Prompt yêu cầu rõ ràng: *"Sinh timestamp theo thời gian TUYỆT ĐỐI (tính từ đầu video, KHÔNG phải từ đầu chunk)."* Đây là bản sửa lỗi sau khi phát hiện timestamp tương đối gây ra lỗi double-offset khi ghép các chunk.

3. **Chống lặp lại (Anti-repeat injection):** Kết quả dịch của các chunk trước được chèn vào prompt với nhãn *"ĐÃ TẠO — KHÔNG LẶP LẠI"*, ngăn VLM tạo lại nội dung từ các chunk chồng lấp.

4. **Chỉ tóm tắt mức cao (HIGH-LEVEL OVERVIEW ONLY):** Prompt yêu cầu VLM mô tả MỤC ĐÍCH và CHỦ ĐỀ thay vì hành động UI cụ thể (ví dụ: "click menu File"). Tránh narration quá chi tiết.

5. **Output JSON có cấu trúc:** Tất cả prompt yêu cầu mảng JSON với schema:

```json
{
  "index": 1,
  "start_time": "00:01:23,456",
  "end_time": "00:01:26,789",
  "original_text": "English narration",
  "translated_text": "Vietnamese narration"
}
```

#### Ràng buộc SRT nhúng trong prompt

Các quy tắc phụ đề được nhúng trực tiếp vào hướng dẫn VLM:

- Thời lượng phụ đề: 2–7 giây
- Tối đa 2 dòng mỗi phụ đề
- Tối đa 42 ký tự mỗi dòng
- Tốc độ đọc tối đa: 25 ký tự/giây
- Không chồng lấp timestamp

#### Quy tắc dịch thuật

Quy tắc cụ thể khi dịch nội dung kỹ thuật tiếng Anh sang tiếng Việt:

- **Code/lệnh:** Giữ nguyên tiếng Anh (ví dụ: `pip install`, `def function()`)
- **Thuật ngữ kỹ thuật:** Giữ tiếng Anh kèm giải thích tiếng Việt ở lần xuất hiện đầu (ví dụ: "thuật toán Machine Learning (Học máy)")
- **Text UI/nút bấm:** Dịch phần giải thích, giữ nhãn tiếng Anh (ví dụ: "nhấn nút Save để lưu")
- **Phong cách:** Tiếng Việt giao tiếp, phong cách nói chuyên nghiệp, phù hợp cho narration

### 2.3 Hệ thống ngữ cảnh 2 lớp

Hệ thống ngăn vấn đề "chunk cô lập" — mỗi chunk được dịch mà không biết nội dung chunk lân cận.

**Lớp 1 — Tóm tắt toàn cục (Pass 0):**
- 15 frame lấy mẫu đều trên toàn bộ video
- VLM sinh: chủ đề, phong cách, các phần, thuật ngữ chính, tóm tắt tổng quan
- Thiết lập một lần, chèn vào prompt của mọi chunk
- Cung cấp ngữ cảnh video-level bất kể chunk đang xử lý

**Lớp 2 — Cửa sổ trượt (Sliding window):**
- Duy trì cửa sổ 3 chunk gần nhất (~135 giây ngữ cảnh)
- Hiển thị TẤT CẢ phụ đề đã sinh với nhãn "KHÔNG LẶP LẠI"
- Tự động trích xuất bảng thuật ngữ (gốc → đã dịch)
- Ngăn dịch trùng lặp và thuật ngữ không nhất quán giữa các chunk

### 2.4 Lựa chọn engine TTS

**VieNeu-TTS v2 Turbo** được chọn cho zero-shot Vietnamese voice cloning:

| Khía cạnh | Quyết định | Lý do |
|-----------|------------|-------|
| Engine | VieNeu-TTS v2 Turbo | Hỗ trợ tiếng Việt gốc, zero-shot cloning |
| Backend | GGUF + ONNX | Không cần PyTorch/Transformers khi chạy, footprint nhỏ |
| Audio tham chiếu | 3–5 giây | Cân bằng giữa chất lượng cloning và tiện lợi cho người dùng |
| Xử lý | Tuần tự | SDK VieNeu không thread-safe; tuần tự tránh race condition |
| Cache | ref_codes in-memory (khóa SHA1) | Encode reference tốn kém; cache tránh re-encode cùng audio |
| Output | float32, 24kHz mono | Chuẩn tương thích với xử lý audio downstream |

### 2.5 Quy trình xử lý video

**Phát hiện cảnh:** PySceneDetect `ContentDetector` với ngưỡng 27.0. Chia video thành các chunk xử lý (mặc định 45s) với overlap 5 giây để ngăn khoảng trống narration tại ranh giới chunk.

**Trích xuất frame:** Lấy mẫu thích ứng dựa trên cấu trúc cảnh:
- Cảnh ngắn (< 5s): frame onset + midpoint
- Cảnh dài (> 5s): onset + midpoint + frame trung gian
- Ràng buộc min/max: 3–30 frame mỗi chunk

**Loại trùng lặp SSIM:** SSIM nhanh tự triển khai (không phụ thuộc scikit-image). Thu nhỏ về 256x256 để tăng tốc. So sánh với frame ĐƯỢC GIỮ CUỐI CÙNG (không phải frame trước đó) để ngăn drift dần. Thường loại bỏ 40–70% frame.

---

## 3. Module 1: VLM Extraction

### 3.1 Các thành phần đã triển khai

| Thành phần | Mô tả |
|-----------|--------|
| **SceneDetector** | Phân đoạn video theo nội dung bằng PySceneDetect. Chia chunk với overlap cấu hình được. Sinh adaptive frame timestamps dựa trên cấu trúc cảnh. |
| **FrameExtractor** | Trích xuất frame tại timestamp cụ thể. Mã hóa base64 JPEG (max 768px width). Hỗ trợ lấy mẫu đều cho Pass 0. |
| **FrameDeduplicator** | Loại trùng lặp dựa trên SSIM. So sánh với frame cuối được giữ để ngăn drift. Giảm 40–70% frame. |
| **VLMClient** | Suy diễn VLM multi-backend. API: Gemini 1.5 Flash với retry tự động. Local: Qwen3.5-2B (mặc định) hoặc Qwen2.5-VL-3B (legacy). |
| **PromptChain** | Prompt engineering với single-prompt + 3-step chain. Anti-hallucination, quy tắc dịch thuật, ràng buộc SRT. |
| **ContextWindow** | Ngữ cảnh 2 lớp: tóm tắt toàn cục + cửa sổ trượt. Bảng thuật ngữ tự động. |
| **SRTBuilder** | Build, deduplicate (time + text), validate, lưu file SRT. |
| **SubtitleValidator** | Validate định dạng, overlap, thời lượng (0.5–10s) + tự động sửa overlap. |

### 3.2 Kiểm thử Module 1

**pytest chính thức (8 files):**

| File test | Phạm vi |
|-----------|---------|
| `tests/m1_vlm/test_scene_detector.py` | SceneDetector — phát hiện cảnh, chia chunk, adaptive timestamps |
| `tests/m1_vlm/test_frame_extractor.py` | FrameExtractor — trích xuất frame, mã hóa base64 |
| `tests/m1_vlm/test_context_window.py` | ContextWindow — ngữ cảnh 2 lớp, sliding window, thuật ngữ |
| `tests/m1_vlm/test_prompt_chain.py` | PromptChain — template single + 3-step, anti-hallucination |
| `tests/m1_vlm/test_srt_builder.py` | SRTBuilder — build, deduplicate, validate, I/O |
| `tests/m1_vlm/test_validator.py` | SubtitleValidator — validate format, overlap, duration |
| `tests/m1_vlm/test_glm_ocr_integration.py` | GLM-OCR — trích xuất text trên frame |
| `tests/test_pipeline_e2e.py` | End-to-end — chạy toàn bộ pipeline M1 |

**Script kiểm thử:**

| Script | Dòng | Mô tả |
|--------|------|--------|
| `scripts/run_vlm_extract.py` | 630 | Script production chính, chạy pipeline trích xuất đầy đủ |
| `scripts/verify_m1_vlm.py` | 535 | Xác minh toàn bộ M1 — tạo video synthetic, chạy pipeline, lưu kết quả trung gian |

### 3.3 Vấn đề timestamp

Timestamp trong SRT sinh ra chưa hoàn toàn chính xác. Cách tiếp cận ban đầu sinh timestamp tương đối (từ đầu chunk), gây lỗi double-offset khi ghép các chunk. Đã khắc phục một phần bằng:

1. Đổi prompt yêu cầu timestamp tuyệt đối
2. Đặt `chunk_offset=0.0` trong script trích xuất
3. Thêm deduplication cho các entry từ chunk chồng lấp

Tuy nhiên, độ chính xác timestamp vẫn cần cải thiện — VLM đôi khi sinh start/end time không chính xác, đặc biệt khi narration không có cue hình ảnh rõ ràng.

---

## 4. Module 2: TTS Voice Cloning

### 4.1 Các thành phần đã triển khai

| Thành phần | Mô tả |
|-----------|--------|
| **TTSClient** | Engine TTS chính với pipeline voice cloning: encode reference → synthesize → save. Hỗ trợ reference audio, preset voices, cache ref_codes in-memory. Output float32 24kHz mono. |
| **SpeakerEncoder** | Quản lý nhận dạng giọng nói 2 đường: VieNeu (cho synthesis) và resemblyzer (cho đánh giá embedding). |
| **BatchInference** | Xử lý batch tuần tự (SDK VieNeu không thread-safe). Encode reference một lần, tái sử dụng cho tất cả segments. |

### 4.2 Kiểm thử Module 2

`scripts/test_m2_tts.py` — 936 dòng, 24 test case:

#### Unit tests (12 test, không cần GPU)

| Test | Mô tả |
|------|--------|
| Import & instantiation | Kiểm tra import chain, khởi tạo TTSClient |
| Text normalization | 8 edge case tiếng Việt (dấu câu, khoảng trắng, ellipsis) |
| Cache key stability | Kiểm tra SHA1 cache key nhất quán |
| Engine validation | Từ chối engine không hợp lệ |
| Reserved engine rejection | F5-TTS/viXTTS raises NotImplementedError |
| SpeakerEncoder instantiation | Khởi tạo dual-path encoder |
| Cosine similarity | 5 vector đã biết, kiểm tra kết quả cosine |
| Resemblyzer disk caching | Kiểm tra cache embedding trên disk (mocked) |
| BatchInference init | Khởi tạo batch processor |
| Batch preparation | 6 kịch bản bao gồm boundary conditions |
| Missing key handling | Xử lý thiếu config |

#### Integration tests (12 test, cần GPU + vieneu SDK)

| Test | Mô tả |
|------|--------|
| Model load + preset voices | Nạp model, liệt kê preset voices |
| Preset voice synthesis | Sinh audio bằng preset voice |
| Reference encode + cache hit | Encode reference audio, kiểm tra cache hit (đo thời gian) |
| Voice-cloned synthesis (file) | Sinh audio với giọng clone, lưu file |
| Voice-cloned synthesis (ndarray) | Sinh audio, kiểm tra float32 array |
| Empty text | Trả về silence khi text rỗng |
| Missing file error | FileNotFoundError khi file không tồn tại |
| Context manager | Load/unload model tự động |
| SpeakerEncoder delegation | VieNeu path delegate đúng |
| BatchInference.process_all | 4 segments bao gồm empty text |
| Sequential synthesis | 4 vòng sinh liên tiếp |
| Reference duration variants | Test với reference audio 2s, 3s, 5s, 8s |

**Script smoke test:**

`scripts/test_vieneu_tts.py` — 195 dòng: Import chain, lazy loading, preset synthesis, reference encoding, voice cloning.

### 4.3 Tích hợp M1→M2 và kiểm thử end-to-end (21/04/2026)

Sau khi hoàn thành unit/integration test riêng lẻ, bước tiếp theo là kết nối đầu ra Module 1 (SRT) vào đầu vào Module 2 (TTS). Bridge sử dụng `SRTBuilder.load_srt()` nạp SRT file thành list of dicts với key `translated_text`, trực tiếp truyền vào `BatchInference.process_all()` — không cần adapter hay transformation.

#### a) Fixture test

Tạo `data/test_fixtures/sample_output.srt` gồm 10 entry phụ đề tiếng Việt mô phỏng đầu ra thực tế từ M1 VLM: câu ngắn (41 ký tự) đến câu dài (75 ký tự), thời lượng 2.0s–3.5s, nội dung đa dạng (chào hỏi, giới thiệu, kỹ thuật, kết thúc).

#### b) Script test tích hợp

Tạo `scripts/test_m1_m2_integration.py` (~440 dòng) với pipeline: nạp SRT → khởi tạo TTSClient → encode reference → sinh audio từng segment (có timing) → validate output → báo cáo so sánh SRT duration vs Audio duration. Hỗ trợ CLI flags: `--ref-audio`, `--srt`, `--output-dir`, `--vieneu-mode`, `--device`.

#### c) Kết quả kiểm thử với 3 loại reference audio

Tất cả đều PASS (10/10 segments):

| Lần test | Audio tham chiếu | Thông số | Segments | Tổng thời gian | TB/segment | Ref encode |
|----------|-----------------|----------|----------|---------------|------------|------------|
| 1 | VieNeu sample — Bình (nam miền Bắc) | 2.65s, 24kHz | 10/10 | 40.9s | 1.52s | 25.7s |
| 2 | VieNeu sample — Đoan (nữ miền Nam) | 6.14s, 24kHz | 10/10 | 19.7s | 1.36s | 6.1s |
| 3 | Real audio — user_D | 10.0s, 16kHz | 10/10 | 28.5s | 2.40s | 4.5s |

Kết quả chi tiết lần test với real audio (user_D):

| Segment | SRT Duration | Audio Duration | Ratio | Synth Time |
|---------|-------------|----------------|-------|------------|
| 1 | 2.50s | 3.20s | 1.28 | 1.305s |
| 2 | 2.80s | 3.08s | 1.10 | 1.241s |
| 3 | 2.70s | 3.00s | 1.11 | 1.556s |
| 4 | 3.50s | 3.40s | 0.97 | 1.314s |
| 5 | 3.30s | 3.52s | 1.07 | 1.394s |
| 6 | 2.50s | 2.08s | 0.83 | 0.981s |
| 7 | 3.50s | 3.68s | 1.05 | 4.894s |
| 8 | 3.50s | 3.48s | 0.99 | 5.969s |
| 9 | 3.20s | 3.90s | 1.22 | 3.978s |
| 10 | 2.50s | 3.08s | 1.23 | 1.324s |

#### d) Phát hiện kỹ thuật

1. **Bridge sạch**: `SRTBuilder.load_srt()` → `BatchInference.process_all()` không cần adapter
2. **Tốc độ synthesis**: ~1.0–1.7s/segment cho VieNeu sample audio (RTF ~0.4x)
3. **Ratio SRT/Audio**: 0.83–1.28, audio sinh ra gần đúng thời lượng SRT — input quan trọng cho Module 3 (audio alignment)
4. **Audio 16kHz**: Một số segment có RTF >1.0 khi dùng audio 16kHz (chậm hơn realtime) — do sample rate thấp hơn 24kHz native
5. **Output ổn định**: Tất cả WAV 24kHz mono, peak amplitude 0.26–0.57

Output audio lưu tại `debug_output/m1_m2_integration*/audio_chunks/` (3 test runs × 10 files = 30 WAV).

---

## 5. Chất lượng code

### 5.1 Thiết kế

- **Lazy model loading:** Tất cả model nặng (VLM, TTS, OCR) chỉ nạp khi lần đầu gọi inference, không nạp khi import module. Giảm thời gian khởi động và tiết kiệm VRAM khi chưa cần dùng.
- **Xử lý tuần tự cho TTS:** Batch inference chạy tuần tự có chủ đích vì SDK VieNeu không thread-safe.
- **Cơ chế callback tiến trình:** Pipeline runner hỗ trợ progress callback để tích hợp với CLI progress bar hoặc web UI.
- **Dual config:** `.env` cho secrets + `configs/*.yaml` cho default cấu trúc, tách biệt rõ ràng.
- **Custom exception hierarchy:** `PipelineError` → `StepError`, `VLMError`, `TTSError`, `SyncError`, `ConfigError`, `ValidationError`.

### 5.2 Công cụ

| Công cụ | Phạm vi |
|---------|---------|
| **ruff** | Lint `src/`, `app/`, `tests/` |
| **black** | Format `src/`, `app/`, `tests/` |
| **pytest** | 8 file test chính thức cho M1, 24 test case cho M2 |
| **loguru** | Logging console (color) + file handler (rotation 10MB, retention 7 ngày) |

### 5.3 Cấu trúc file

```
src/
├── m1_vlm/                    # Module 1: VLM extraction
│   ├── vlm_client.py          # VLMClient — multi-backend inference (458 dòng)
│   ├── prompt_chain.py        # PromptChain — prompt engineering (432 dòng)
│   ├── context_window.py      # ContextWindow — 2-layer context (173 dòng)
│   ├── scene_detector.py      # SceneDetector — PySceneDetect (269 dòng)
│   ├── frame_extractor.py     # FrameExtractor — frame extraction (223 dòng)
│   ├── frame_dedup.py         # FrameDeduplicator — SSIM dedup (201 dòng)
│   ├── srt_builder.py         # SRTBuilder — SRT management (215 dòng)
│   ├── validator.py           # SubtitleValidator — validation (165 dòng)
│   └── glm_ocr.py             # GLMOCR — on-screen text (226 dòng)
├── m2_tts/                    # Module 2: TTS voice cloning
│   ├── tts_client.py          # TTSClient — VieNeu integration (429 dòng)
│   ├── speaker_encoder.py     # SpeakerEncoder — dual-path (176 dòng)
│   └── batch_inference.py     # BatchInference — sequential batch (180 dòng)
├── m4_pipeline/               # Pipeline orchestrator (scaffolding)
│   ├── runner.py              # PipelineRunner — 8-step async
│   ├── config.py              # PipelineConfig — .env + YAML
│   ├── exceptions.py          # Custom exception hierarchy
│   └── logger.py              # Loguru logging setup
└── utils/                     # Tiện ích dùng chung
    ├── async_utils.py         # Async helpers, retry, progress tracker
    ├── audio_utils.py         # Load, normalize, save, resample audio
    ├── file_utils.py          # Directory, temp file, filename utilities
    └── video_utils.py         # Duration, resolution, FPS, extract audio
```

---

## 6. Kết luận

Phase 1 hoàn thành hai module cốt lõi và tích hợp thành công:

**Module 1 (VLM Extraction):** Pipeline trích xuất phụ đề hoạt động đầy đủ từ video → SRT với 8 thành phần, 8 file pytest, và 2 script kiểm thử production. Điểm cần cải thiện chính là độ chính xác timestamp.

**Module 2 (TTS Voice Cloning):** Hệ thống tổng tạo giọng nói với zero-shot cloning hoàn chỉnh, 24 test case bao phủ cả unit (12) và integration (12) test. Đã tích hợp thành công với M1 qua bridge `SRTBuilder.load_srt()` → `BatchInference.process_all()`, kiểm thử end-to-end với 3 loại reference audio (VieNeu nam, VieNeu nữ, real audio), tất cả PASS.

---

## 7. Trạng thái tổng thể sau 4 tuần (31/03 – 21/04/2026)

### Timeline tổng hợp

| Tuần | Ngày | Công việc |
|------|------|-----------|
| 1 | 31/03 | Khởi tạo project, scaffolding Module 1, cấu hình Gemini API + Qwen2.5-VL-3B |
| 2 | 13/04 | Phát triển M1: scene detection, frame extraction, VLM inference, prompt engineering |
| 3 | 14/04 | Chuyển sang Qwen3.5-2B, tối ưu prompt (absolute timestamps), hoàn thiện M1 |
| 3 | 17/04 | Viết lại M2: F5-TTS/viXTTS → VieNeu-TTS v2 Turbo, 24 test case, báo cáo Phase 1 |
| 4 | 21/04 | **Tích hợp M1→M2**, test với VieNeu sample + real audio, đo hiệu năng |

### Trạng thái module

| Module | Trạng thái | Ghi chú |
|--------|-----------|---------|
| **M1: VLM Extraction** | ✅ Hoàn thành | 8 components, 8 pytest files, 2 test scripts. Timestamp accuracy cần cải thiện. |
| **M2: TTS Voice Cloning** | ✅ Hoàn thành | 24 test case + tích hợp M1→M2 thành công. Test 3 loại reference audio, bridge SRT→TTS hoạt động end-to-end. |
| **M3: Audio Sync** | ⚠️ Scaffolding | AudioAligner, TimeStretcher, FFmpegRenderer — chưa test. |
| **M4: Pipeline Orchestrator** | ⚠️ Scaffolding | PipelineRunner 8 steps, PipelineConfig — chưa test end-to-end. |
| **M5: Evaluation** | ⚠️ Một phần | BLEU, Speaker Similarity, Sync Accuracy hoạt động. MOS chưa implement. |
| **Web App** | ⚠️ Scaffolding | FastAPI + Gradio — chưa test. |

### Công việc tiếp theo (Phase 2)

1. **Module 3 integration** — Kết nối audio output từ M2 với audio alignment + time stretching + video rendering
2. **End-to-end pipeline** — Chạy video qua toàn bộ M1→M2→M3, đánh giá chất lượng output
3. **Cải thiện timestamp** — Đánh giá độ chính xác timestamp, tối ưu nếu cần
4. **Module 5 evaluation** — Chạy metrics trên output thực, implement MOS estimation
5. **Web UI** — Test và hoàn thiện giao diện FastAPI + Gradio
