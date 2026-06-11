# Báo Cáo Tiến Độ Giai Đoạn 2

**Hệ Thống Lồng Tiếng Tiếng Việt Tự Động Cho Video**
Sử dụng Mô Hình Ngôn Ngữ - Thị Giác (VLM) và Nhân Bản Giọng Nói Zero-shot

| | |
|---|---|
| **Sinh viên** | Ngô Nguyễn Tấn Quân |
| **Ngày** | 27 tháng 5, 2026 |
| **Giai đoạn** | 2 — Kết nối End-to-End, Độ chính xác Timestamp, Chế độ OCR |
| **Thời gian** | 18 tháng 4 – 27 tháng 5, 2026 |

---

## 1. Tóm tắt

Giai đoạn 2 tập trung vào việc khắc phục ba vấn đề lớn nhất từ Giai đoạn 1: (1) các module đã được xây dựng khung nhưng chưa bao giờ được kết nối thành một pipeline chạy hoàn chỉnh, (2) pipeline có lỗi offset timestamp, và (3) timestamp do VLM tạo ra không chính xác. Đến cuối Giai đoạn 2, hệ thống có thể tạo ra video lồng tiếng Việt hoàn chỉnh từ video tiếng Anh thông qua hai pipeline riêng biệt: `--mode vlm` (dựa trên VLM) và `--mode ocr` (dựa trên OCR cho video có phụ đề gắn sẵn).

---

## 2. Các công việc đã hoàn thành

### 2.1 Pipeline End-to-End (Phase 2.A)

- **Kiểm tra Module 3 (Sync/Render) độc lập** — Tạo script kiểm tra riêng để xác nhận audio alignment và video rendering hoạt động đúng với output TTS thực tế, trước khi tích hợp vào orchestrator.
- **Kết nối toàn bộ 8 bước pipeline vào PipelineRunner** — Phát hiện cảnh, trích xuất frame, trích xuất OCR, dịch VLM, tạo SRT, nhân bản giọng nói, căn chỉnh audio, và render video giờ chạy end-to-end tuần tự.
- **Sửa lỗi chunk_offset** — Lỗi đã biết từ Giai đoạn 1 tại `runner.py:175` gây ra timestamp bị offset kép. Đã sửa bằng cách dùng `chunk_offset=0.0` vì prompt phát ra timestamp tuyệt đối.
- **Sửa các vấn đề tương thích Windows** — Chuẩn hóa ASCII cho tên file video có ký tự đặc biệt (ví dụ: `【VNEXT】`) để tương thích subprocess; thêm mã hóa UTF-8 cho output subprocess; hỗ trợ biến môi trường `RUBBERBAND_PATH` tùy chỉnh cho time stretcher.
- **Thêm quản lý bộ nhớ VLM** — Pipeline giờ giải phóng trọng số VLM sau khi tạo SRT để TTS có thể chạy trên GPU 8 GB.
- **Đồng bộ hành vi pipeline với script độc lập** — Pipeline giờ sử dụng lấy mẫu frame thích ứng, loại bỏ trùng lặp SSIM, phân tích JSON VLM linh hoạt, và luôn lưu phản hồi VLM thô để debug.
- **Kích hoạt Pass 0 tóm tắt toàn cục** — Tóm tắt ngữ cảnh cấp video (đã triển khai trong Giai đoạn 1 nhưng chưa được kích hoạt) giờ được gọi bởi pipeline runner ở cả hai chế độ.
- **Tạo video lồng tiếng đầu tiên hoàn chỉnh** — Đã tạo thành công video lồng tiếng Việt hoàn chỉnh từ tài liệu kiểm tra `Demo-Module-5.mp4` (60.4 giây).
- **Cập nhật tài liệu** — Cập nhật `README.md` và thêm hướng dẫn `docs/RUNNING.md`.

### 2.2 Đồng bộ Timestamp SRT (Phase 2.C)

- **Xây dựng EntryRetimer** — Module tái định thời gian hậu VLM, loại bỏ timestamp do VLM tự tạo và tái tạo timing hợp lý dựa trên độ dài chunk, điểm neo frame, và độ dài văn bản tiếng Việt (mô hình ký tự trên giây). Giải quyết vấn đề tất cả phụ đề bị dồn vào 16 giây đầu của chunk 45 giây.
- **Viết lại AudioAligner với budgeted stretch** — Thay vì nén mọi segment để vừa một cách vô điều kiện, aligner giờ quyết định từng segment với bốn chiến lược: giữ nguyên, nén vừa vặn, nén tối đa với trượt, hoặc đệm bằng khoảng lặng.
- **Triển khai slip cascade** — Khi segment TTS quá dài ngay cả sau khi nén tối đa, phần tràn sẽ đẩy điểm bắt đầu các segment tiếp theo về phía sau thay vì ép tăng tốc bất thường lên mọi segment.
- **Thêm giới hạn time stretching** — Phương thức `stretch_capped` mới trong TimeStretcher giới hạn nén rubberband ở tốc độ tối đa có thể cấu hình (mặc định 1.25x) để bảo toàn sự tự nhiên của giọng nói.
- **Trạng thái: chất lượng còn mở** — Retimer loại bỏ các vấn đề timing tệ nhất, nhưng độ rộng slot vẫn dựa trên heuristic. Timestamp ở chế độ VLM vẫn chưa đạt yêu cầu cho lồng tiếng chất lượng cao, đây là động lực để phát triển phương pháp OCR bên dưới.

### 2.3 Pipeline Timestamp dựa trên OCR (Phase 2.OCR)

- **Xây dựng CaptionTimeline** — Module mới lấy mẫu frame video ở 2 fps, cắt vùng phụ đề phía dưới, chạy GLM-OCR để đọc phụ đề tiếng Anh gắn sẵn, và nhóm các kết quả đọc liên tiếp giống nhau thành các segment phụ đề có thời gian. Bao gồm phát hiện ảo giác (vòng lặp lặp lại) và chuẩn hóa các lỗi OCR phổ biến.
- **Thêm bộ phân loại tường thuật** — Bộ phân loại dựa trên VLM gán nhãn mỗi segment OCR là `narration` (tường thuật) hoặc `screen` (nhãn UI, đoạn code, tiêu đề slide), loại bỏ các hàng không phải tường thuật trước khi dịch. Điều này ngăn pipeline lồng tiếng cho văn bản hiển thị trên màn hình không phải lời nói.
- **Triển khai ghép segment bảo thủ** — Xử lý hai trường hợp: (1) phụ đề kiểu typewriter/hoạt ảnh khi segment trước là tiền tố của segment sau, và (2) phân mảnh nhỏ khi các segment ngắn liền kề được nối trong ngưỡng có thể cấu hình.
- **Thêm kéo dài thời gian kết thúc** — Thời gian kết thúc mỗi segment được kéo dài thêm tối đa 0.5 giây (giới hạn bởi thời gian bắt đầu segment tiếp theo) để TTS có thêm không gian cho tường thuật tiếng Việt, vốn dài hơn tiếng Anh khoảng 30%.
- **Tích hợp chế độ OCR vào PipelineRunner** — Thêm cờ `--mode ocr` chuyển đến pipeline OCR 8 bước riêng biệt trong khi giữ nguyên `--mode vlm` làm mặc định không thay đổi.
- **Thêm xử lý lỗi cho các lỗi đặc thù OCR** — Không phát hiện phụ đề nào, quá ít segment tường thuật, và quá ít segment đã dịch đều phát ra lỗi mô tả kèm gợi ý xử lý.

### 2.4 Module Đánh Giá (Phase 2.E)

- **Sửa lỗi SyncAccuracy** — `evaluate_segments` đang dùng giá trị cố định `0.0` thay vì `srt_start_sec` thực tế của SRT entry. Đã sửa và thêm test TDD với fixture onset tổng hợp 440 Hz.
- **Thêm đánh giá sync full-mix** — Phương thức mới `evaluate_against_merged_audio` đánh giá một file audio đã merge so với toàn bộ timeline SRT bằng phát hiện onset theo cửa sổ, báo cáo mean delay, p50, p95, và số onset bỏ lỡ.
- **Xây dựng wrapper đánh giá dịch thuật** — `evaluate_translation_from_srts()` tính BLEU-4 và chrF++ cấp corpus giữa SRT tạo ra và SRT tham chiếu biên tập thủ công, cộng thêm chrF++ cấp câu với chẩn đoán 5 câu tệ nhất.
- **Xây dựng wrapper đánh giá nhân bản giọng** — `evaluate_voice_clone()` tính độ tương đồng giọng nói (resemblyzer cosine) qua ba giai đoạn pipeline: chunk trước căn chỉnh, chunk sau căn chỉnh, và audio merge, với thống kê mean/std/p05 theo giai đoạn.
- **Xây dựng bộ đánh giá caption drift** — `compare_caption_tracks()` so khớp entry phụ đề dự đoán với ground-truth bằng tương đồng văn bản mờ trong cửa sổ thời gian, báo cáo drift start/end (mean/p50/p95), Jaccard overlap, và số bỏ lỡ/thừa.
- **Mở rộng ReportGenerator** — Thêm `generate_markdown()` cho báo cáo Markdown định dạng với bảng Tóm tắt, Dịch thuật, Nhân bản giọng, Đồng bộ, Caption Drift, và Config Snapshot (bỏ qua section trống). Mở rộng `generate_summary_csv()` với cột mới cho pipeline mode, speaker similarity merged, sync p95, và caption drift p95.
- **Tạo CLI đánh giá** — `scripts/run_evaluation.py` điều phối tất cả wrapper đánh giá qua tham số dòng lệnh, tạo báo cáo JSON + Markdown + CSV với snapshot đầy đủ PipelineConfig (20 thông số).
- **13 test đánh giá** — Tất cả pass: sửa sync accuracy (2), caption drift (6), translation eval (1), report markdown (2), CLI smoke test (2).

### 2.5 Sửa lỗi Audio & Pipeline OCR (26–27 tháng 5)

- **Sửa lỗi chuẩn hóa âm lượng** — Bộ lọc `amix` của FFmpegRenderer chia âm lượng cho số lượng segment đầu vào (ví dụ: 20 segment → âm lượng giảm còn 1/20). Thêm `normalize=0` vì các segment được phân tách theo thời gian không chồng lấn, khôi phục âm lượng đầy đủ.
- **Sửa lỗi trích xuất frame** — Thay thế phương pháp lấy mẫu frame dựa trên seek (`CAP_PROP_POS_MSEC`) trong `iter_video_samples` bằng giải mã tuần tự. Phương pháp seek nhảy đến keyframe gần nhất, khiến nhiều timestamp trả về cùng một frame và bỏ lỡ các chuyển đổi phụ đề thực tế trong hoạt ảnh.
- **Tăng tốc độ lấy mẫu OCR** — Giá trị mặc định `OCR_SAMPLE_FPS` tăng từ 2.0 lên 3.0 (một frame mỗi 0.33s thay vì 0.5s) để bắt được các phụ đề tồn tại ngắn và chuyển đổi hoạt ảnh.
- **Thêm loại trùng văn bản giống nhau vào ghép segment** — Chiến lược mới trong `merge_short_segments` phát hiện khi các segment liền kề có văn bản tiếng Anh giống hệt hoặc gần giống (tỷ lệ SequenceMatcher ≥ 0.85) trong khoảng cách lên đến ~7.5s, gộp chúng thành một segment trước khi dịch VLM. Ngăn chặn các bản dịch trùng lặp do chu kỳ hoạt ảnh khi cùng một phụ đề xuất hiện, mờ dần, và xuất hiện lại.

### 2.6 Kiểm thử

- **Thêm 7 file test mới** bao gồm:
  - EntryRetimer (phân phối, slot có trọng số, snap vào timestamp frame)
  - AudioAligner budgeted stretch (cả bốn chiến lược và slip cascade)
  - TimeStretcher giới hạn nén (ma trận quyết định)
  - CaptionTimeline (lấy mẫu, cắt vùng, nhóm, loại trùng, thời lượng tối thiểu)
  - Prompt chỉ dịch (kiểm tra định dạng)
  - Phân nhánh chế độ Runner (VLM vs OCR dispatch)
  - Thuộc tính cấu hình chế độ OCR (giá trị mặc định biến môi trường)
- **Tạo script smoke test thủ công** cho CaptionTimeline xuất segment và SRT để kiểm tra trực quan.

---

## 3. Trạng thái Module sau Giai đoạn 2

| Module | Trạng thái | Thay đổi so với Giai đoạn 1 |
|---|---|---|
| **M1 — Trích xuất VLM** | Hoạt động (cả hai chế độ) | Thêm EntryRetimer; thêm CaptionTimeline + bộ phân loại tường thuật + ghép/kéo dài; GLM-OCR được sử dụng trong chế độ OCR; Pass 0 kết nối vào runner; sửa giải mã frame tuần tự; FPS lấy mẫu 2→3; loại trùng văn bản giống nhau trong ghép |
| **M2 — TTS** | Không thay đổi | Được tái sử dụng bởi cả hai chế độ qua BatchInference |
| **M3 — Sync/Render** | Hoạt động | AudioAligner viết lại với budgeted stretch + slip cascade; thêm giới hạn time stretching; sửa lỗi Windows; sửa chuẩn hóa âm lượng amix |
| **M4 — Pipeline** | Hoạt động | Thêm bộ điều phối chế độ; sửa lỗi chunk_offset; quản lý bộ nhớ VLM; chuẩn hóa ASCII đường dẫn |
| **M5 — Đánh giá** | Hoạt động | Sửa lỗi SyncAccuracy + thêm biến thể full-mix; 3 wrapper mới (dịch thuật, giọng nói, caption-drift); ReportGenerator mở rộng với Markdown emitter; CLI `scripts/run_evaluation.py`; 13 test |
| **Web App** | Không thay đổi | Không được chạm đến trong Giai đoạn 2 |

---

## 4. Các vấn đề đã biết

| Vấn đề | Mức độ |
|---|---|
| Timestamp `--mode vlm` vẫn không chính xác (dựa trên heuristic); chế độ OCR đáng tin cậy hơn cho video có phụ đề gắn sẵn | Trung bình |
| ~~Chất lượng chế độ OCR chưa được đo lường định lượng~~ — **Đã giải quyết:** bộ đánh giá caption-drift + chỉ số dịch thuật đã sẵn sàng | ~~Cao~~ Xong |
| Tỷ lệ vùng phụ đề cố định (0.10) chỉ hoạt động cho một bố cục video | Trung bình |
| ~~Module M5 đánh giá chưa bao giờ được chạy end-to-end~~ — **Đã giải quyết:** CLI `scripts/run_evaluation.py` điều phối tất cả wrapper M5 | ~~Trung bình~~ Xong |
| Chỉ kiểm tra trên một video (`Demo-Module-5.mp4`, 60.4 giây) | Trung bình |
| Video có ít phụ đề gắn sẵn tạo quá ít segment OCR (VD: Demo-Module-1: 2 segment, 1 tường thuật) | Trung bình |
| Web UI chưa được kiểm tra với pipeline thực tế | Thấp |

---

## 5. Giai đoạn 3 — Các công việc còn lại (26 tháng 5 – 20 tháng 6, 2026)

### 5.1 Đánh giá & Chỉ số (Ưu tiên: Cao)

- [x] Kết nối Module 5 đánh giá vào pipeline runner để báo cáo chất lượng tự động — `scripts/run_evaluation.py`
- [x] Chạy chấm điểm BLEU-4 và chrF++ so với bản dịch tiếng Việt tham chiếu — `evaluate_translation_from_srts()`
- [x] Đo độ tương đồng giọng nói (resemblyzer cosine) giữa audio tham chiếu và output lồng tiếng — `evaluate_voice_clone()`
- [x] Đo độ chính xác đồng bộ (librosa onset detection) giữa timestamp phụ đề và audio lồng tiếng — `SyncAccuracy.evaluate_against_merged_audio()`
- [x] Đo độ lệch timestamp chế độ OCR so với phụ đề ground-truth (mean/median/p95) — `compare_caption_tracks()`
- [ ] Đo precision/recall của bộ phân loại tường thuật

### 5.2 Kiểm tra đa video (Ưu tiên: Cao)

- [ ] Kiểm tra trên ít nhất 2–3 video không có audio với tường thuật gắn sẵn, độ dài khác nhau
- [ ] Kiểm tra trên video có phụ đề dài hơn (kiểm tra độ trễ OCR và ghép segment)
- [ ] Ghi nhận các lỗi và trường hợp biên phát hiện được

### 5.3 Cải thiện chất lượng (Ưu tiên: Trung bình)

- [ ] Hiệu chỉnh `VI_CHARS_PER_SEC` theo từng giọng bằng cách đo tốc độ nói từ output TTS, thay thế giá trị cố định 15.0
- [ ] Tự động phát hiện vị trí vùng phụ đề bằng cách phân cụm tọa độ Y bounding-box OCR, thay thế `CAPTION_BAND_RATIO` cố định
- [x] Tinh chỉnh các tham số chế độ OCR (FPS lấy mẫu 2→3, giải mã frame tuần tự, loại trùng văn bản giống nhau trong ghép) dựa trên kết quả đánh giá

### 5.4 Tích hợp Web UI (Ưu tiên: Trung bình)

- [ ] Hiển thị lựa chọn `--mode {vlm, ocr}` trong giao diện Gradio
- [ ] Kết nối callback tiến độ pipeline với UI để hiển thị trạng thái thời gian thực
- [ ] Kiểm tra luồng hoàn chỉnh upload → xử lý → download qua giao diện web

### 5.5 Tài liệu & Viết luận văn (Ưu tiên: Cao)

- [ ] Tạo báo cáo đánh giá với đầy đủ chỉ số cho phụ lục luận văn
- [ ] Tài liệu hóa kiến trúc hệ thống và các quyết định thiết kế
- [ ] Chuẩn bị video demo thể hiện toàn bộ pipeline hoạt động
- [ ] Hoàn thành viết luận văn

### 5.6 Nếu còn thời gian

- [ ] Đánh giá hiệu năng chế độ OCR trên video 10+ phút
