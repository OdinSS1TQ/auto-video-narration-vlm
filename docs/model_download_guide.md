# Hướng dẫn tải Model Weights

## 1. GLM-OCR 0.9B (OCR cho video frames)

### Option A: Tự động (đơn giản nhất)
Không cần tải trước! Chỉ cần dùng HuggingFace model ID — sẽ tự tải khi chạy:
```python
from src.m1_vlm.glm_ocr import GLMOCR
ocr = GLMOCR(model_path="zai-org/GLM-OCR")  # auto-download lần đầu
```

### Option B: Tải trước bằng script
```bash
python scripts/download_models.py --model glm-ocr
```

### Option C: Tải bằng hf CLI
```bash
pip install -U "huggingface_hub[cli]"
hf download zai-org/GLM-OCR --local-dir ./models/glm-ocr
```

### Option D: Dùng Ollama
```bash
ollama pull glm-ocr
ollama run glm-ocr "Text Recognition:" ./test_image.png
```

> **Thông số**: 0.9B params, ~2GB VRAM, hỗ trợ transformers/vLLM/SGLang/Ollama

---

## 2. Qwen3.5-0.8B ⭐ (Default Local VLM — Khuyên dùng)

Model multimodal thế hệ mới, nhỏ gọn nhưng mạnh mẽ:
- **0.8B params**, chỉ ~1.6GB VRAM (BF16) hoặc ~0.5GB (4-bit)
- Context window 262K tokens — xử lý video dài tốt hơn
- Benchmark tương đương hoặc tốt hơn Qwen2.5-VL-3B (gấp 3.75x nhỏ hơn!)
- Hỗ trợ thinking mode cho reasoning phức tạp

### Option A: Tải bằng script (khuyến nghị)
```bash
python scripts/download_models.py --model qwen3.5
```

### Option B: Tải bằng hf CLI
```bash
hf download Qwen/Qwen3.5-0.8B --local-dir ./models/qwen3.5-0.8b
```

### Option C: Dùng Gemini API (không cần tải model)
Nếu chưa có GPU, dùng Gemini API miễn phí:
```bash
# Trong .env
VLM_MODE=api
GEMINI_API_KEY=your_key_here
```
Lấy API key tại: https://aistudio.google.com/apikey

---

## 3. Qwen2.5-VL-3B-Instruct (Legacy VLM)

> ⚠ **Legacy**: Đã được thay thế bởi Qwen3.5-0.8B làm default.
> Vẫn hỗ trợ nếu bạn đã tải sẵn hoặc cần so sánh benchmark.

### Tải bằng script
```bash
python scripts/download_models.py --model qwen2.5-vl
```

### Tải bằng hf CLI
```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen2.5-vl-3b
```

### Sử dụng legacy model
```bash
# Khi chạy VLM extract
python scripts/run_vlm_extract.py --video video.mp4 --model ./models/qwen2.5-vl-3b

# Hoặc dùng flag tiện
python scripts/run_vlm_extract.py --video video.mp4 --qwen25
```

> **Thông số**: 3B params, ~6GB VRAM, chuyên biệt cho VLM tasks

---

## Kiểm tra cài đặt

```bash
# Liệt kê các model có thể tải
python scripts/download_models.py --list

# Test tất cả model (GLM-OCR + Qwen3.5 + Qwen2.5)
python scripts/test_model_load.py

# Test GLM-OCR
python -c "
from src.m1_vlm.glm_ocr import GLMOCR
ocr = GLMOCR(model_path='zai-org/GLM-OCR')
print(ocr.extract_text_from_file('test_image.png'))
"
```

## Cấu trúc thư mục models/ sau khi tải

```
models/
├── glm-ocr/           # GLM-OCR 0.9B weights
├── qwen3.5-0.8b/      # Qwen3.5-0.8B (default) ⭐
├── qwen2.5-vl-3b/     # Qwen2.5-VL-3B (legacy)
└── vitts/              # TTS models (Module 2)
```

## So sánh Qwen3.5 vs Qwen2.5-VL

| Tiêu chí | Qwen3.5-0.8B ⭐ | Qwen2.5-VL-3B |
|:---|:---|:---|
| **VRAM** | ~1.6GB | ~6GB |
| **Context** | 262K tokens | ~8-32K |
| **VideoMME** | ~63.8 | ~61 |
| **OCRBench** | ~74.5 | ~74 |
| **Tốc độ** | Nhanh hơn (nhỏ hơn) | Chậm hơn |
