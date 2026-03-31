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

## 2. Qwen2.5-VL-3B-Instruct (Local VLM)

### Option A: Tải bằng script (khuyến nghị — ~6GB VRAM)
```bash
python scripts/download_models.py --model qwen2.5-vl
```

### Option B: Tải bằng hf CLI
```bash
hf download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen2.5-vl-3b
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

## Kiểm tra cài đặt

```bash
# Liệt kê các model có thể tải
python scripts/download_models.py --list

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
├── qwen2.5-vl-3b/     # Qwen2.5-VL-3B full HF
└── vitts/              # TTS models (Module 2)
```
