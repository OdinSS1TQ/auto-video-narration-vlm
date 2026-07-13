"""
Quick model load test — Kiểm tra load Qwen3.5-2B và Qwen2.5-VL-3B.

Chạy: python scripts/test_model_load.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def test_qwen35():
    """Test Qwen3.5-0.8B (default local VLM)."""
    print(f"\n{BOLD}{CYAN}{'─' * 50}{RESET}")
    print(f"{BOLD}  1. Qwen3.5-2B (Default VLM){RESET}")
    print(f"{CYAN}{'─' * 50}{RESET}")

    model_path = "./models/qwen3.5-2b"
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen3.5-2B"

    print(f"  Model path: {model_path}")

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
        import torch

        print("  Loading processor...", end="", flush=True)
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path)
        print(f" OK ({time.perf_counter() - t0:.1f}s)")

        print("  Loading model...", end="", flush=True)
        t0 = time.perf_counter()
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
        )
        dt = time.perf_counter() - t0
        print(f" OK ({dt:.1f}s)")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"  Device:     {device}")
        print(f"  Dtype:      {dtype}")
        print(f"  Parameters: {num_params:.0f}M")

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM used:  {vram:.2f} GB")

        # Quick inference test
        print("\n  Running test inference...", end="", flush=True)
        from PIL import Image
        import numpy as np

        img = np.ones((200, 400, 3), dtype=np.uint8) * 200
        pil_img = Image.fromarray(img)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[text], images=[pil_img],
            padding=True, return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
        trimmed = output_ids[0][inputs.input_ids.shape[1]:]
        result = processor.decode(trimmed, skip_special_tokens=True)
        dt = time.perf_counter() - t0
        print(f" OK ({dt:.1f}s)")
        print(f"  VLM output: {result.strip()!r}")

        # Cleanup
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  {GREEN}{BOLD}✓ Qwen3.5-2B — LOAD & INFERENCE OK{RESET}")
        return True

    except Exception as e:
        print(f"\n  {RED}{BOLD}✗ Qwen3.5-2B — FAILED: {e}{RESET}")
        return False


def test_qwen25_vl():
    """Test Qwen2.5-VL-3B (legacy VLM)."""
    print(f"\n{BOLD}{CYAN}{'─' * 50}{RESET}")
    print(f"{BOLD}  2. Qwen2.5-VL-3B-Instruct (Legacy){RESET}")
    print(f"{CYAN}{'─' * 50}{RESET}")

    model_path = "./models/qwen2.5-vl-3b"
    if not Path(model_path).exists():
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

    print(f"  Model path: {model_path}")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        import torch

        print("  Loading processor...", end="", flush=True)
        t0 = time.perf_counter()
        processor = AutoProcessor.from_pretrained(model_path)
        print(f" OK ({time.perf_counter() - t0:.1f}s)")

        print("  Loading model...", end="", flush=True)
        t0 = time.perf_counter()
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto",
        )
        dt = time.perf_counter() - t0
        print(f" OK ({dt:.1f}s)")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        num_params = sum(p.numel() for p in model.parameters()) / 1e6

        print(f"  Device:     {device}")
        print(f"  Dtype:      {dtype}")
        print(f"  Parameters: {num_params:.0f}M")

        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024**3
            print(f"  VRAM used:  {vram:.2f} GB")

        # Quick inference test
        print("\n  Running test inference...", end="", flush=True)
        from PIL import Image
        import numpy as np
        from qwen_vl_utils import process_vision_info

        img = np.ones((200, 400, 3), dtype=np.uint8) * 200
        pil_img = Image.fromarray(img)

        messages = [{"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text", "text": "Describe this image in one sentence."},
        ]}]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=128)
        trimmed = output_ids[0][inputs.input_ids.shape[1]:]
        result = processor.decode(trimmed, skip_special_tokens=True)
        dt = time.perf_counter() - t0
        print(f" OK ({dt:.1f}s)")
        print(f"  VLM output: {result.strip()!r}")

        # Cleanup
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n  {GREEN}{BOLD}✓ Qwen2.5-VL-3B — LOAD & INFERENCE OK{RESET}")
        return True

    except Exception as e:
        print(f"\n  {RED}{BOLD}✗ Qwen2.5-VL-3B — FAILED: {e}{RESET}")
        return False


if __name__ == "__main__":
    print(f"\n{BOLD}{CYAN}{'═' * 50}{RESET}")
    print(f"{BOLD}{CYAN}  MODEL LOAD TEST{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 50}{RESET}")

    import torch
    print(f"\n  Python:  {sys.version.split()[0]}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA:    {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" — {torch.cuda.get_device_name(0)}")
        print(f"  VRAM:    {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print()

    r1 = test_qwen35()
    r2 = test_qwen25_vl()

    print(f"\n{BOLD}{CYAN}{'═' * 50}{RESET}")
    print(f"{BOLD}  SUMMARY{RESET}")
    print(f"{CYAN}{'═' * 50}{RESET}")
    print(f"  Qwen3.5-2B:       {'✓ OK' if r1 else '✗ FAILED'}  {YELLOW}(default){RESET}")
    print(f"  Qwen2.5-VL-3B:    {'✓ OK' if r2 else '✗ FAILED'}  (legacy)")
    print()
