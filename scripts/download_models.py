"""
Auto-download model weights from HuggingFace.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model qwen2.5-vl
    python scripts/download_models.py --model glm-ocr
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MODELS = {
    "qwen2.5-vl": {
        "name": "Qwen2.5-VL-3B-Instruct (Full HuggingFace)",
        "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "output_dir": "models/qwen2.5-vl-3b",
        "description": "Full Qwen2.5-VL-3B for transformers inference (~6GB VRAM)",
    },
    "glm-ocr": {
        "name": "GLM-OCR-0.9B",
        "repo_id": "zai-org/GLM-OCR",
        "output_dir": "models/glm-ocr",
        "description": "Lightweight OCR model (0.9B params, ~2GB VRAM)",
    },
}


def download_model(model_key: str):
    """Download a specific model from HuggingFace."""
    info = MODELS.get(model_key)
    if not info:
        print(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")
        return False

    output_dir = Path(info["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📥 Downloading {info['name']}...")
    print(f"   Repo: {info['repo_id']}")
    print(f"   Output: {output_dir}")
    print()

    try:
        from huggingface_hub import snapshot_download

        downloaded_path = snapshot_download(
            repo_id=info["repo_id"],
            local_dir=str(output_dir),
        )
        print(f"✅ Downloaded to: {downloaded_path}")
        return True

    except ImportError:
        print("❌ huggingface_hub not installed. Install with:")
        print("   pip install huggingface_hub")
        print()
        print("Or download manually with hf CLI:")
        print(f"   hf download {info['repo_id']} --local-dir ./{info['output_dir']}")
        return False

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Try downloading manually with hf CLI:")
        print(f"   hf download {info['repo_id']} --local-dir ./{info['output_dir']}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download model weights for Video Dubbing Vietnamese"
    )
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available models and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("📦 Available models:")
        for key, info in MODELS.items():
            print(f"  {key:20s} — {info['description']}")
        return

    if args.model == "all":
        for key in MODELS:
            download_model(key)
            print()
    else:
        download_model(args.model)


if __name__ == "__main__":
    main()
