"""
GLM-OCR — Text extraction from video frames using GLM-OCR model.

Uses zai-org/GLM-OCR (0.9B params) via Hugging Face Transformers.
Extracts on-screen text (code, slides, captions) to augment VLM context,
reducing hallucination and improving translation accuracy.
"""

import io
from pathlib import Path
from typing import List, Optional

import numpy as np
from loguru import logger


class GLMOCR:
    """GLM-OCR wrapper for on-screen text extraction.

    Supports both HuggingFace model ID (auto-downloads) and local model path.

    Usage:
        ocr = GLMOCR(model_path="zai-org/GLM-OCR")  # auto-download
        ocr = GLMOCR(model_path="./models/glm-ocr")  # local
        text = ocr.extract_text(frame)
    """

    def __init__(
        self,
        model_path: str = "zai-org/GLM-OCR",
        device: str = "cuda",
        torch_dtype: str = "auto",
    ):
        """
        Args:
            model_path: HuggingFace model ID or local path.
            device: Device for inference ('cuda', 'cpu', or 'auto').
            torch_dtype: Torch dtype ('auto', 'float16', 'bfloat16').
        """
        self.model_path = model_path
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy load model to save VRAM."""
        if self._model is not None:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        logger.info(f"Loading GLM-OCR from {self.model_path}...")

        self._processor = AutoProcessor.from_pretrained(self.model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            pretrained_model_name_or_path=self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device == "auto" else None,
        )

        # If device_map was not "auto", manually move to device
        if self.device != "auto":
            import torch
            device = torch.device(self.device if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(device)

        self._model.eval()
        logger.info(f"GLM-OCR loaded on {self._model.device}")

    def extract_text(
        self,
        frame: np.ndarray,
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 8192,
    ) -> str:
        """
        Extract text from a single frame.

        Args:
            frame: Frame as numpy array (BGR or RGB, HWC format).
            prompt: OCR prompt (default: "Text Recognition:").
            max_new_tokens: Max tokens to generate.

        Returns:
            Extracted text string.
        """
        from PIL import Image

        self._load_model()

        # Convert BGR (OpenCV) to RGB PIL Image
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = frame[:, :, ::-1]  # BGR -> RGB
        else:
            rgb_frame = frame

        pil_image = Image.fromarray(rgb_frame)

        return self._run_inference(pil_image, prompt, max_new_tokens)

    def extract_text_from_file(
        self,
        image_path: str | Path,
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 8192,
    ) -> str:
        """
        Extract text from an image file.

        Args:
            image_path: Path to the image file.
            prompt: OCR prompt.
            max_new_tokens: Max tokens to generate.

        Returns:
            Extracted text string.
        """
        from PIL import Image

        self._load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = Image.open(str(image_path)).convert("RGB")

        return self._run_inference(pil_image, prompt, max_new_tokens)

    def _run_inference(
        self,
        pil_image,
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 8192,
    ) -> str:
        """
        Run GLM-OCR inference on a PIL image.

        Args:
            pil_image: PIL Image object.
            prompt: OCR prompt text.
            max_new_tokens: Max tokens to generate.

        Returns:
            Decoded output text.
        """
        import torch

        # Build chat message in GLM-OCR format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Tokenize
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        # Remove token_type_ids if present (some models don't need it)
        inputs.pop("token_type_ids", None)

        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        # Decode only the new tokens (skip input tokens)
        output_text = self._processor.decode(
            generated_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return output_text.strip()

    def extract_text_batch(
        self,
        frames: List[np.ndarray],
        prompt: str = "Text Recognition:",
    ) -> List[str]:
        """
        Extract text from multiple frames.

        Args:
            frames: List of frames as numpy arrays.
            prompt: OCR prompt.

        Returns:
            List of extracted text strings.
        """
        results = []
        for i, frame in enumerate(frames):
            try:
                text = self.extract_text(frame, prompt=prompt)
                results.append(text)
                logger.debug(f"GLM-OCR frame {i+1}/{len(frames)}: {len(text)} chars")
            except Exception as e:
                logger.warning(f"GLM-OCR failed on frame {i+1}: {e}")
                results.append("")
        return results

    def unload_model(self):
        """Unload model from memory to free VRAM."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GLM-OCR model unloaded")
