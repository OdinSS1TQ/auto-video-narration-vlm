"""
VLM Client — Gemini / Qwen2.5-VL client for video understanding.

Supports:
  - API mode: Gemini 1.5 Flash (native video understanding, 1M token context)
  - Local mode: Qwen2.5-VL-7B-Instruct via Transformers
"""

import asyncio
import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger


class VLMClient:
    """Unified client for Vision Language Model inference."""

    def __init__(
        self,
        mode: str = "api",
        model_name: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        local_model_path: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        device: str = "auto",
    ):
        """
        Args:
            mode: 'api' for Gemini API or 'local' for Qwen2.5-VL.
            model_name: Model identifier.
            api_key: API key for Gemini (required if mode='api').
            local_model_path: HuggingFace model ID or local path for Qwen2.5-VL.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            device: Device for local model ('auto', 'cuda', 'cpu').
        """
        self.mode = mode
        self.model_name = model_name
        self.api_key = api_key
        self.local_model_path = local_model_path or "Qwen/Qwen2.5-VL-3B-Instruct"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.device = device
        self._client = None
        self._processor = None

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _init_api_client(self):
        """Initialize Gemini API client (google-genai SDK)."""
        if self._client is not None:
            return

        from google import genai

        self._client = genai.Client(api_key=self.api_key)
        logger.info(f"Initialized Gemini API client: {self.model_name}")

    def _init_local_client(self):
        """Initialize local Qwen2.5-VL model via Transformers."""
        if self._client is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Loading Qwen2.5-VL from {self.local_model_path}...")

        self._processor = AutoProcessor.from_pretrained(self.local_model_path)

        self._client = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.local_model_path,
            torch_dtype="auto",
            device_map=self.device,
        )

        self._client.eval()
        logger.info(f"Qwen2.5-VL loaded on {self._client.device}")

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    async def generate(
        self,
        prompt: str,
        images_base64: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> str:
        """
        Generate response from VLM.

        Args:
            prompt: Text prompt.
            images_base64: List of base64-encoded images (for frame-based input).
            video_path: Path to video file (for native video input, Gemini only).

        Returns:
            Model response text.
        """
        if self.mode == "api":
            return await self._generate_api(prompt, images_base64, video_path)
        else:
            return await self._generate_local(prompt, images_base64)

    async def _generate_api(
        self,
        prompt: str,
        images_base64: Optional[List[str]] = None,
        video_path: Optional[str] = None,
        max_retries: int = 5,
        initial_wait: float = 30.0,
    ) -> str:
        """Generate using Gemini API (google-genai SDK) with auto-retry."""
        self._init_api_client()

        from google.genai import types

        # Build content parts
        parts = []

        # Add video if provided
        if video_path:
            video_file = self._client.files.upload(file=video_path)
            # Wait for processing
            import time as _time
            while video_file.state.name == "PROCESSING":
                await asyncio.sleep(2)
                video_file = self._client.files.get(name=video_file.name)
            parts.append(types.Part.from_uri(
                file_uri=video_file.uri, mime_type=video_file.mime_type,
            ))

        # Add images if provided
        if images_base64:
            for img_b64 in images_base64:
                img_bytes = base64.b64decode(img_b64)
                parts.append(types.Part.from_bytes(
                    data=img_bytes, mime_type="image/jpeg",
                ))

        # Add text prompt
        parts.append(types.Part.from_text(prompt))

        contents = [types.Content(role="user", parts=parts)]

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            response_mime_type="application/json",
        )

        # Generate with retry on rate limit
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(
                    self._client.models.generate_content,
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )
                return response.text

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = (
                    "429" in str(e)
                    or "resource_exhausted" in error_str
                    or "quota" in error_str
                    or "rate" in error_str
                )

                if is_rate_limit and attempt < max_retries - 1:
                    wait = initial_wait * (2 ** attempt)
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait:.0f}s..."
                    )
                    print(f"\n    ⏳ Rate limited — waiting {wait:.0f}s "
                          f"(attempt {attempt + 1}/{max_retries})...",
                          end="", flush=True)
                    await asyncio.sleep(wait)
                    print(" retrying", flush=True)
                else:
                    raise

    async def _generate_local(
        self,
        prompt: str,
        images_base64: Optional[List[str]] = None,
    ) -> str:
        """Generate using local Qwen2.5-VL model."""
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        self._init_local_client()

        # Build content list for Qwen2.5-VL message format
        content = []

        # Add images if provided
        if images_base64:
            for img_b64 in images_base64:
                img_bytes = base64.b64decode(img_b64)
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                content.append({
                    "type": "image",
                    "image": pil_image,
                })

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info (Qwen2.5-VL specific)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._client.device)

        # Generate
        with torch.no_grad():
            generated_ids = self._client.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        # Decode only new tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return output_text

    # ------------------------------------------------------------------ #
    # JSON Generation Helper
    # ------------------------------------------------------------------ #

    async def generate_json(
        self,
        prompt: str,
        images_base64: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Any:
        """
        Generate and parse JSON response.

        Args:
            prompt: Text prompt.
            images_base64: List of base64-encoded images.
            video_path: Path to video file.

        Returns:
            Parsed JSON object.

        Raises:
            json.JSONDecodeError: If response is not valid JSON.
        """
        response = await self.generate(prompt, images_base64, video_path)

        # Clean response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]

        return json.loads(cleaned.strip())

    # ------------------------------------------------------------------ #
    # Model Management
    # ------------------------------------------------------------------ #

    def unload_model(self):
        """Unload local model from memory."""
        if self._client is not None and self.mode == "local":
            del self._client
            del self._processor
            self._client = None
            self._processor = None
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Unloaded local VLM model")
