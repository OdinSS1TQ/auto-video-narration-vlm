"""Quick test: verify Gemini API key works with new google-genai package."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
print(f"API Key: {api_key[:10]}...{api_key[-4:]}")

from google import genai

client = genai.Client(api_key=api_key)

# Test 1: List available models
print("\n── Available Gemini models ──")
for m in client.models.list():
    print(f"  {m.name}")

# Test 2: Simple text call (no images, minimal tokens)
print("\n── Test simple text call ──")

models_to_try = [
    "gemini-2.5-flash-lite",   # Best: 1000 RPD, 15 RPM
    "gemini-2.5-flash",        # Good: 250 RPD, 10 RPM
    "gemini-2.0-flash-lite",   # Fallback
    "gemini-2.0-flash",        # Fallback
]

for model_name in models_to_try:
    print(f"\n  Trying {model_name}...", end=" ", flush=True)
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Say 'hello' in Vietnamese. Reply with ONLY the word.",
        )
        print(f"✓ Response: {response.text.strip()}")
    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "429" in err or "resource_exhausted" in err.lower():
            print(f"✗ Rate limited")
        elif "not found" in err.lower() or "404" in err:
            print(f"✗ Model not found")
        else:
            print(f"✗ {err[:120]}")

# Test 3: Image test (small synthetic image)
print("\n── Test with image ──")
import base64
import numpy as np

try:
    # Create tiny test image
    img = np.ones((100, 200, 3), dtype=np.uint8) * 200
    import cv2
    _, buf = cv2.imencode(".jpg", img)
    b64 = base64.b64encode(buf).decode()

    from google.genai import types

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg"),
                    types.Part.from_text("Describe this image in 5 words."),
                ],
            )
        ],
    )
    print(f"  ✓ Image test: {response.text.strip()}")
except Exception as e:
    err = str(e)
    if "quota" in err.lower() or "429" in err:
        print(f"  ✗ Rate limited (try again in ~1 min)")
    else:
        print(f"  ✗ {err[:120]}")
