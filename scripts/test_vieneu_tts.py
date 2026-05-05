"""
Smoke test — VieNeu-TTS v2 Turbo Module 2 integration.

Run from project root:
    python -m scripts.test_vieneu_tts

Tests:
  1. Import chain: TTSClient, SpeakerEncoder, BatchInference
  2. TTSClient instantiation (engine=vieneu, device=cuda)
  3. Model lazy-load + list_preset_voices()
  4. Preset voice synthesis (no reference audio needed)
  5. encode_reference() with a generated test wav
  6. synthesize_to_file() with cloned voice
  7. Unload + VRAM cleanup
"""

import sys
import traceback
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def make_test_wav(path: Path, duration_sec: float = 3.0, sample_rate: int = 24000) -> Path:
    """Create a minimal sine-wave WAV for reference audio testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # 220 Hz sine (A3) — close to human voice frequency range
    samples = (np.sin(2 * np.pi * 220 * t) * 0.3).astype(np.float32)
    samples_int16 = (samples * 32767).astype(np.int16)

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
    return path


def run_tests():
    ok = []
    fail = []

    # ── Test 1: Import chain ─────────────────────────────────────────────────
    print("\n[1/6] Import chain...")
    try:
        from src.m2_tts.tts_client import TTSClient, VIENEU_TURBO_REPO, VIENEU_SAMPLE_RATE
        from src.m2_tts.speaker_encoder import SpeakerEncoder
        from src.m2_tts.batch_inference import BatchInference
        assert VIENEU_TURBO_REPO == "pnnbao-ump/VieNeu-TTS-v2-Turbo"
        assert VIENEU_SAMPLE_RATE == 24_000
        print("  OK All imports successful")
        ok.append("imports")
    except Exception as e:
        print(f"  FAIL Import failed: {e}")
        traceback.print_exc()
        fail.append("imports")
        return ok, fail  # Cannot continue without imports

    # ── Test 2: TTSClient instantiation ─────────────────────────────────────
    print("\n[2/6] TTSClient instantiation (no model load yet)...")
    try:
        client = TTSClient(
            engine="vieneu",
            backbone_repo="pnnbao-ump/VieNeu-TTS-v2-Turbo",
            backbone_device="cuda",
            codec_device="cuda",
            vieneu_mode="standard",
        )
        assert client.engine == "vieneu"
        assert client.sample_rate == 24_000
        assert client._tts is None, "Model should NOT be loaded yet (lazy load)"
        print(f"  OK TTSClient created | engine={client.engine} | device={client.backbone_device}")
        ok.append("instantiation")
    except Exception as e:
        print(f"  FAIL TTSClient init failed: {e}")
        traceback.print_exc()
        fail.append("instantiation")
        return ok, fail

    # ── Test 3: Model load + preset voices ──────────────────────────────────
    print("\n[3/6] Loading VieNeu model + list_preset_voices()...")
    print("      (First run downloads model from HuggingFace — may take a few minutes)")
    try:
        voices = client.list_preset_voices()
        assert isinstance(voices, list), "Expected list of (desc, id) tuples"
        assert len(voices) > 0, "Expected at least one preset voice"
        print(f"  OK Model loaded | {len(voices)} preset voices available:")
        for desc, vid in voices[:5]:
            print(f"    - [{vid}] {desc}")
        ok.append("model_load")
    except Exception as e:
        print(f"  FAIL Model load or preset voices failed: {e}")
        traceback.print_exc()
        fail.append("model_load")
        # Try to continue with remaining tests

    # ── Test 4: Preset voice synthesis ──────────────────────────────────────
    print("\n[4/6] Synthesize with preset voice (no reference audio)...")
    try:
        out_dir = ROOT / "debug_output" / "vieneu_smoke_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        voices = client.list_preset_voices()
        default_voice_id = voices[0][1]
        audio = client.synthesize_with_preset(
            text="Xin chào! Đây là bài kiểm tra tổng hợp giọng nói.",
            voice_id=default_voice_id,
            output_path=out_dir / "preset_voice_test.wav",
        )
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        duration = len(audio) / client.sample_rate
        print(f"  OK Preset synthesis OK | voice={default_voice_id} | duration={duration:.2f}s")
        print(f"    Saved → {out_dir / 'preset_voice_test.wav'}")
        ok.append("preset_synthesis")
    except Exception as e:
        print(f"  FAIL Preset synthesis failed: {e}")
        traceback.print_exc()
        fail.append("preset_synthesis")

    # ── Test 5: encode_reference() ───────────────────────────────────────────
    print("\n[5/6] encode_reference() with test WAV...")
    ref_codes = None
    try:
        test_wav = ROOT / "debug_output" / "vieneu_smoke_test" / "test_ref.wav"
        make_test_wav(test_wav, duration_sec=4.0)

        ref_codes = client.encode_reference(test_wav, use_cache=True)
        assert ref_codes is not None
        # Verify in-memory cache works
        ref_codes_cached = client.encode_reference(test_wav, use_cache=True)
        assert ref_codes_cached is ref_codes, "Cache miss — should return same object"
        print(f"  OK encode_reference OK | cache hit verified")
        ok.append("encode_reference")
    except Exception as e:
        print(f"  FAIL encode_reference failed: {e}")
        traceback.print_exc()
        fail.append("encode_reference")

    # ── Test 6: Voice-cloned synthesis ──────────────────────────────────────
    print("\n[6/6] synthesize_to_file() with cloned voice (ref_codes)...")
    try:
        out_path = ROOT / "debug_output" / "vieneu_smoke_test" / "cloned_voice_test.wav"
        client.synthesize_to_file(
            text="Một hệ thống tổng hợp giọng nói tiếng Việt chất lượng cao.",
            output_path=out_path,
            ref_codes=ref_codes,
            ref_text=None,  # Turbo v2 zero-shot
        )
        assert out_path.exists(), "Output file not created"
        file_size = out_path.stat().st_size
        assert file_size > 1000, f"Output file too small: {file_size} bytes"
        print(f"  OK Voice-cloned synthesis OK | output={out_path} ({file_size:,} bytes)")
        ok.append("cloned_synthesis")
    except Exception as e:
        print(f"  FAIL Voice-cloned synthesis failed: {e}")
        traceback.print_exc()
        fail.append("cloned_synthesis")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("\n[Cleanup] Unloading model...")
    try:
        client.unload_model()
        assert client._tts is None
        print("  OK Model unloaded, VRAM freed")
        ok.append("unload")
    except Exception as e:
        print(f"  FAIL Unload failed: {e}")
        fail.append("unload")

    return ok, fail


if __name__ == "__main__":
    print("=" * 60)
    print("  VieNeu-TTS v2 Turbo — Module 2 Smoke Test")
    print("=" * 60)

    ok, fail = run_tests()

    print("\n" + "=" * 60)
    print(f"  Results: {len(ok)} passed / {len(fail)} failed")
    if fail:
        print(f"  FAIL   : {', '.join(fail)}")
    else:
        print("  ALL TESTS PASSED OK")
    print("=" * 60)

    sys.exit(0 if not fail else 1)
