"""
Comprehensive test script for m2_tts module (VieNeu-TTS v2 Turbo).

Usage:
    # Unit tests only (no GPU, no model download)
    python -m scripts.test_m2_tts --unit

    # Integration tests (requires GPU + vieneu SDK)
    python -m scripts.test_m2_tts --integration

    # All tests
    python -m scripts.test_m2_tts --all

    # Quick smoke (unit + critical integration)
    python -m scripts.test_m2_tts --quick

Test coverage:
  TTSClient:
    - instantiation & config validation
    - lazy model loading guard
    - text normalization (Vietnamese text, edge cases)
    - cache key stability
    - encode_reference / synthesize / synthesize_to_file
    - preset voice listing + synthesis
    - context manager (__enter__ / __exit__)
    - error paths (invalid engine, missing file, empty text)

  SpeakerEncoder:
    - backend selection
    - VieNeu reference encoding (delegates to TTSClient)
    - resemblyzer embedding extraction + disk caching
    - cosine similarity computation
    - cache management

  BatchInference:
    - batch preparation (sizing, splitting)
    - process_all (async) with ref_codes / reference_audio
    - empty text segment handling
    - text truncation for long segments
    - partial failure tolerance
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

# Fix Windows cp1252 console — Vietnamese chars from vieneu SDK logs
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_DIR = ROOT / "debug_output" / "m2_tts_test"


# ==============================================================================═
# Helpers
# ==============================================================================═

class TestResult:
    """Simple test result collector."""

    def __init__(self):
        self.passed: List[str] = []
        self.failed: List[Tuple[str, str]] = []

    def ok(self, name: str):
        self.passed.append(name)
        print(f"  PASS  {name}")

    def fail(self, name: str, reason: str):
        self.failed.append((name, reason))
        print(f"  FAIL  {name}: {reason}")

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.ok(name)
        else:
            self.fail(name, detail or "assertion failed")

    @property
    def total(self):
        return len(self.passed) + len(self.failed)

    def summary(self) -> str:
        lines = [f"Results: {len(self.passed)}/{self.total} passed"]
        if self.failed:
            lines.append("Failed:")
            for name, reason in self.failed:
                lines.append(f"  - {name}: {reason}")
        else:
            lines.append("ALL TESTS PASSED")
        return "\n".join(lines)


def make_test_wav(
    path: Path,
    duration_sec: float = 3.0,
    sample_rate: int = 24000,
    frequency: float = 220.0,
) -> Path:
    """Create a minimal sine-wave WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    samples = (np.sin(2 * np.pi * frequency * t) * 0.3).astype(np.float32)
    samples_int16 = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
    return path


def make_realistic_wav(
    path: Path,
    duration_sec: float = 4.0,
    sample_rate: int = 24000,
) -> Path:
    """Create a WAV that somewhat resembles speech (mixed harmonics + envelope)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # Mix several harmonics in voice frequency range
    signal = (
        0.4 * np.sin(2 * np.pi * 150 * t)   # fundamental
        + 0.3 * np.sin(2 * np.pi * 300 * t)  # 2nd harmonic
        + 0.2 * np.sin(2 * np.pi * 450 * t)  # 3rd harmonic
        + 0.1 * np.sin(2 * np.pi * 600 * t)  # 4th harmonic
    )
    # Add amplitude modulation (speech-like envelope)
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)  # 3 Hz modulation
    signal *= envelope

    samples = (signal * 0.3).astype(np.float32)
    samples_int16 = (samples * 32767).astype(np.int16)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())
    return path


# ==============================================================================═
# UNIT TESTS — no GPU, no model download required
# ==============================================================================═

def test_imports(r: TestResult):
    """Verify all public symbols are importable."""
    print("\n-- Imports --")
    try:
        from src.m2_tts import TTSClient, SpeakerEncoder, BatchInference
        from src.m2_tts.tts_client import VIENEU_TURBO_REPO, VIENEU_SAMPLE_RATE
        r.check("import_TTSClient", TTSClient is not None)
        r.check("import_SpeakerEncoder", SpeakerEncoder is not None)
        r.check("import_BatchInference", BatchInference is not None)
        r.check("const_REPO", VIENEU_TURBO_REPO == "pnnbao-ump/VieNeu-TTS-v2-Turbo")
        r.check("const_SAMPLE_RATE", VIENEU_SAMPLE_RATE == 24_000)
    except Exception as exc:
        r.fail("imports", str(exc))


def test_tts_client_instantiation(r: TestResult):
    """TTSClient construction — default and custom params."""
    print("\n-- TTSClient Instantiation --")
    from src.m2_tts.tts_client import TTSClient

    # Default engine
    client = TTSClient()
    r.check("default_engine", client.engine == "vieneu")
    r.check("default_sample_rate", client.sample_rate == 24_000)
    r.check("default_mode", client.vieneu_mode == "turbo")
    r.check("default_lazy", client._tts is None, "Model should not be loaded at construction")

    # Custom params
    client2 = TTSClient(
        engine="vieneu",
        vieneu_mode="standard",
        backbone_repo="custom/repo",
        backbone_device="cpu",
        codec_device="cpu",
        codec_repo="neuphonic/distill-neucodec",
        sample_rate=16000,
    )
    r.check("custom_repo", client2.backbone_repo == "custom/repo")
    r.check("custom_device", client2.backbone_device == "cpu")
    r.check("custom_mode", client2.vieneu_mode == "standard")
    r.check("custom_sr", client2.sample_rate == 16000)


def test_text_normalization(r: TestResult):
    """Text normalization edge cases."""
    print("\n-- Text Normalization --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()

    cases = [
        ("hello world", "hello world", "basic text unchanged"),
        ("  spaces  ", "spaces", "leading/trailing whitespace stripped"),
        ("multiple   spaces", "multiple spaces", "excess whitespace collapsed"),
        ("wait...", "wait…", "ellipsis normalized to single char"),
        ("", "", "empty string stays empty"),
        ("  ", "", "whitespace-only becomes empty"),
        ("Xin chào! Đây là test.", "Xin chào! Đây là test.", "Vietnamese text preserved"),
        ("Line1\nLine2\tTab", "Line1 Line2 Tab", "newlines and tabs collapsed"),
    ]
    for input_text, expected, desc in cases:
        result = client._normalize_text(input_text)
        r.check(f"norm[{desc}]", result == expected, f"got '{result}', expected '{expected}'")


def test_cache_key_stability(r: TestResult):
    """Cache key is deterministic for the same file."""
    print("\n-- Cache Key Stability --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    test_file = OUTPUT_DIR / "cache_test.wav"
    make_test_wav(test_file)

    key1 = TTSClient._make_cache_key(test_file)
    key2 = TTSClient._make_cache_key(test_file)
    r.check("same_key_same_file", key1 == key2, f"{key1} != {key2}")

    # Different file -> different key
    test_file2 = OUTPUT_DIR / "cache_test_2.wav"
    make_test_wav(test_file2, frequency=440.0)
    key3 = TTSClient._make_cache_key(test_file2)
    r.check("diff_key_diff_file", key1 != key3)


def test_invalid_engine(r: TestResult):
    """Reject unknown engine at load time."""
    print("\n-- Invalid Engine --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient(engine="nonexistent")
    try:
        client._load_model()
        r.fail("invalid_engine", "Expected ValueError for unknown engine")
    except ValueError as exc:
        r.check("invalid_engine_raises", "Unknown TTS engine" in str(exc))
    except Exception as exc:
        r.fail("invalid_engine", f"Wrong exception type: {type(exc).__name__}: {exc}")


def test_reserved_engine(r: TestResult):
    """f5-tts and vixtts engines raise NotImplementedError."""
    print("\n-- Reserved Engine --")
    from src.m2_tts.tts_client import TTSClient

    for engine_name in ("f5-tts", "vixtts"):
        client = TTSClient(engine=engine_name)
        try:
            client._load_model()
            r.fail(f"reserved_{engine_name}", "Expected NotImplementedError")
        except NotImplementedError as exc:
            r.check(f"reserved_{engine_name}", "reserved for future" in str(exc).lower() or "not implemented" in str(exc).lower())
        except Exception as exc:
            r.fail(f"reserved_{engine_name}", f"Wrong exception: {exc}")


def test_speaker_encoder_instantiation(r: TestResult):
    """SpeakerEncoder construction — both backends."""
    print("\n-- SpeakerEncoder Instantiation --")
    from src.m2_tts.speaker_encoder import SpeakerEncoder

    enc_resem = SpeakerEncoder(backend="resemblyzer", cache_dir=str(OUTPUT_DIR / "spk_cache"))
    r.check("resemblyzer_backend", enc_resem.backend == "resemblyzer")
    r.check("resemblyzer_lazy", enc_resem._encoder is None)

    enc_vn = SpeakerEncoder(backend="vieneu", cache_dir=str(OUTPUT_DIR / "spk_cache_vn"))
    r.check("vieneu_backend", enc_vn.backend == "vieneu")


def test_cosine_similarity(r: TestResult):
    """Cosine similarity math — known vectors."""
    print("\n-- Cosine Similarity --")
    from src.m2_tts.speaker_encoder import SpeakerEncoder

    enc = SpeakerEncoder(backend="resemblyzer")

    # Identical vectors -> 1.0
    a = np.array([1.0, 0.0, 0.0])
    sim_same = enc.compute_similarity(a, a)
    r.check("sim_identical", abs(sim_same - 1.0) < 1e-6, f"got {sim_same}")

    # Orthogonal -> 0.0
    b = np.array([0.0, 1.0, 0.0])
    sim_ortho = enc.compute_similarity(a, b)
    r.check("sim_orthogonal", abs(sim_ortho) < 1e-6, f"got {sim_ortho}")

    # Opposite -> -1.0
    c = np.array([-1.0, 0.0, 0.0])
    sim_opp = enc.compute_similarity(a, c)
    r.check("sim_opposite", abs(sim_opp + 1.0) < 1e-6, f"got {sim_opp}")

    # Zero vector -> 0.0 (guard against division by zero)
    zero = np.zeros(3)
    sim_zero = enc.compute_similarity(a, zero)
    r.check("sim_zero_vector", sim_zero == 0.0, f"got {sim_zero}")

    # Random vectors -> value in [-1, 1]
    rng = np.random.default_rng(42)
    d = rng.standard_normal(256).astype(np.float32)
    e = rng.standard_normal(256).astype(np.float32)
    sim_rand = enc.compute_similarity(d, e)
    r.check("sim_in_range", -1.0 <= sim_rand <= 1.0, f"got {sim_rand}")


def test_speaker_encoder_cache(r: TestResult):
    """Disk caching for resemblyzer embeddings (mocked encoder)."""
    print("\n-- SpeakerEncoder Cache --")
    from src.m2_tts.speaker_encoder import SpeakerEncoder

    cache_dir = OUTPUT_DIR / "spk_test_cache"
    enc = SpeakerEncoder(backend="resemblyzer", cache_dir=str(cache_dir))

    test_wav = OUTPUT_DIR / "spk_test_ref.wav"
    make_realistic_wav(test_wav)

    # Patch resemblyzer to avoid needing the actual model.
    # resemblyzer is imported inside the methods, so we patch the top-level module.
    fake_embedding = np.random.default_rng(123).standard_normal(256).astype(np.float32)
    mock_voice_encoder = MagicMock()
    mock_voice_encoder.embed_utterance.return_value = fake_embedding

    mock_resemblyzer = MagicMock()
    mock_resemblyzer.VoiceEncoder.return_value = mock_voice_encoder
    mock_resemblyzer.preprocess_wav.return_value = np.zeros(24000)

    with patch.dict("sys.modules", {"resemblyzer": mock_resemblyzer}):
        # Reset lazy encoder so it picks up the mock
        enc._encoder = None

        # First call: computes and caches
        result1 = enc.extract_embedding(test_wav, use_cache=True)
        np.testing.assert_array_equal(result1, fake_embedding)
        r.ok("cache_write")

        # Verify .npy file exists on disk
        cache_path = cache_dir / f"{test_wav.stem}.npy"
        r.check("cache_file_exists", cache_path.exists())

        # Second call: reads from cache (no re-encoding)
        result2 = enc.extract_embedding(test_wav, use_cache=True)
        np.testing.assert_array_equal(result2, fake_embedding)
        r.ok("cache_read")

    # clear_cache
    enc.clear_cache()
    r.check("cache_cleared", not cache_dir.exists() or not list(cache_dir.iterdir()))


def test_batch_inference_init(r: TestResult):
    """BatchInference construction."""
    print("\n-- BatchInference Init --")
    from src.m2_tts.batch_inference import BatchInference
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    bi = BatchInference(tts_client=client, max_batch_size=4, max_text_length=100)
    r.check("bi_client", bi.tts_client is client)
    r.check("bi_batch_size", bi.max_batch_size == 4)
    r.check("bi_max_length", bi.max_text_length == 100)


def test_batch_preparation(r: TestResult):
    """BatchInference.prepare_batches — sizing and splitting logic."""
    print("\n-- Batch Preparation --")
    from src.m2_tts.batch_inference import BatchInference
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    bi = BatchInference(tts_client=client, max_batch_size=3, max_text_length=50)

    # 5 segments -> should split into 2 batches (3 + 2)
    segments = [
        {"translated_text": f"Short text {i}"} for i in range(5)
    ]
    batches = bi.prepare_batches(segments)
    r.check("batch_count_5seg", len(batches) == 2, f"got {len(batches)} batches")
    r.check("batch1_size", len(batches[0]) == 3)
    r.check("batch2_size", len(batches[1]) == 2)

    # Empty input
    empty_batches = bi.prepare_batches([])
    r.check("batch_empty", empty_batches == [])

    # Single segment
    single_batches = bi.prepare_batches([{"translated_text": "Hello"}])
    r.check("batch_single", len(single_batches) == 1)
    r.check("batch_single_size", len(single_batches[0]) == 1)

    # Exactly at batch boundary
    exact_segments = [{"translated_text": "A"} for _ in range(3)]
    exact_batches = bi.prepare_batches(exact_segments)
    r.check("batch_exact_boundary", len(exact_batches) == 1)
    r.check("batch_exact_size", len(exact_batches[0]) == 3)

    # Over text length limit forces new batch
    long_seg = {"translated_text": "A" * 300}  # 300 chars >> max_text_length * max_batch_size
    short_seg = {"translated_text": "B"}
    mixed_batches = bi.prepare_batches([long_seg, short_seg])
    r.check("batch_long_split", len(mixed_batches) == 2, f"got {len(mixed_batches)}")


def test_batch_preparation_missing_key(r: TestResult):
    """Segments without 'translated_text' get empty string."""
    print("\n-- Batch Prep Missing Key --")
    from src.m2_tts.batch_inference import BatchInference
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    bi = BatchInference(tts_client=client, max_batch_size=8, max_text_length=200)

    segments = [
        {"translated_text": "Normal"},
        {"wrong_key": "Oops"},
        {"translated_text": "Also normal"},
    ]
    batches = bi.prepare_batches(segments)
    r.check("batch_missing_key_count", len(batches) >= 1)
    # All 3 should end up in batch(es) without crash
    total = sum(len(b) for b in batches)
    r.check("batch_missing_key_total", total == 3, f"got {total}")


def run_unit_tests() -> TestResult:
    """Run all unit tests (no GPU needed)."""
    r = TestResult()
    print("=" * 60)
    print("  m2_tts Unit Tests (no GPU required)")
    print("=" * 60)

    test_imports(r)
    test_tts_client_instantiation(r)
    test_text_normalization(r)
    test_cache_key_stability(r)
    test_invalid_engine(r)
    test_reserved_engine(r)
    test_speaker_encoder_instantiation(r)
    test_cosine_similarity(r)
    test_speaker_encoder_cache(r)
    test_batch_inference_init(r)
    test_batch_preparation(r)
    test_batch_preparation_missing_key(r)

    return r


# ==============================================================================═
# INTEGRATION TESTS — requires GPU + vieneu SDK installed
# ==============================================================================═

def test_model_load_and_preset_voices(r: TestResult):
    """Load VieNeu model + list preset voices."""
    print("\n-- Model Load + Preset Voices --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    r.check("model_lazy", client._tts is None)

    voices = client.list_preset_voices()
    r.check("voices_is_list", isinstance(voices, list))
    r.check("voices_not_empty", len(voices) > 0, f"got {len(voices)} voices")

    # Each entry should be (desc, id) tuple
    first_voice = voices[0]
    r.check("voice_tuple_format", isinstance(first_voice, tuple) and len(first_voice) == 2)

    print(f"  Found {len(voices)} preset voices:")
    for desc, vid in voices[:5]:
        # Safely print — Vietnamese names may fail on Windows cp1252 console
        try:
            print(f"    - [{vid}] {desc}")
        except UnicodeEncodeError:
            print(f"    - [{vid}] (voice name has Vietnamese chars)")

    client.unload_model()
    r.check("model_unloaded", client._tts is None)
    return client, voices


def test_preset_synthesis(r: TestResult, client, voices):
    """Synthesize with preset voice (no reference audio)."""
    print("\n-- Preset Voice Synthesis --")
    from src.m2_tts.tts_client import TTSClient

    # Re-create client since we unloaded
    client = TTSClient()
    voice_id = voices[0][1]

    out_path = OUTPUT_DIR / "preset_test.wav"
    audio = client.synthesize_with_preset(
        text="Xin chào! Đây là bài kiểm tra tổng hợp giọng nói tiếng Việt.",
        voice_id=voice_id,
        output_path=out_path,
    )
    r.check("preset_audio_ndarray", isinstance(audio, np.ndarray))
    r.check("preset_audio_nonempty", len(audio) > 0)
    r.check("preset_file_exists", out_path.exists())
    duration = len(audio) / client.sample_rate
    r.check("preset_duration_reasonable", 0.5 < duration < 15.0, f"duration={duration:.2f}s")
    print(f"  Preset voice synthesis: voice={voice_id}, duration={duration:.2f}s -> {out_path}")
    client.unload_model()


def test_encode_reference(r: TestResult):
    """encode_reference with test WAV + cache verification."""
    print("\n-- encode_reference --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    ref_wav = OUTPUT_DIR / "ref_voice.wav"
    make_realistic_wav(ref_wav, duration_sec=4.0)

    # First encode
    t0 = time.perf_counter()
    ref_codes = client.encode_reference(ref_wav, use_cache=True)
    elapsed_first = time.perf_counter() - t0
    r.check("ref_codes_not_none", ref_codes is not None)
    print(f"  First encode: {elapsed_first:.2f}s")

    # Second encode (cache hit — should be near-instant)
    t0 = time.perf_counter()
    ref_codes_cached = client.encode_reference(ref_wav, use_cache=True)
    elapsed_cached = time.perf_counter() - t0
    r.check("cache_is_same_object", ref_codes_cached is ref_codes, "Cache should return same object")
    r.check("cache_is_fast", elapsed_cached < 0.1, f"Cache hit took {elapsed_cached:.4f}s")
    print(f"  Cache hit: {elapsed_cached:.4f}s (same object: {ref_codes_cached is ref_codes})")

    client.unload_model()
    return ref_codes


def test_synthesize_with_ref_codes(r: TestResult, ref_codes):
    """synthesize + synthesize_to_file using cloned voice."""
    print("\n-- Voice-Cloned Synthesis --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()

    # encode_reference again (new client instance)
    ref_wav = OUTPUT_DIR / "ref_voice.wav"
    ref_codes = client.encode_reference(ref_wav, use_cache=True)

    # synthesize_to_file
    out_path = OUTPUT_DIR / "cloned_test.wav"
    client.synthesize_to_file(
        text="Một hệ thống tổng hợp giọng nói tiếng Việt chất lượng cao.",
        output_path=out_path,
        ref_codes=ref_codes,
    )
    r.check("cloned_file_exists", out_path.exists())
    file_size = out_path.stat().st_size
    r.check("cloned_file_nontrivial", file_size > 1000, f"size={file_size}")
    print(f"  Cloned voice -> {out_path} ({file_size:,} bytes)")

    # synthesize (returns ndarray)
    audio = client.synthesize(
        text="Kiểm tra trả về mảng numpy.",
        ref_codes=ref_codes,
    )
    r.check("synthesize_ndarray", isinstance(audio, np.ndarray))
    r.check("synthesize_dtype", audio.dtype == np.float32)
    r.check("synthesize_nonempty", len(audio) > 0)

    client.unload_model()


def test_empty_text_silence(r: TestResult):
    """Empty text returns silence array."""
    print("\n-- Empty Text -> Silence --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    client._load_model()

    audio = client.synthesize(text="")
    r.check("empty_is_ndarray", isinstance(audio, np.ndarray))
    r.check("empty_is_silence", np.allclose(audio, 0.0), f"max abs={np.max(np.abs(audio))}")
    r.check("empty_correct_length", len(audio) == client.sample_rate)

    # Whitespace-only text
    audio_ws = client.synthesize(text="   ")
    r.check("whitespace_is_silence", np.allclose(audio_ws, 0.0))

    client.unload_model()


def test_missing_reference_file(r: TestResult):
    """encode_reference raises FileNotFoundError for missing file."""
    print("\n-- Missing Reference File --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    fake_path = OUTPUT_DIR / "nonexistent_12345.wav"
    try:
        client.encode_reference(fake_path)
        r.fail("missing_ref_file", "Expected FileNotFoundError")
    except FileNotFoundError as exc:
        r.check("missing_ref_file", "not found" in str(exc).lower())
    except Exception as exc:
        r.fail("missing_ref_file", f"Wrong exception: {type(exc).__name__}: {exc}")
    finally:
        client.unload_model()


def test_context_manager(r: TestResult):
    """TTSClient context manager loads + unloads correctly."""
    print("\n-- Context Manager --")
    from src.m2_tts.tts_client import TTSClient

    with TTSClient() as client:
        # Trigger lazy load
        client.list_preset_voices()
        r.check("ctx_model_loaded", client._tts is not None)

    r.check("ctx_model_cleaned", client._tts is None, "Model should be unloaded after context exit")


def test_speaker_encoder_vieneu_path(r: TestResult):
    """SpeakerEncoder.encode_vieneu_reference delegates to TTSClient."""
    print("\n-- SpeakerEncoder VieNeu Path --")
    from src.m2_tts.speaker_encoder import SpeakerEncoder
    from src.m2_tts.tts_client import TTSClient

    enc = SpeakerEncoder(backend="vieneu")
    client = TTSClient()

    ref_wav = OUTPUT_DIR / "ref_voice.wav"
    if not ref_wav.exists():
        make_realistic_wav(ref_wav, duration_sec=4.0)

    ref_codes = enc.encode_vieneu_reference(ref_wav, tts_client=client, use_cache=True)
    r.check("spk_enc_vieneu_result", ref_codes is not None)

    client.unload_model()


async def test_batch_process_all(r: TestResult):
    """BatchInference.process_all end-to-end."""
    print("\n-- BatchInference.process_all --")
    from src.m2_tts.batch_inference import BatchInference
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    bi = BatchInference(tts_client=client, max_batch_size=4, max_text_length=200)

    ref_wav = OUTPUT_DIR / "ref_voice.wav"
    if not ref_wav.exists():
        make_realistic_wav(ref_wav, duration_sec=4.0)

    segments = [
        {"translated_text": "Câu đầu tiên của bài test."},
        {"translated_text": "Đây là câu thứ hai."},
        {"translated_text": ""},  # Empty — should be skipped gracefully
        {"translated_text": "Câu cuối cùng."},
    ]

    batch_out_dir = OUTPUT_DIR / "batch_output"
    results = await bi.process_all(
        segments=segments,
        output_dir=batch_out_dir,
        reference_audio=ref_wav,
    )

    r.check("batch_result_count", len(results) == 4, f"got {len(results)}")
    r.check("batch_seg0_has_audio", results[0].get("audio_path") is not None)
    r.check("batch_seg1_has_audio", results[1].get("audio_path") is not None)
    r.check("batch_seg2_empty", results[2].get("audio_path") is None, "Empty text should have no audio")
    r.check("batch_seg2_error", results[2].get("error") == "empty_text")
    r.check("batch_seg3_has_audio", results[3].get("audio_path") is not None)

    # Verify files exist on disk
    for i in [0, 1, 3]:
        path = Path(results[i]["audio_path"])
        r.check(f"batch_file_{i}_exists", path.exists(), f"{path}")

    # Check ref_codes was encoded only once
    r.check("batch_ref_cache_size", len(client._ref_codes_cache) == 1)

    client.unload_model()


async def test_batch_with_long_text(r: TestResult):
    """BatchInference truncates long text segments."""
    print("\n-- BatchInference Long Text Truncation --")
    from src.m2_tts.batch_inference import BatchInference
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    bi = BatchInference(tts_client=client, max_text_length=50)

    ref_wav = OUTPUT_DIR / "ref_voice.wav"
    if not ref_wav.exists():
        make_realistic_wav(ref_wav, duration_sec=4.0)

    long_text = "Đây là một câu rất dài " * 30  # ~540 chars
    segments = [{"translated_text": long_text}]

    batch_out_dir = OUTPUT_DIR / "batch_long_text"
    results = await bi.process_all(
        segments=segments,
        output_dir=batch_out_dir,
        reference_audio=ref_wav,
    )

    r.check("long_text_result_count", len(results) == 1)
    r.check("long_text_has_audio", results[0].get("audio_path") is not None)
    print(f"  Long text ({len(long_text)} chars) -> truncated to 50, synthesized OK")

    client.unload_model()


def test_multiple_synthesis_rounds(r: TestResult):
    """Synthesize multiple segments sequentially to verify stability."""
    print("\n-- Multiple Sequential Synthesis --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()
    ref_wav = OUTPUT_DIR / "ref_voice.wav"
    if not ref_wav.exists():
        make_realistic_wav(ref_wav, duration_sec=4.0)

    ref_codes = client.encode_reference(ref_wav)

    texts = [
        "Xin chào các bạn.",
        "Hôm nay thời tiết rất đẹp.",
        "Chúng ta sẽ cùng nhau học tập.",
        "Cảm ơn các bạn đã lắng nghe.",
    ]

    out_dir = OUTPUT_DIR / "multi_synthesis"
    for i, text in enumerate(texts):
        out_path = out_dir / f"seg_{i:03d}.wav"
        client.synthesize_to_file(
            text=text,
            output_path=out_path,
            ref_codes=ref_codes,
        )
        r.check(f"multi_synthesis_{i}", out_path.exists(), f"seg {i} missing")
        duration = out_path.stat().st_size / (client.sample_rate * 2)  # rough estimate
        print(f"  [{i+1}/{len(texts)}] {text[:40]}... -> {out_path.name}")

    client.unload_model()


def test_reference_audio_variants(r: TestResult):
    """Verify different reference audio durations work."""
    print("\n-- Reference Audio Duration Variants --")
    from src.m2_tts.tts_client import TTSClient

    client = TTSClient()

    durations = [2.0, 3.0, 5.0, 8.0]
    for dur in durations:
        ref_path = OUTPUT_DIR / f"ref_{dur}s.wav"
        make_realistic_wav(ref_path, duration_sec=dur)
        ref_codes = client.encode_reference(ref_path, use_cache=True)
        r.check(f"ref_{dur}s", ref_codes is not None, f"Failed for {dur}s ref")
        print(f"  {dur}s reference -> encoded OK")

    client.unload_model()


def run_integration_tests() -> TestResult:
    """Run all integration tests (requires GPU + vieneu SDK)."""
    r = TestResult()
    print("\n" + "=" * 60)
    print("  m2_tts Integration Tests (GPU + vieneu SDK required)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client_voices = None
    voices = None

    try:
        client_voices, voices = test_model_load_and_preset_voices(r)
    except Exception as exc:
        r.fail("model_load", str(exc))
        print(f"  Skipping remaining integration tests — model load failed: {exc}")
        return r

    try:
        test_preset_synthesis(r, client_voices, voices)
    except Exception as exc:
        r.fail("preset_synthesis", str(exc))

    ref_codes = None
    try:
        ref_codes = test_encode_reference(r)
    except Exception as exc:
        r.fail("encode_reference", str(exc))

    if ref_codes is not None:
        try:
            test_synthesize_with_ref_codes(r, ref_codes)
        except Exception as exc:
            r.fail("cloned_synthesis", str(exc))

    try:
        test_empty_text_silence(r)
    except Exception as exc:
        r.fail("empty_text", str(exc))

    try:
        test_missing_reference_file(r)
    except Exception as exc:
        r.fail("missing_ref", str(exc))

    try:
        test_context_manager(r)
    except Exception as exc:
        r.fail("context_manager", str(exc))

    try:
        test_speaker_encoder_vieneu_path(r)
    except Exception as exc:
        r.fail("spk_encoder_vieneu", str(exc))

    try:
        asyncio.run(test_batch_process_all(r))
    except Exception as exc:
        r.fail("batch_process", str(exc))

    try:
        asyncio.run(test_batch_with_long_text(r))
    except Exception as exc:
        r.fail("batch_long_text", str(exc))

    try:
        test_multiple_synthesis_rounds(r)
    except Exception as exc:
        r.fail("multi_synthesis", str(exc))

    try:
        test_reference_audio_variants(r)
    except Exception as exc:
        r.fail("ref_variants", str(exc))

    return r


def run_quick_tests() -> TestResult:
    """Unit tests + critical integration path."""
    r = TestResult()
    print("=" * 60)
    print("  m2_tts Quick Tests (unit + critical integration)")
    print("=" * 60)

    # Unit tests
    test_imports(r)
    test_tts_client_instantiation(r)
    test_text_normalization(r)
    test_cache_key_stability(r)
    test_cosine_similarity(r)
    test_batch_preparation(r)

    # Critical integration path
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        client_voices, voices = test_model_load_and_preset_voices(r)
        test_preset_synthesis(r, client_voices, voices)
        ref_codes = test_encode_reference(r)
        if ref_codes is not None:
            test_synthesize_with_ref_codes(r, ref_codes)
    except Exception as exc:
        r.fail("integration_critical", str(exc))

    return r


# ==============================================================================═
# Main
# ==============================================================================═

def main():
    parser = argparse.ArgumentParser(description="m2_tts module tests")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--unit", action="store_true", help="Unit tests only (no GPU)")
    group.add_argument("--integration", action="store_true", help="Integration tests (GPU required)")
    group.add_argument("--all", action="store_true", help="All tests")
    group.add_argument("--quick", action="store_true", help="Unit + critical integration")
    args = parser.parse_args()

    results: List[TestResult] = []

    if args.unit or args.all:
        results.append(run_unit_tests())

    if args.integration or args.all:
        results.append(run_integration_tests())

    if args.quick:
        results.append(run_quick_tests())

    # Final summary
    print("\n" + "=" * 60)
    total_passed = sum(len(r.passed) for r in results)
    total_tests = sum(r.total for r in results)
    all_failed = []
    for r in results:
        all_failed.extend(r.failed)
    print(f"  TOTAL: {total_passed}/{total_tests} passed")
    if all_failed:
        print(f"  FAILED ({len(all_failed)}):")
        for name, reason in all_failed:
            print(f"    - {name}: {reason}")
    else:
        print("  ALL TESTS PASSED")
    print("=" * 60)

    sys.exit(0 if not all_failed else 1)


if __name__ == "__main__":
    main()
