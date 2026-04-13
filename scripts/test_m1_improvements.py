"""
Test script for Module 1 (VLM) improvements.

Tests all P0 + P1 changes:
  1. Scene-Aware Adaptive Frame Sampling
  2. Global Summary Pass (prompt + context)
  3. Overlapping Chunks
  4. SSIM Frame Deduplication

Usage:
    python scripts/test_m1_improvements.py <video_path>

Example:
    python scripts/test_m1_improvements.py data/raw/sample.mp4
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def test_imports():
    """Test 1: Verify all new classes import correctly."""
    divider("TEST 1: Imports")
    
    from src.m1_vlm import (
        SceneDetector,
        FrameExtractor,
        FrameDeduplicator,
        ContextWindow,
        PromptChain,
    )
    
    print("✅ SceneDetector imported")
    print("✅ FrameExtractor imported")
    print("✅ FrameDeduplicator imported")
    print("✅ ContextWindow imported")
    print("✅ PromptChain imported")
    
    return True


def test_scene_detector(video_path: str):
    """Test 2: Scene detection + adaptive timestamps."""
    divider("TEST 2: SceneDetector — Adaptive Timestamps")
    
    from src.m1_vlm.scene_detector import SceneDetector
    
    sd = SceneDetector(threshold=27.0)
    
    # 2a: Detect all scenes
    print(f"[2a] Detecting scenes in: {video_path}")
    t0 = time.time()
    scenes = sd.detect_scenes(video_path)
    t1 = time.time()
    print(f"     Found {len(scenes)} scenes in {t1-t0:.2f}s")
    for i, (start, end) in enumerate(scenes):
        print(f"     Scene {i}: {start:.2f}s — {end:.2f}s (duration: {end-start:.2f}s)")
    
    # 2b: Cache test (should be instant)
    t0 = time.time()
    scenes2 = sd.detect_scenes(video_path)
    t1 = time.time()
    assert scenes == scenes2, "Cache mismatch!"
    print(f"\n[2b] Cache test: {t1-t0:.6f}s (should be ~0s) ✅")
    
    # 2c: Scenes in range
    if scenes:
        mid = (scenes[0][0] + scenes[-1][1]) / 2
        range_scenes = sd.detect_scenes_in_range(video_path, 0, mid)
        print(f"\n[2c] Scenes in range [0, {mid:.1f}s]: {len(range_scenes)} scenes")
    
    # 2d: Adaptive timestamps
    if scenes:
        start, end = scenes[0][0], scenes[-1][1]
        timestamps = sd.get_adaptive_timestamps(video_path, start, end)
        print(f"\n[2d] Adaptive timestamps for full video:")
        print(f"     Total: {len(timestamps)} frames")
        print(f"     Timestamps: {[f'{t:.2f}s' for t in timestamps]}")
        
        # Compare with fixed interval
        duration = end - start
        fixed_count = int(duration / 2.0) + 1
        print(f"\n     ⚡ Comparison:")
        print(f"     Fixed 2s interval: {fixed_count} frames")
        print(f"     Adaptive:          {len(timestamps)} frames")
        diff_pct = (1 - len(timestamps) / max(fixed_count, 1)) * 100
        print(f"     Difference:        {diff_pct:+.0f}% frames")
    
    # 2e: Split into chunks with overlap
    chunks_no_overlap = sd.split_into_chunks(video_path, chunk_duration=45, overlap=0)
    chunks_overlap = sd.split_into_chunks(video_path, chunk_duration=45, overlap=5)
    print(f"\n[2e] Chunks (no overlap): {len(chunks_no_overlap)}")
    for i, (s, e) in enumerate(chunks_no_overlap):
        print(f"     Chunk {i}: {s:.2f}s — {e:.2f}s")
    print(f"     Chunks (5s overlap): {len(chunks_overlap)}")
    for i, (s, e) in enumerate(chunks_overlap):
        print(f"     Chunk {i}: {s:.2f}s — {e:.2f}s")
    
    return True


def test_frame_extractor(video_path: str):
    """Test 3: Frame extraction methods."""
    divider("TEST 3: FrameExtractor — Adaptive + Evenly")
    
    from src.m1_vlm.frame_extractor import FrameExtractor
    from src.m1_vlm.scene_detector import SceneDetector
    
    fe = FrameExtractor(max_width=768)
    sd = SceneDetector()
    
    # 3a: Extract frames evenly (for Global Summary)
    print("[3a] Extracting 10 frames evenly (Global Summary pass)...")
    global_frames = fe.extract_frames_evenly(video_path, n=10)
    print(f"     Got {len(global_frames)} frames")
    if global_frames:
        h, w = global_frames[0].shape[:2]
        print(f"     Frame size: {w}x{h}")
    
    # 3b: Extract at adaptive timestamps
    scenes = sd.detect_scenes(video_path)
    if scenes:
        start, end = scenes[0][0], scenes[-1][1]
        timestamps = sd.get_adaptive_timestamps(video_path, start, end)
        
        print(f"\n[3b] Extracting {len(timestamps)} frames at adaptive timestamps...")
        t0 = time.time()
        adaptive_frames = fe.extract_frames_at_timestamps(video_path, timestamps)
        t1 = time.time()
        print(f"     Got {len(adaptive_frames)} frames in {t1-t0:.2f}s")
        print(f"     Timestamps: {[f'{ts:.2f}s' for ts, _ in adaptive_frames]}")
        
        # 3c: Compare with fixed interval
        print(f"\n[3c] Extracting frames with fixed 2s interval...")
        t0 = time.time()
        fixed_frames = fe.extract_frames_in_range(video_path, start, end, interval=2.0)
        t1 = time.time()
        print(f"     Got {len(fixed_frames)} frames in {t1-t0:.2f}s")
        
        print(f"\n     ⚡ Frame count comparison:")
        print(f"     Fixed 2.0s:  {len(fixed_frames)} frames")
        print(f"     Adaptive:    {len(adaptive_frames)} frames")
    
    # 3d: Base64 encoding
    if global_frames:
        b64 = fe.frame_to_base64(global_frames[0])
        print(f"\n[3d] Base64 encoding: {len(b64)} chars ({len(b64)/1024:.1f} KB)")
    
    return True


def test_frame_dedup(video_path: str):
    """Test 4: SSIM frame deduplication."""
    divider("TEST 4: FrameDeduplicator — SSIM Dedup")
    
    from src.m1_vlm.frame_extractor import FrameExtractor
    from src.m1_vlm.frame_dedup import FrameDeduplicator
    
    fe = FrameExtractor()
    
    # Extract frames at 1s interval (dense sampling for dedup testing)
    print("[4a] Extracting frames at 1s interval for dedup test...")
    frames = fe.extract_frames_in_range(video_path, 0, 30, interval=1.0)
    print(f"     Input: {len(frames)} frames")
    
    # Test different thresholds
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        dedup = FrameDeduplicator(ssim_threshold=threshold)
        result = dedup.deduplicate(frames)
        pct = (1 - len(result) / max(len(frames), 1)) * 100
        print(f"\n[4b] Threshold={threshold}: {len(frames)} → {len(result)} frames "
              f"({pct:.0f}% removed)")
        if result:
            print(f"     Kept timestamps: {[f'{ts:.1f}s' for ts, _ in result]}")
    
    # SSIM profile
    print(f"\n[4c] SSIM similarity profile:")
    dedup = FrameDeduplicator()
    profile = dedup.compute_similarity_profile(frames)
    for ts, score in profile:
        bar = "█" * int(score * 30) if score > 0 else "▁"
        print(f"     {ts:6.1f}s | {score:.3f} {bar}")
    
    return True


def test_prompt_chain():
    """Test 5: Prompt chain with global context."""
    divider("TEST 5: PromptChain — Global Summary + Context Injection")
    
    from src.m1_vlm.prompt_chain import PromptChain
    
    pc = PromptChain(source_lang="English", target_lang="Vietnamese")
    
    # 5a: Global summary prompt
    global_prompt = pc.build_global_summary_prompt()
    print(f"[5a] Global Summary Prompt:")
    print(f"     Length: {len(global_prompt)} chars")
    print(f"     Preview: {global_prompt[:100]}...")
    
    # 5b: Single prompt WITHOUT global context
    prompt_no_global = pc.build_single_prompt(
        chunk_info="Chunk 0: 0.0s — 45.0s",
    )
    
    # 5c: Single prompt WITH global context
    fake_global = '{"topic": "Flask Tutorial", "sections": ["Setup", "Coding", "Testing"]}'
    prompt_with_global = pc.build_single_prompt(
        chunk_info="Chunk 0: 0.0s — 45.0s",
        global_context=fake_global,
    )
    
    print(f"\n[5b] Single prompt WITHOUT global context: {len(prompt_no_global)} chars")
    print(f"[5c] Single prompt WITH global context:    {len(prompt_with_global)} chars")
    print(f"     Difference: +{len(prompt_with_global) - len(prompt_no_global)} chars (global context injected)")
    
    # Verify global context is actually in the prompt
    assert "Video Overview" in prompt_with_global, "Global context not injected!"
    assert "Flask Tutorial" in prompt_with_global, "Global content missing!"
    print(f"     ✅ Global context properly injected into prompt")
    
    # 5d: step1_extract with global context
    step1 = pc.step1_extract(
        chunk_info="Chunk 1: 45.0s — 90.0s",
        global_context=fake_global,
        context_summary="Previous chunk covered project setup.",
    )
    assert "Video Overview" in step1, "Global context not in step1!"
    print(f"\n[5d] step1_extract with global context: {len(step1)} chars ✅")
    
    return True


def test_context_window():
    """Test 6: Context window with global summary."""
    divider("TEST 6: ContextWindow — Hierarchical Context")
    
    from src.m1_vlm.context_window import ContextWindow
    
    cw = ContextWindow(window_size=3)
    
    # 6a: Set global summary
    global_summary = '{"topic": "Flask API Tutorial", "key_terms": ["Flask", "API", "REST"]}'
    cw.set_global_summary(global_summary)
    assert cw.get_global_summary() == global_summary
    print("[6a] Global summary set and retrieved ✅")
    
    # 6b: Add chunk results
    cw.add_chunk_result(0, [
        {"original_text": "Hello world", "translated_text": "Xin chào thế giới"},
    ])
    cw.add_chunk_result(1, [
        {"original_text": "Install Flask", "translated_text": "Cài đặt Flask"},
    ])
    
    context = cw.get_context_summary()
    print(f"\n[6b] Sliding context after 2 chunks:")
    print(f"     {context}")
    
    # 6c: Clear resets both global and sliding
    cw.clear()
    assert cw.get_global_summary() is None
    assert cw.get_context_summary() == ""
    print(f"\n[6c] Clear resets both layers ✅")
    
    return True


def test_full_pipeline_m1(video_path: str):
    """Test 7: Full Module 1 pipeline simulation (without VLM call)."""
    divider("TEST 7: Full M1 Pipeline Simulation")
    
    from src.m1_vlm.scene_detector import SceneDetector
    from src.m1_vlm.frame_extractor import FrameExtractor
    from src.m1_vlm.frame_dedup import FrameDeduplicator
    from src.m1_vlm.context_window import ContextWindow
    from src.m1_vlm.prompt_chain import PromptChain
    
    # Initialize components
    sd = SceneDetector(threshold=27.0)
    fe = FrameExtractor(max_width=768)
    dedup = FrameDeduplicator(ssim_threshold=0.85)
    cw = ContextWindow(window_size=3)
    pc = PromptChain()
    
    print("Step 0: Global Summary Pass")
    global_frames = fe.extract_frames_evenly(video_path, n=15)
    global_images_b64 = fe.frames_to_base64_batch(global_frames)
    global_prompt = pc.build_global_summary_prompt()
    print(f"  → {len(global_frames)} frames sampled, {len(global_prompt)} char prompt")
    print(f"  → Total base64 size: {sum(len(b) for b in global_images_b64) / 1024:.0f} KB")
    # In real pipeline: global_summary = vlm.generate(global_prompt, global_images_b64)
    fake_global = '{"topic": "Demo tutorial", "sections": ["Intro", "Main", "Outro"]}'
    cw.set_global_summary(fake_global)
    print(f"  ✅ Global summary set")
    
    print("\nStep 1: Scene Detection + Chunking")
    chunks = sd.split_into_chunks(video_path, chunk_duration=45, overlap=5)
    scenes = sd.detect_scenes(video_path)
    print(f"  → {len(scenes)} scenes, {len(chunks)} chunks")
    
    print("\nStep 2: Per-Chunk Processing")
    total_fixed = 0
    total_adaptive = 0
    total_deduped = 0
    
    for i, (start, end) in enumerate(chunks):
        # Adaptive timestamps
        timestamps = sd.get_adaptive_timestamps(video_path, start, end)
        
        # Extract frames
        frames = fe.extract_frames_at_timestamps(video_path, timestamps)
        
        # Dedup
        frames_deduped = dedup.deduplicate(frames)
        
        # Compare with fixed
        fixed = fe.extract_frames_in_range(video_path, start, end, interval=2.0)
        
        total_fixed += len(fixed)
        total_adaptive += len(frames)
        total_deduped += len(frames_deduped)
        
        # Build prompt
        prompt = pc.build_single_prompt(
            chunk_info=f"Chunk {i}: {start:.1f}s — {end:.1f}s",
            context_summary=cw.get_context_summary(),
            global_context=cw.get_global_summary(),
        )
        
        print(f"  Chunk {i} [{start:.1f}s-{end:.1f}s]: "
              f"fixed={len(fixed)}, adaptive={len(frames)}, deduped={len(frames_deduped)}, "
              f"prompt={len(prompt)} chars")
    
    print(f"\n📊 SUMMARY:")
    print(f"  Fixed interval (2s):    {total_fixed} total frames")
    print(f"  Adaptive sampling:      {total_adaptive} total frames")
    print(f"  After SSIM dedup:       {total_deduped} total frames")
    if total_fixed > 0:
        savings = (1 - total_deduped / total_fixed) * 100
        print(f"  Token savings:          {savings:.0f}% fewer frames sent to VLM")
    
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_m1_improvements.py <video_path>")
        print("\nExample:")
        print("  python scripts/test_m1_improvements.py data/raw/sample.mp4")
        print("\nRunning import + prompt tests only...\n")
        
        # Run tests that don't need video
        test_imports()
        test_prompt_chain()
        test_context_window()
        
        print(f"\n{'='*60}")
        print("  ✅ All non-video tests passed!")
        print(f"{'='*60}")
        return
    
    video_path = sys.argv[1]
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)
    
    print(f"🎬 Testing with video: {video_path}")
    print(f"   File size: {Path(video_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    tests = [
        ("Imports", lambda: test_imports()),
        ("SceneDetector", lambda: test_scene_detector(video_path)),
        ("FrameExtractor", lambda: test_frame_extractor(video_path)),
        ("FrameDeduplicator", lambda: test_frame_dedup(video_path)),
        ("PromptChain", lambda: test_prompt_chain()),
        ("ContextWindow", lambda: test_context_window()),
        ("Full M1 Pipeline", lambda: test_full_pipeline_m1(video_path)),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print(f"  ✅ All tests passed!")
    else:
        print(f"  ❌ {failed} test(s) failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
