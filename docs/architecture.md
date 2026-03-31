# System Architecture

## Pipeline Overview

```
Video Input ──► Scene Detection ──► Frame Extraction ──► GLM-OCR (optional)
                                                            │
                                        VLM (Gemini 1.5 Flash / Qwen2-VL-7B)
                                                            │
                                                     Vietnamese SRT
                                                            │
                  Audio Reference ──► Speaker Encoder ──► Voice Cloning (F5-TTS)
                                                            │
                                                      Audio Chunks
                                                            │
                                              Time Stretcher (Rubberband)
                                                            │
                                              Audio Aligner + FFmpeg
                                                            │
                                                   Dubbed Video Output
```

## Module Dependencies

1. **VLM Module** (`src/vlm/`) — No internal deps, uses external APIs
2. **TTS Module** (`src/tts/`) — No internal deps
3. **Sync Module** (`src/sync/`) — No internal deps
4. **Pipeline** (`src/pipeline/`) — Depends on VLM, TTS, Sync
5. **Evaluation** (`src/evaluation/`) — Depends on TTS (speaker encoder)
6. **Web UI** (`app/`) — Depends on Pipeline

## Data Flow

1. User uploads video + reference audio
2. SceneDetector splits video into chunks
3. FrameExtractor samples key frames per chunk
4. (Optional) GLM-OCR extracts on-screen text
5. VLMClient generates Vietnamese subtitles via prompt chain
6. SRTBuilder merges chunks into .srt file
7. TTSClient generates audio for each subtitle entry
8. AudioAligner adjusts timing (stretch/pad/truncate)
9. FFmpegRenderer merges audio track with original video
10. Output: dubbed video + SRT file
