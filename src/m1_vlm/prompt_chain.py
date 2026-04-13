"""
Prompt Chain — Prompt engineering for VLM-based video subtitle generation.

Architecture: VLM-only (no separate OCR model needed).
The VLM handles both text extraction and translation in a single pass.

Supports two modes:
  - 3-step chain: Extract → Translate → Timestamp (for precision)
  - Single prompt: All-in-one (for speed, recommended for Gemini/large context)
"""

from typing import Any, Dict, List, Optional


# --- JSON Schema for structured output ---

SUBTITLE_ENTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "index": {"type": "integer"},
        "start_time": {"type": "string", "description": "SRT format HH:MM:SS,mmm"},
        "end_time": {"type": "string", "description": "SRT format HH:MM:SS,mmm"},
        "original_text": {"type": "string"},
        "translated_text": {"type": "string"},
    },
    "required": ["index", "start_time", "end_time", "original_text", "translated_text"],
}


class PromptChain:
    """Build prompts for VLM-based video subtitle generation.

    The VLM directly analyzes video frames to:
    1. Read all on-screen text (code, UI, slides, captions)
    2. Identify spoken narration/dialogue
    3. Translate to target language
    4. Generate SRT-formatted subtitles

    No separate OCR model is needed — the VLM handles everything.
    """

    def __init__(
        self,
        source_lang: str = "English",
        target_lang: str = "Vietnamese",
    ):
        self.source_lang = source_lang
        self.target_lang = target_lang

    # ─────────────────────────────────────────────────────────────────
    # Global Summary (Pass 0)
    # ─────────────────────────────────────────────────────────────────

    def build_global_summary_prompt(self) -> str:
        """
        Build prompt for Pass 0: Global Video Understanding.

        AI Researcher Perspective:
            Hierarchical context is critical for coherent narration.
            Without a global overview, each chunk is processed in isolation,
            leading to:
            - Inconsistent terminology across chunks
            - Redundant introductions ("In this tutorial..." repeated)
            - Missing cross-references ("as we saw earlier...")

            Pass 0 samples ~15 frames across the full video to extract
            a high-level outline BEFORE per-chunk processing begins.
            This outline is then injected into every chunk's prompt,
            giving each chunk awareness of the full video structure.

        Returns:
            Prompt string for global video analysis.
        """
        return """You are a video content analyst. Analyze these sampled frames from across the entire video to provide a high-level overview.

## Your Tasks
1. **Identify the main TOPIC** of this video (e.g., "Setting up a Python Flask web app")
2. **List the major SECTIONS** in chronological order (e.g., ["Environment Setup", "Create Project", "Write API Routes", "Testing"])
3. **Extract key TERMS** — technical terms, tool names, library names that appear throughout
4. **Determine the STYLE** — tutorial, demo, presentation, code walkthrough, etc.
5. **Write a brief SUMMARY** (2-3 sentences) of what this video covers

## Output Format
Return ONLY a JSON object:
```json
{
  "topic": "Main topic of the video",
  "style": "tutorial",
  "sections": ["Section 1: ...", "Section 2: ..."],
  "key_terms": ["term1", "term2"],
  "summary": "Brief overview of the entire video content."
}
```

Return ONLY the JSON. No markdown, no explanation."""

    # ─────────────────────────────────────────────────────────────────
    # 3-Step Chain Mode
    # ─────────────────────────────────────────────────────────────────

    def step1_extract(
        self,
        chunk_info: Optional[str] = None,
        context_summary: Optional[str] = None,
        global_context: Optional[str] = None,
    ) -> str:
        """
        Step 1: Extract narration/script from video frames.

        The VLM looks at the screen to understand context,
        but only extracts the narration script (what the speaker
        is explaining or instructing), NOT the literal on-screen text.

        AI Engineer Prompting Design:
            The prompt uses a "observe → describe → narrate" chain-of-thought
            structure that guides the VLM through the cognitive process:
            1. First LOOK at what's on screen (grounding in visual evidence)
            2. Then UNDERSTAND the action being performed (reasoning)
            3. Finally WRITE narration for a viewer (generation)

            This prevents the common failure mode where VLMs simply OCR
            the screen text instead of generating meaningful narration.
        """
        prompt = f"""You are a video tutorial analyst. Your task is to write the narration script for a tutorial/demo video.

## Your Tasks
1. **Observe the screen**: Look at what is happening on screen (code, UI, terminal, slides) to understand the CONTEXT — what step is being performed
2. **Write the narration script**: Based on what you see, write what a narrator would say to explain each step. This is the spoken script, NOT the on-screen text
3. **Provide timestamps**: When each narration segment starts and ends

## Important Rules
- Do NOT copy/transcribe the on-screen text literally
- Instead, DESCRIBE what is happening: what action is being performed, what the user is doing, what the result is
- Write as a narrator explaining steps to a viewer
- Example: If screen shows code `pip install torch`, write: "First, we install the PyTorch library using pip"
- Example: If screen shows a form being filled, write: "Now we fill in the product information in the form"
- Keep each segment focused on ONE action or step

## Output Format
Return a JSON array:
```json
[
  {{
    "start_time": "HH:MM:SS,mmm",
    "end_time": "HH:MM:SS,mmm",
    "text": "narration script describing this step",
    "screen_context": "brief description of what is visible on screen"
  }}
]
```
"""

        # Global context gives the VLM awareness of the full video structure
        if global_context:
            prompt += f"\n## Video Overview (Full Video Context)\n{global_context}\n"

        if chunk_info:
            prompt += f"\n## Video Chunk\n{chunk_info}\n"
            prompt += (
                "IMPORTANT: Generate narration ONLY for the time range of this chunk. "
                "Your timestamps must start from 00:00:00 relative to this chunk's start. "
                "Do NOT repeat or regenerate content from earlier parts of the video.\n"
            )

        if context_summary:
            prompt += f"\n## Previous Context\n{context_summary}\n"

        prompt += "\nAnalyze every frame. Write narration for each step shown."
        return prompt

    def step2_translate(
        self,
        extracted_text: str,
        context_summary: Optional[str] = None,
        previous_translations: Optional[str] = None,
    ) -> str:
        """
        Step 2: Translate extracted text to target language.
        """
        prompt = f"""You are a professional {self.source_lang} to {self.target_lang} translator specializing in technical video content.

## Task
Translate the following extracted text segments into natural {self.target_lang} narration suitable for voiceover dubbing.

## Translation Rules
1. **Code/commands**: Keep in English, do NOT translate
   - "pip install torch" → "pip install torch"
2. **Technical terms**: Keep in English with Vietnamese explanation when first introduced
   - "decorator" → "decorator"
   - "API endpoint" → "API endpoint"
3. **Numbers in speech**: Translate naturally
   - "1000 users" → "một nghìn người dùng"
4. **UI/button text**: Translate the explanation, keep label in English
   - "Click the Submit button" → "Nhấn nút Submit"
5. **Natural narration**: Use conversational Vietnamese, as if explaining to a student

## Extracted Content
```json
{extracted_text}
```
"""

        if context_summary:
            prompt += f"\n## Context from Previous Chunks\n{context_summary}\n"

        if previous_translations:
            prompt += f"\n## Previous Translations (maintain consistency)\n```\n{previous_translations}\n```\n"

        prompt += f"""
## Output Format
Return a JSON array with the same structure, adding "translated_text":
```json
[
  {{
    "start_time": "HH:MM:SS,mmm",
    "end_time": "HH:MM:SS,mmm",
    "original_text": "original",
    "translated_text": "bản dịch tiếng Việt",
    "type": "speech"
  }}
]
```
"""
        return prompt

    def step3_format_srt(
        self,
        translated_segments: str,
        video_duration: float,
        chunk_start: float = 0.0,
    ) -> str:
        """
        Step 3: Format into clean SRT entries with proper timing.
        """
        prompt = f"""You are a subtitle timing specialist. Convert these translated segments into properly formatted SRT subtitle entries.

## Rules
1. No overlapping subtitles (minimum 0.1s gap between entries)
2. Duration: minimum 1s, maximum 7s per entry
3. Reading speed: maximum 25 characters per second in {self.target_lang}
4. Maximum 2 lines per subtitle
5. Maximum 42 characters per line
6. If text is too long, split into multiple subtitle entries
7. Code snippets: show briefly (2-3s), they're also visible on screen

## Translated Segments
```json
{translated_segments}
```

## Timing Info
- Chunk starts at: {chunk_start:.3f}s
- Video duration: {video_duration:.3f}s
- Format: HH:MM:SS,mmm

## Output
Return ONLY a JSON array:
```json
[
  {{
    "index": 1,
    "start_time": "00:00:01,000",
    "end_time": "00:00:03,500",
    "original_text": "Hello and welcome",
    "translated_text": "Xin chào và chào mừng"
  }}
]
```
"""
        return prompt

    # ─────────────────────────────────────────────────────────────────
    # Single Prompt Mode (recommended for Gemini / large context)
    # ─────────────────────────────────────────────────────────────────

    def build_single_prompt(
        self,
        chunk_info: Optional[str] = None,
        context_summary: Optional[str] = None,
        previous_translations: Optional[str] = None,
        global_context: Optional[str] = None,
    ) -> str:
        """
        All-in-one prompt: Extract + Translate + SRT in a single VLM call.

        Recommended for:
        - Gemini (1M token context, fast)
        - Any model with large context window

        AI Engineer Prompting Design:
            This prompt combines all 3 steps into a single inference call.
            The global_context parameter injects the Pass 0 video overview,
            which dramatically improves:
            - Terminology consistency (VLM knows all key terms upfront)
            - Section awareness (VLM knows where this chunk fits in the video)
            - Narration flow (avoids redundant introductions)
        """
        prompt = f"""You are an expert video tutorial analyst and {self.source_lang} to {self.target_lang} subtitle translator.

## Task
Analyze the provided video frames and create {self.target_lang} narration subtitles:
1. **Observe** the screen to understand CONTEXT (what action is being performed)
2. **Write narration**: Describe what is happening as a step-by-step tutorial script — do NOT copy on-screen text literally
3. **Translate** the narration to natural {self.target_lang} for voiceover dubbing
4. **Format** as SRT subtitles with accurate timestamps

## Narration Rules
- Write what a narrator would SAY to explain each step
- Use the screen content as CONTEXT, not as text to transcribe
- Example: Screen shows `git clone ...` → Narration: "First, we clone the repository"
- Example: Screen shows a dashboard → Narration: "Here we can see the project dashboard"
- Keep each subtitle focused on ONE action or step

## Translation Rules
- Technical terms: Keep in English (API, database, deploy, etc.)
- Use conversational {self.target_lang}, like explaining to a student
- Numbers in speech: Translate naturally

## SRT Constraints
- No overlapping subtitles (min 0.1s gap)
- Duration: 1s — 7s per subtitle
- Max 2 lines, max 42 characters per line
- Reading speed: max 25 characters per second
- If too long, split into multiple entries
"""

        # Global context: video-level overview from Pass 0
        # This is the key architectural improvement — each chunk now "knows"
        # the full video structure before processing its local content
        if global_context:
            prompt += f"\n## Video Overview (Full Video Context)\n{global_context}\n"

        if chunk_info:
            prompt += f"\n## Video Chunk\n{chunk_info}\n"
            prompt += (
                "IMPORTANT: Generate subtitles ONLY for the time range of this chunk. "
                "Your timestamps must start from 00:00:00 relative to this chunk's start. "
                "Do NOT repeat or regenerate content from earlier parts of the video.\n"
            )

        if context_summary:
            prompt += f"\n## Context from Previous Chunks\n{context_summary}\n"

        if previous_translations:
            prompt += f"\n## Previous Translations (for consistency)\n```\n{previous_translations}\n```\n"

        prompt += f"""
## Required JSON Output
```json
[
  {{
    "index": 1,
    "start_time": "00:00:01,000",
    "end_time": "00:00:03,500",
    "original_text": "Hello and welcome to this tutorial",
    "translated_text": "Xin chào và chào mừng đến với hướng dẫn này"
  }}
]
```

Return ONLY the JSON array. No markdown, no explanation.
"""
        return prompt

    # ─────────────────────────────────────────────────────────────────
    # Legacy compatibility
    # ─────────────────────────────────────────────────────────────────

    def step1_extract_text(self, **kwargs) -> str:
        """Legacy alias for step1_extract."""
        return self.step1_extract(**kwargs)

    def step3_assign_timestamps(self, **kwargs) -> str:
        """Legacy alias for step3_format_srt."""
        return self.step3_format_srt(**kwargs)
