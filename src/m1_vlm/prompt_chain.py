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
        frame_timestamps: Optional[List[float]] = None,
        chunk_start: float = 0.0,
        chunk_end: float = 0.0,
        chunk_index: int = 0,
        total_chunks: int = 1,
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

        Key Anti-Hallucination Design:
            - Frame timestamps are EXPLICITLY listed so VLM must map output to real frames
            - Per-frame description is REQUIRED before generating narration
            - Strict instruction to ONLY describe visible content, never invent
        """
        prompt = f"""You are an expert video analyst and {self.source_lang} to {self.target_lang} subtitle translator.

## Task
You are given {len(frame_timestamps) if frame_timestamps else 'several'} frames from a video segment (Chunk {chunk_index + 1}/{total_chunks}, from {chunk_start:.1f}s to {chunk_end:.1f}s).

Create {self.target_lang} narration subtitles that provide a HIGH-LEVEL overview:
1. **Look at the frames** — understand the general topic/section being shown
2. **Write overview narration** — describe WHAT this section is about at a high level, NOT what buttons/menus are being clicked
3. **Translate** to natural {self.target_lang} for voiceover
4. **Assign timestamps** — use the frame timestamps below as anchor points
"""

        # ── Frame timestamp anchoring (critical for anti-hallucination) ──
        if frame_timestamps:
            prompt += f"""
## Frame Timestamps (ABSOLUTE — relative to video start)
You are given exactly {len(frame_timestamps)} frames at these timestamps:
"""
            for i, ts in enumerate(frame_timestamps):
                prompt += f"  - Frame {i + 1}: {ts:.1f}s\n"

            prompt += f"""
CRITICAL RULES for timestamps:
- Your subtitle timestamps MUST fall within [{chunk_start:.1f}s, {chunk_end:.1f}s]
- Use the frame timestamps above as ANCHOR POINTS for your subtitles
- Each subtitle's start_time should be near a frame timestamp
- Generate timestamps in ABSOLUTE time (relative to video start, NOT relative to chunk start)
- You MUST generate subtitles that span the FULL duration of this chunk, not just the first few seconds
- If a frame shows the same screen as the previous frame, it may mean the narrator is still explaining — extend the previous subtitle or describe what changed
"""

        # ── Narration rules ──
        prompt += f"""
## Language Strategy
- Use English for Phase A (observation) — describe UI elements and actions in English
- Use Vietnamese for Phase B (script writing) — write the voiceover in natural Vietnamese
- In the output JSON: original_text = English observation, translated_text = Vietnamese voiceover

## Narration Rules — HIGH-LEVEL OVERVIEW ONLY
- Write a GENERAL OVERVIEW of what this section of the video is about
- Do NOT describe specific UI actions in detail (e.g., "click this button", "navigate to this menu")
- Instead, describe the PURPOSE or TOPIC of what's being shown
- Examples of what TO DO:
  ✓ "Phần này giới thiệu về tính năng quản lý tài liệu" (This section introduces the document management feature)
  ✓ "Tiếp theo, chúng ta tìm hiểu cách hệ thống xử lý câu hỏi" (Next, we learn how the system handles questions)
  ✓ "Ở đây, hệ thống đang thực hiện việc chuyển đổi dữ liệu" (Here, the system is performing data conversion)
- Examples of what NOT to do:
  ✗ "Người dùng nhấn vào nút Submit ở góc trên bên phải" (User clicks Submit button in top-right corner)
  ✗ "Màn hình hiển thị một form với 3 trường nhập liệu" (Screen shows a form with 3 input fields)
- Keep narration concise — 1-2 sentences per subtitle
- Write as a CONTINUOUS narrative with natural flow between sections
- Use connecting words: "Tiếp theo" (Next), "Sau đó" (Then), "Bây giờ" (Now), "Ở đây" (Here)

## Anti-Hallucination Rules
- ONLY describe topics you can ACTUALLY SEE in the frames
- Do NOT invent welcome messages, introductions, or conclusions that aren't visible
- If frames show similar content, create fewer subtitles rather than inventing content

## Translation Rules
- Technical terms: Keep in {self.source_lang} (API, database, deploy, etc.)
- Use conversational {self.target_lang}, professional spoken style
- Make it sound like a narrator giving an overview, not reading a manual

## SRT Constraints
- No overlapping subtitles (min 0.1s gap)
- Duration: 2s — 7s per subtitle
- Max 2 lines, max 42 characters per line
- Reading speed: max 25 characters per second
"""

        # ── Global context ──
        if global_context:
            prompt += f"\n## Video Overview (Full Video Context)\n{global_context}\n"

        if chunk_info:
            prompt += f"\n## Video Chunk Info\n{chunk_info}\n"

        # ── Anti-repeat context ──
        if context_summary:
            prompt += f"\n## Previously Generated Content (DO NOT REPEAT)\n{context_summary}\n"
            prompt += "IMPORTANT: Do NOT regenerate any of the above entries. Continue the narration from where it left off.\n"

        if previous_translations:
            prompt += f"\n## Previous Translations (for style consistency, DO NOT REPEAT)\n```\n{previous_translations}\n```\n"

        prompt += f"""
## Required JSON Output
Return ONLY a JSON array with subtitles for THIS chunk. Timestamps must be ABSOLUTE (relative to video start).
```json
[
  {{
    "index": 1,
    "start_time": "{self._format_timestamp(chunk_start + 1.0)}",
    "end_time": "{self._format_timestamp(chunk_start + 4.0)}",
    "original_text": "Description of what is happening on screen",
    "translated_text": "Mô tả bằng tiếng Việt về những gì đang diễn ra"
  }}
]
```

Return ONLY the JSON array. No markdown code fences, no explanation.
"""
        return prompt

    def build_translation_only_prompt(
        self,
        en_text: str,
        chunk_info: Optional[str] = None,
        global_context: Optional[str] = None,
        previous_translations: Optional[str] = None,
    ) -> str:
        """Translate ONE English caption to Vietnamese voiceover.

        Assumes a single frame image is attached as visual grounding so the
        translator can disambiguate UI/code references in `en_text`.

        Output: JSON object {"translated_text": "..."}. No timing field.
        """
        prompt = f"""You are a professional {self.source_lang}-to-{self.target_lang} translator for technical tutorial voiceover.

## Source caption
"{en_text}"

## Visual context
One frame from the video at the moment this caption appeared is attached as an image. Use it to disambiguate any UI labels, code identifiers, or on-screen elements the caption refers to. Do NOT describe the frame; only use it to inform the translation.

## Translation rules
- Output natural, conversational {self.target_lang} suitable for spoken voiceover.
- Keep technical terms, code, commands, and UI labels in {self.source_lang} (e.g., "API", "pip install torch", "Submit").
- Translate numbers and conjunctions naturally.
- Do not add greetings, introductions, or conclusions that aren't in the source.
- One sentence in, one sentence out — do not split or merge.

## Continuity (only if "Previous translations" appears below)
- When a previous translation is provided, this caption is the NEXT line of the same narration. Make the two flow naturally.
- Begin the translation with a soft Vietnamese connector when it sounds natural — e.g., "Tiếp theo,", "Sau đó,", "Bây giờ,", "Ở đây,", "Tiếp đến,". Do not force a connector if the caption already starts smoothly.
- Do NOT repeat content already covered by previous translations.
- Avoid awkward English carryovers — phrase the link as a Vietnamese narrator would speak between two sentences.
"""
        if global_context:
            prompt += f"\n## Video overview\n{global_context}\n"
        if chunk_info:
            prompt += f"\n## Chunk info\n{chunk_info}\n"
        if previous_translations:
            prompt += (
                "\n## Previous translations (the line(s) spoken just before this one — keep flow natural)\n"
                f"{previous_translations}\n"
            )

        prompt += """
## Output
Return ONLY a JSON object with this exact shape, no markdown fences, no explanation.
Replace <VI_TRANSLATION_HERE> with the actual Vietnamese translation of the source caption above — do NOT copy the placeholder token verbatim.
```json
{"translated_text": "<VI_TRANSLATION_HERE>"}
```
"""
        return prompt

    def build_narration_classifier_prompt(self, captions: list[str]) -> str:
        """Batched classifier: label each caption as 'narration' or 'screen'.

        Used to drop UI/header/footer text that GLM-OCR picks up alongside
        real narration captions. Returns a prompt that asks for a JSON object:
            {"labels": ["narration", "screen", ...]}
        with exactly len(captions) entries in input order.
        """
        if not captions:
            raise ValueError("captions must be non-empty")

        prompt = f"""You are a binary classifier for tutorial-video captions.

Each input below is one line of text extracted from a video frame. Many of these tutorial videos are voiced demos where the narrator READS THE INSTRUCTIONS ALOUD while clicking through the UI — so short imperative phrases like "choose the language" or "upload the file" are almost always NARRATION, not screen-only text.

Decide for each line:
- "narration": a sentence or instruction a narrator would speak. Includes spoken UI instructions ("choose the language", "click submit", "upload your file", "generate summary"), descriptive sentences, imperative steps. Anything with a verb and natural sentence structure counts.
- "screen": page chrome a narrator would NEVER speak — copyright footers, brand watermarks alone, version strings, long ALL-CAPS document titles, error toasts, dates, raw numeric IDs.

**Lean toward "narration" when uncertain.** It is much worse to drop spoken content than to keep one extra line.

## Examples

| Text | Label | Why |
|---|---|---|
| "click the submit button to send your data" | narration | sentence with action |
| "choose the language of the meeting" | narration | instruction; narrator says this |
| "upload the audio or video of a meeting" | narration | instruction; narrator says this |
| "generate summary" | narration | short, but a spoken step |
| "the generation time depends on the length of the meeting" | narration | descriptive sentence |
| "the summary box shows the summary of the meeting" | narration | descriptive sentence |
| "© 2026 vnext - powered by stt & llm" | screen | static copyright footer |
| "© 2026 vnext · powered by stt & llm" | screen | static copyright footer |
| "TOYOBEAUTY - MODULE 5 SALES MEETING MINUTES AUTO CREATION" | screen | long ALL-CAPS title chrome |
| "USER REGISTRATION" | screen | page header only |
| "v2.4.1" | screen | version string |
| "Submit" | screen | bare 1-word button label, no verb context |
| "marketing marketing marketing" | screen | repeated brand placeholder, no narration shape |

## Captions to classify

There are {len(captions)} captions. Classify each in the order given:

"""
        for i, c in enumerate(captions, start=1):
            # Truncate extremely long lines so the prompt fits a small model.
            safe = c.replace("\n", " ").strip()
            if len(safe) > 200:
                safe = safe[:200] + "..."
            prompt += f'{i}. "{safe}"\n'

        prompt += f"""

## Output

Return ONLY a JSON object with exactly this shape, no markdown fences, no explanation:
```json
{{"labels": ["narration", "screen", ...]}}
```

The `labels` array MUST have exactly {len(captions)} entries, in the same order as the captions above.
Each entry MUST be the string "narration" or the string "screen". No other values are allowed.
"""
        return prompt

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as SRT timestamp HH:MM:SS,mmm."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ─────────────────────────────────────────────────────────────────
    # Legacy compatibility
    # ─────────────────────────────────────────────────────────────────

    def step1_extract_text(self, **kwargs) -> str:
        """Legacy alias for step1_extract."""
        return self.step1_extract(**kwargs)

    def step3_assign_timestamps(self, **kwargs) -> str:
        """Legacy alias for step3_format_srt."""
        return self.step3_format_srt(**kwargs)
