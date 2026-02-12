import os
import asyncio
from openai import AsyncOpenAI

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DETECT_PROMPT = """Analyze this transcript from a live stream/video and determine the speaker's tone and register.

TRANSCRIPT:
{text}

Choose exactly ONE of these speech register levels that would best match the speaker's tone:

1. casual (friends talking, gaming streams, very relaxed)
   Use when: slang, filler words, addressing chat directly, cursing, incomplete sentences

2. casual_polite (friendly but polite, most YouTube content)
   Use when: conversational but structured, educational but approachable

3. formal (news, lectures, business presentations)
   Use when: professional vocabulary, structured speech, formal setting

4. narrative (documentaries, storytelling, essays)
   Use when: descriptive, third person, explaining concepts with authority

Respond with ONLY the tone name (casual, casual_polite, formal, or narrative). Nothing else."""

TONE_INSTRUCTIONS_KOREAN = {
    "casual": (
        "Use casual Korean (해체/반말). Examples: ~해, ~했어, ~할게, ~인데, ~거든, ~잖아, ~임, ~ㅋㅋ. "
        "Sound natural like talking to friends or streaming. No formal endings."
    ),
    "casual_polite": (
        "Use casual polite Korean (해요체). Examples: ~해요, ~했어요, ~할 거예요, ~이에요. "
        "Friendly but polite tone."
    ),
    "formal": (
        "Use formal polite Korean (합니다체). Examples: ~합니다, ~했습니다, ~하겠습니다. "
        "Maintain professional, respectful tone throughout."
    ),
    "narrative": (
        "Use written/narrative Korean (하다체). Examples: ~한다, ~했다, ~할 것이다, ~이다. "
        "Maintain a descriptive, storytelling tone."
    ),
}

TONE_INSTRUCTIONS_JAPANESE = {
    "casual": "Use casual Japanese (タメ口). Examples: ~だ, ~だよ, ~じゃん, ~っけ. Sound natural and relaxed.",
    "casual_polite": "Use polite Japanese (です/ます体). Examples: ~です, ~ました, ~でしょう. Friendly but polite.",
    "formal": "Use formal Japanese (敬語). Examples: ~でございます, ~いたします. Maintain professional, respectful tone.",
    "narrative": "Use written/narrative Japanese (だ/である体). Examples: ~である, ~した, ~のだ. Descriptive, storytelling tone.",
}

TONE_INSTRUCTIONS_GENERIC = {
    "casual": "Use casual, relaxed language. Sound natural like talking to friends. Use informal expressions and contractions.",
    "casual_polite": "Use a friendly but polite tone. Conversational yet structured.",
    "formal": "Use formal, professional language. Maintain a respectful and structured tone throughout.",
    "narrative": "Use a written, narrative style. Descriptive and authoritative, like a documentary or essay.",
}

TONE_INSTRUCTIONS_BY_LANG = {
    "Korean": TONE_INSTRUCTIONS_KOREAN,
    "Japanese": TONE_INSTRUCTIONS_JAPANESE,
}

DEFAULT_TONE = "casual_polite"


class ToneDetector:
    def __init__(self, target_lang: str = "Korean"):
        self.target_lang = target_lang
        self.tone_instructions = TONE_INSTRUCTIONS_BY_LANG.get(target_lang, TONE_INSTRUCTIONS_GENERIC)
        self.word_buffer: list[str] = []
        self.current_tone = DEFAULT_TONE
        self.detected = False
        self._detecting = False
        self._detect_task: asyncio.Task | None = None

    def feed_text(self, text: str):
        """Feed transcript text. Triggers detection after ~50 words."""
        if self.detected:
            return

        words = text.split()
        self.word_buffer = words  # Keep latest full partial

        if len(self.word_buffer) >= 30 and not self._detecting:
            self._detecting = True
            self._detect_task = asyncio.create_task(self._detect())

    async def _detect(self):
        text = " ".join(self.word_buffer[-100:])
        try:
            response = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": DETECT_PROMPT.format(text=text)},
                ],
                temperature=0,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip().lower()

            if result in self.tone_instructions:
                old = self.current_tone
                self.current_tone = result
                self.detected = True
                print(f"🎭 Tone detected: {old} → {result} (from {len(self.word_buffer)}w)")
            else:
                print(f"🎭 Tone detection unclear: '{result}', keeping {self.current_tone}")
                self._detecting = False  # Retry later

        except Exception as e:
            print(f"🎭 Tone detection error: {e}")
            self._detecting = False

    def get_tone_instruction(self) -> str:
        return self.tone_instructions.get(self.current_tone, self.tone_instructions[DEFAULT_TONE])