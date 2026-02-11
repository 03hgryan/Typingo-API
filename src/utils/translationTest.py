import os
import time
import asyncio
from openai import AsyncOpenAI
from utils.tone import ToneDetector

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

TARGET_LANG = "Korean"
MAX_WORDS = 25

SYSTEM_PROMPT = """You are a real-time subtitle translator. Translate the given English text to {lang}.

{tone_instruction}

Rules:
- Translate naturally, not word-by-word
- Match the speaker's energy and intent
- Output ONLY the translation, nothing else
- No quotes, no explanations, no labels"""


class Translator:
    def __init__(self, on_translation=None, partial_interval=3):
        self.on_translation = on_translation
        self.partial_interval = partial_interval
        self.seq = 0
        self.latest_seq = -1
        self.partial_count = 0
        self.tone_detector = ToneDetector()
        self._last_tone = self.tone_detector.current_tone
        self._last_fire_time = None
        self._last_fire_words = 0

    def _get_prompt(self) -> str:
        tone = self.tone_detector.current_tone
        if tone != self._last_tone:
            self._last_tone = tone
            print(f"🎭 Tone updated → {tone}")

        return SYSTEM_PROMPT.format(
            lang=TARGET_LANG,
            tone_instruction=self.tone_detector.get_tone_instruction(),
        )

    def feed_partial(self, partial_text: str):
        self.tone_detector.feed_text(partial_text)
        self.partial_count += 1

        if self.partial_count == 1:
            pass
        elif self.partial_count % self.partial_interval != 0:
            return None

        words = partial_text.split()
        total_words = len(words)
        text = " ".join(words[-MAX_WORDS:]) if total_words > MAX_WORDS else partial_text

        if not text.strip():
            return None

        now = time.monotonic()
        if self._last_fire_time is not None:
            gap_ms = (now - self._last_fire_time) * 1000
            word_delta = total_words - self._last_fire_words
            print(f"⏱️  fire gap:{gap_ms:.0f}ms | +{word_delta}w | total:{total_words}w | partial#{self.partial_count}")
        else:
            print(f"⏱️  first fire | total:{total_words}w | partial#{self.partial_count}")
        self._last_fire_time = now
        self._last_fire_words = total_words

        self.seq += 1
        return asyncio.create_task(self._run(text.strip(), self.seq))

    async def _run(self, text: str, seq: int):
        try:
            t_start = time.monotonic()
            word_count = len(text.split())

            stream = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self._get_prompt()},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=200,
                stream=True,
            )

            translated = ""
            ttft = None
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if ttft is None:
                        ttft = (time.monotonic() - t_start) * 1000
                    translated += delta

            total_ms = (time.monotonic() - t_start) * 1000

            if not translated.strip():
                return

            if seq < self.latest_seq:
                print(f"🗑️  [{seq}] STALE | {word_count}w | ttft:{ttft:.0f}ms total:{total_ms:.0f}ms")
                return

            self.latest_seq = seq

            tone = self.tone_detector.current_tone
            print(f"🌐 [{seq}] {word_count}w [{tone}] | ttft:{ttft:.0f}ms total:{total_ms:.0f}ms")
            print(f"    EN: {text[:80]}")
            print(f"    KR: {translated.strip()[:80]}")

            if self.on_translation:
                await self.on_translation(text, translated.strip(), seq)

        except Exception as e:
            import traceback
            print(f"Translation error: {type(e).__name__}: {e}")
            traceback.print_exc()