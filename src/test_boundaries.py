import os
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv
load_dotenv()

oai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """You are a real-time subtitle translator. Translate the given English text to Korean.

Rules:
- Translate naturally, not word-by-word
- Match the speaker's energy and intent
- Output ONLY the translation, nothing else
- CRITICAL: The input contains numbered boundary markers like [1], [2], etc. You MUST place the same markers at the corresponding points in your Korean translation. Each marker must appear exactly once in your output, in the same order."""

async def test(english: str):
    response = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": english},
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content

async def main():
    tests = [
        'Hey, Vsauce. [1] Michael here. [2] When something becomes part of the past, [3] can it ever truly be experienced again? [4]',
        'Obviously my beard will grow back, [1] but it won\'t be the same beard and it won\'t be on the same person. [2] It will be on a slightly older, [3] different Michael. [4]',
        'But of course, [1] the bearded Michael of the past isn\'t completely gone. [2] No, [3] he still exists in our minds as a memory and in the form of records of the past, [4] like images and video. [5]',
        'Hi, [1] I\'m the slightly older, [2] different Michael you heard about. [3] I am 130 days older than that guy. [4] Wow. [5] 130 days. [6]',
    ]

    for i, test_text in enumerate(tests):
        print(f"\n{'='*60}")
        print(f"EN: {test_text}")
        
        # Run 3 times to check consistency
        for run in range(3):
            result = await test(test_text)
            print(f"KR{run+1}: {result}")

asyncio.run(main())