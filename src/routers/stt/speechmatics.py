import os
import json
import base64
import time
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from speechmatics.rt import (
    AsyncClient,
    ServerMessageType,
    TranscriptionConfig,
    TranscriptResult,
    OperatingPoint,
    AudioFormat,
    AudioEncoding,
    AuthenticationError,
)
from utils.translationTest import Translator
from utils.combiner import Combiner

router = APIRouter()

SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY")


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()
    print("\n" + "=" * 80)
    print("🎤 SPEECHMATICS SESSION START")
    print("=" * 80)

    if not SPEECHMATICS_API_KEY:
        await ws.send_json({"type": "error", "message": "SPEECHMATICS_API_KEY not configured"})
        await ws.close()
        return

    speechmatics_client = None

    # --- Translation pipeline ---
    async def on_combined(full_text, seq):
        await ws.send_json({
            "type": "combined",
            "seq": seq,
            "full": full_text,
        })

    combiner = Combiner(on_combined=on_combined)

    async def on_translation(english_source, translated_text, seq):
        await ws.send_json({"type": "translation", "seq": seq, "text": translated_text})
        combiner.feed_translation(english_source, translated_text)

    translator = Translator(on_translation=on_translation, partial_interval=6)

    class SessionState:
        def __init__(self):
            self.first_audio_time = None
            self.first_partial_time = None
            self.first_final_time = None
            self.final_text = ""
            self.current_partial = ""
            self.prev_full_text = ""
            self.partial_count = 0

        def get_full_text(self) -> str:
            if not self.final_text:
                return self.current_partial.strip()
            if not self.current_partial:
                return self.final_text.strip()

            final = self.final_text.rstrip()
            partial = self.current_partial.strip()

            best_overlap = 0
            max_check = min(len(final), len(partial))
            for length in range(1, max_check + 1):
                if final.endswith(partial[:length]):
                    best_overlap = length

            if best_overlap > 0:
                return (final + partial[best_overlap:]).strip()
            else:
                return (final + " " + partial).strip()

    state = SessionState()

    audio_format = AudioFormat(
        encoding=AudioEncoding.PCM_S16LE,
        chunk_size=4096,
        sample_rate=16000,
    )

    transcription_config = TranscriptionConfig(
        language="en",
        enable_partials=True,
        operating_point=OperatingPoint.ENHANCED,
        diarization="speaker",
        speaker_diarization_config={
            "max_speakers": 10
        }
    )

    try:
        print("🔌 Connecting to Speechmatics...")
        speechmatics_client = AsyncClient(api_key=SPEECHMATICS_API_KEY)
        await speechmatics_client.__aenter__()

        @speechmatics_client.on(ServerMessageType.ADD_TRANSCRIPT)
        def handle_final_transcript(message):
            current_time = time.time()

            if state.first_final_time is None and state.first_audio_time:
                state.first_final_time = current_time
                latency = (state.first_final_time - state.first_audio_time) * 1000
                print(f"\n⚡ FIRST FINAL LATENCY: {latency:.0f}ms")

            result = TranscriptResult.from_message(message)
            transcript = result.metadata.transcript

            if transcript:
                if state.final_text:
                    state.final_text += transcript
                else:
                    state.final_text = transcript

                full_text = state.get_full_text()
                if full_text and full_text != state.prev_full_text:
                    state.prev_full_text = full_text
                    print(f"📗 ({len(full_text.split())}w): ...{full_text[-90:]}")

        @speechmatics_client.on(ServerMessageType.ADD_PARTIAL_TRANSCRIPT)
        def handle_partial_transcript(message):
            current_time = time.time()

            if state.first_partial_time is None and state.first_audio_time:
                state.first_partial_time = current_time
                latency = (state.first_partial_time - state.first_audio_time) * 1000
                print(f"\n⚡ FIRST PARTIAL LATENCY: {latency:.0f}ms\n")

            result = TranscriptResult.from_message(message)
            transcript = result.metadata.transcript

            state.current_partial = transcript or ""

            full_text = state.get_full_text()
            if not full_text or full_text == state.prev_full_text:
                return
            state.prev_full_text = full_text

            print(f"📝 ({len(full_text.split())}w): ...{full_text[-90:]}")

            loop = asyncio.get_event_loop()
            loop.create_task(ws.send_json({"type": "partial", "text": full_text}))

            translator.feed_partial(full_text)

        print("🚀 Starting session...")
        await speechmatics_client.start_session(
            transcription_config=transcription_config,
            audio_format=audio_format,
        )
        print("✅ Session started successfully")
        print("─" * 80)

        await ws.send_json({"type": "session_started", "data": {"status": "connected"}})

        await forward_audio(ws, speechmatics_client, state)

    except AuthenticationError as e:
        print(f"\n❌ Authentication Error: {e}")
        await ws.send_json({"type": "error", "message": f"Authentication error: {e}"})
    except WebSocketDisconnect:
        print("\n👋 Client disconnected")
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
    finally:
        if speechmatics_client:
            try:
                await speechmatics_client.__aexit__(None, None, None)
            except:
                pass

        if state.first_audio_time:
            print("\n" + "─" * 80)
            print("📊 SESSION SUMMARY:")
            if state.first_partial_time:
                print(f"   First Partial: {(state.first_partial_time - state.first_audio_time) * 1000:.0f}ms")
            if state.first_final_time:
                print(f"   First Final: {(state.first_final_time - state.first_audio_time) * 1000:.0f}ms")
            print(f"   Final text length: {len(state.final_text.split())} words")
            print("─" * 80)

        print("=" * 80 + "\n")


async def forward_audio(client_ws: WebSocket, speechmatics_client, state):
    chunk_count = 0

    try:
        while True:
            message = await client_ws.receive()

            if "text" not in message:
                continue

            data = json.loads(message["text"])
            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                chunk_count += 1

                if state.first_audio_time is None:
                    state.first_audio_time = time.time()
                    print(f"🎵 First audio chunk received\n")

                audio_base64 = data.get("audio_base_64", "")
                audio_bytes = base64.b64decode(audio_base64)
                await speechmatics_client.send_audio(audio_bytes)

            elif msg_type == "end_stream":
                print(f"\n🛑 Stream ended ({chunk_count} total chunks)")
                break

    except WebSocketDisconnect:
        print(f"\n👋 Client disconnected after {chunk_count} chunks")
    except Exception as e:
        print(f"\n❌ ERROR in forward_audio: {type(e).__name__}: {e}")