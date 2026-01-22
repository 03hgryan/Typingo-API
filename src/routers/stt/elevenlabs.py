"""
ElevenLabs Scribe v2 Speech-to-Text WebSocket handler with OpenAI Translation.

Endpoint: /stt/elevenlabs

Client Protocol:
- Send: {"type": "audio_chunk", "audio_base_64": "...", ...}
- Send: {"type": "end_stream"}
- Receive: {"type": "session_started", "data": {...}}
- Receive: {"type": "partial_transcript", "data": {...}}
- Receive: {"type": "committed_transcript", "data": {...}}
- Receive: {"type": "translation_delta", "data": {"delta": "...", "text": "..."}}
- Receive: {"type": "translation_complete", "data": {"text": "..."}}
- Receive: {"type": "translation_error", "data": {"message": "...", "text": "..."}}
- Receive: {"type": "error", "message": "..."}
"""

import os
import json
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import openai

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Connection timeout in seconds
CONNECTION_TIMEOUT = 10.0


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()
    print("✅ Client connected")

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    if not OPENAI_API_KEY:
        await ws.send_json({"type": "error", "message": "OPENAI_API_KEY not configured"})
        await ws.close()
        return

    elevenlabs_ws = None

    # Queue for translation jobs (limit size to prevent memory issues)
    translation_queue = asyncio.Queue(maxsize=20)

    # Shutdown event
    shutdown_event = asyncio.Event()

    # Start translation worker task
    translation_worker = asyncio.create_task(
        translation_worker_task(ws, translation_queue, shutdown_event)
    )

    try:
        print("🔌 Connecting to ElevenLabs...")
        elevenlabs_ws = await asyncio.wait_for(
            websockets.connect(
                ELEVENLABS_URL,
                additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
            ),
            timeout=CONNECTION_TIMEOUT
        )

        session_msg = await elevenlabs_ws.recv()
        session_data = json.loads(session_msg)
        print(f"🎙️  Session started: {session_data.get('session_id')}")

        await ws.send_json({"type": "session_started", "data": session_data})

        await asyncio.gather(
            forward_audio(ws, elevenlabs_ws, shutdown_event),
            forward_transcripts(ws, elevenlabs_ws, translation_queue, shutdown_event),
        )

    except asyncio.TimeoutError:
        print("❌ Connection timeout")
        await ws.send_json({"type": "error", "message": "Connection to ElevenLabs timed out"})
    except websockets.InvalidStatusCode as e:
        print(f"❌ Connection failed: {e}")
        await ws.send_json({"type": "error", "message": f"ElevenLabs connection failed: {e}"})
    except WebSocketDisconnect:
        print("👋 Client disconnected")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
    finally:
        # Signal shutdown
        if not shutdown_event.is_set():
            shutdown_event.set()

        # Close ElevenLabs connection first
        if elevenlabs_ws:
            await elevenlabs_ws.close()

        print(f"⏳ Waiting for {translation_queue.qsize()} remaining translations...")
        
        # Wait for queue to drain
        await translation_queue.join()

        # Wait for worker to finish
        await translation_worker

        print("✅ Connection closed cleanly")


async def forward_audio(client_ws: WebSocket, elevenlabs_ws, shutdown_event: asyncio.Event):
    """Forward audio from client to ElevenLabs"""
    try:
        while not shutdown_event.is_set():
            message = await client_ws.receive()

            if "text" not in message:
                continue

            data = json.loads(message["text"])
            msg_type = data.get("type")

            if msg_type == "audio_chunk":
                await elevenlabs_ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": data.get("audio_base_64", ""),
                    "commit": False,
                    "sample_rate": 16000,
                }))

            elif msg_type == "end_stream":
                print("🔚 Sending final commit...")
                await elevenlabs_ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "commit": True,
                    "sample_rate": 16000,
                }))
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"❌ Forward audio error: {type(e).__name__}: {e}")
    finally:
        # If client disconnects, signal shutdown
        if not shutdown_event.is_set():
            shutdown_event.set()


async def forward_transcripts(client_ws: WebSocket, elevenlabs_ws, queue: asyncio.Queue, shutdown_event: asyncio.Event):
    """Forward transcripts from ElevenLabs to client and queue ALL transcripts for translation"""
    committed_received = False
    
    try:
        async for message in elevenlabs_ws:
            if shutdown_event.is_set() and committed_received:
                break

            data = json.loads(message)
            msg_type = data.get("message_type", "unknown")
            text = data.get("text", "").strip()

            # Clean log output
            if msg_type == "partial_transcript":
                print(f"📝 Partial: {text[:60]}{'...' if len(text) > 60 else ''}")
            elif msg_type == "committed_transcript":
                print(f"✔️  Committed: {text[:60]}{'...' if len(text) > 60 else ''}")

            await client_ws.send_json({"type": msg_type, "data": data})

            # Queue translation for BOTH partial and committed transcripts
            if msg_type in ["partial_transcript", "committed_transcript"]:
                if text:
                    try:
                        # Non-blocking put with timeout to prevent hanging
                        await asyncio.wait_for(queue.put(text), timeout=1.0)
                    except asyncio.TimeoutError:
                        print(f"⚠️  Queue full, dropping translation")
                        await client_ws.send_json({
                            "type": "translation_error",
                            "data": {"message": "Translation queue full", "text": text}
                        })
            
            if msg_type == "committed_transcript":
                committed_received = True
                print("🏁 Final transcript received")
                break

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"❌ Transcript error: {type(e).__name__}: {e}")
    finally:
        # Signal shutdown
        if not shutdown_event.is_set():
            shutdown_event.set()


async def translation_worker_task(ws: WebSocket, queue: asyncio.Queue, shutdown_event: asyncio.Event):
    """
    Worker that pulls from the queue and translates sequentially.
    Stops when shutdown_event is set AND queue is empty.
    """
    print("🔄 Translation worker started")

    while not shutdown_event.is_set() or not queue.empty():
        try:
            text = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        remaining = queue.qsize()
        print(f"🌐 Translating ({remaining} in queue): {text[:50]}{'...' if len(text) > 50 else ''}")

        try:
            await translate_and_send(ws, text)
        except WebSocketDisconnect:
            print(f"⚠️  Client disconnected, skipping remaining translations")
            # Drain queue without processing
            while not queue.empty():
                try:
                    queue.get_nowait()
                    queue.task_done()
                except:
                    break
            break
        except Exception as e:
            print(f"❌ Translation failed: {type(e).__name__}")
        finally:
            queue.task_done()

    print("✅ Translation worker stopped")


async def translate_and_send(ws: WebSocket, text: str):
    """Translate using OpenAI Responses API and send streaming translation"""
    try:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # Use the Responses API with streaming
        stream = await client.responses.create(
            model="gpt-5.2",
            input=[
                {"role": "system", "content": "You are a real-time translator. Translate English to Korean."},
                {"role": "user", "content": f"Translate to Korean:\n\n{text}"}
            ],
            stream=True
        )

        translated = ""
        first_delta = True

        async for event in stream:
            # Handle text delta events
            if event.type == "response.output_text.delta":
                if first_delta:
                    print(f"   🇰🇷 ", end="", flush=True)
                    first_delta = False
                
                delta = event.delta
                translated += delta
                print(delta, end="", flush=True)
                
                await ws.send_json({
                    "type": "translation_delta",
                    "data": {"delta": delta, "text": translated}
                })

            # Handle completion
            elif event.type in ["response.done", "response.completed"]:
                if translated:
                    print()  # New line after translation
                await ws.send_json({
                    "type": "translation_complete",
                    "data": {"text": translated}
                })
                break

    except Exception as e:
        print(f"\n❌ Translation API error: {type(e).__name__}")
        raise