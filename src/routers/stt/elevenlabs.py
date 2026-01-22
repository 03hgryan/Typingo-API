"""
ElevenLabs Scribe v2 Speech-to-Text WebSocket handler.

Endpoint: /stt/elevenlabs
Docs: https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime
"""

import os
import json
import asyncio
import websockets
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime?model_id=scribe_v2_realtime"


@router.websocket("")
async def stream(ws: WebSocket):
    await ws.accept()
    print("[ElevenLabs] Client connected")

    if not ELEVENLABS_API_KEY:
        await ws.send_json({"type": "error", "message": "ELEVENLABS_API_KEY not configured"})
        await ws.close()
        return

    elevenlabs_ws = None

    try:
        # Connect to ElevenLabs
        print("[ElevenLabs] Connecting...")
        elevenlabs_ws = await websockets.connect(
            ELEVENLABS_URL,
            additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
        )

        # Wait for session_started
        session_msg = await elevenlabs_ws.recv()
        session_data = json.loads(session_msg)
        print(f"[ElevenLabs] Session started: {session_data.get('session_id')}")

        await ws.send_json({"type": "session_started", "data": session_data})

        # Run send and receive concurrently
        await asyncio.gather(
            forward_audio(ws, elevenlabs_ws),
            forward_transcripts(ws, elevenlabs_ws),
        )

    except websockets.InvalidStatusCode as e:
        print(f"[ElevenLabs] Connection failed: {e}")
        await ws.send_json({"type": "error", "message": str(e)})
    except WebSocketDisconnect:
        print("[ElevenLabs] Client disconnected")
    except Exception as e:
        print(f"[ElevenLabs] Error: {type(e).__name__}: {e}")
    finally:
        if elevenlabs_ws:
            await elevenlabs_ws.close()
        print("[ElevenLabs] Connection closed")


async def forward_audio(client_ws: WebSocket, elevenlabs_ws):
    """Forward audio from client to ElevenLabs"""
    try:
        while True:
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
                print("[ElevenLabs] Sending final commit")
                await elevenlabs_ws.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": "",
                    "commit": True,
                    "sample_rate": 16000,
                }))
                return

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[ElevenLabs] Forward error: {e}")


async def forward_transcripts(client_ws: WebSocket, elevenlabs_ws):
    """Forward transcripts from ElevenLabs to client"""
    try:
        async for message in elevenlabs_ws:
            data = json.loads(message)
            msg_type = data.get("message_type", "unknown")

            print(f"📨 ElevenLabs [{msg_type}]: {json.dumps(data)}")

            await client_ws.send_json({"type": msg_type, "data": data})

    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[ElevenLabs] Transcript error: {e}")