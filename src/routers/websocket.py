"""
Basic WebSocket server boilerplate.
"""

import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from auth.config import AUTH_ENABLED
from auth.dependencies import require_ws_auth

router = APIRouter()


@router.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    user = await require_ws_auth(ws)
    if AUTH_ENABLED and user is None:
        return
    print("✅ Client connected")

    try:
        while True:
            message = await ws.receive()

            # Text message (JSON)
            if "text" in message:
                data = json.loads(message["text"])
                print(f"📨 Received: {data}")

                # Echo back
                await ws.send_json({"type": "received", "data": data})

            # Binary message
            elif "bytes" in message:
                print(f"📦 Binary: {len(message['bytes'])} bytes")

    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🔌 Connection closed")