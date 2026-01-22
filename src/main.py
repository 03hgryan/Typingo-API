from dotenv import load_dotenv
load_dotenv()  # Load .env file before other imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from src.routers import websocket
from src.routers import stt

app = FastAPI(
    title="ELE - WebSocket Server",
    description="FastAPI WebSocket Server",
    version="1.0.0",
)

# CORS configuration
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "ELE WebSocket Server",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "websocket": "/ws/stream",
            "elevenlabs": "/stt/elevenlabs",
        },
    }


# Include routers
app.include_router(
    websocket.router,
    prefix="/ws",
    tags=["websocket"],
)

# Mount STT routers
app.include_router(
    stt.router, 
    prefix="/stt",
    tags=["stt"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ELE WebSocket Server",
    }