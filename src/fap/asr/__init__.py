"""
ASR module for streaming speech recognition
"""

"""
ASR Engine abstraction layer.

Supports multiple backends:
- faster-whisper (local, offline)
- Google Cloud Speech-to-Text (cloud, streaming)

Usage:
    # Using adapter (recommended)
    from fap.asr import create_asr_adapter
    
    adapter = create_asr_adapter("whisper", model_size="medium")
    # or
    adapter = create_asr_adapter("google", language_code="en-US")
    
    hypothesis = adapter.feed(audio_bytes)
    
    # Using engine directly
    from fap.asr import create_asr_engine
    
    engine = create_asr_engine("whisper", model_size="medium")
"""

from .engine import StreamingASR
from .buffer import RollingAudioBuffer
from .utils import is_silence
from .base import ASREngine
from .whisper_engine import WhisperASR
from .google_engine import GoogleCloudASR
from .factory import create_asr_engine, load_shared_model
from .adapter import ASRAdapter, WhisperAdapter, GoogleAdapter, create_asr_adapter


__all__ = ["StreamingASR", "RollingAudioBuffer", "is_silence", "ASREngine", "WhisperASR", "GoogleCloudASR", "create_asr_engine", "load_shared_model", "ASRAdapter", "WhisperAdapter", "GoogleAdapter", "create_asr_adapter" ]
