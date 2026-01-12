# asr/base.py
"""
Abstract base class for ASR engines.

All ASR implementations must follow this interface to ensure
compatibility with the rest of the system.
"""

from abc import ABC, abstractmethod
from typing import TypedDict


class WordInfo(TypedDict):
    """Word with timing and confidence."""
    word: str
    start_ms: int
    end_ms: int
    probability: float


class Hypothesis(TypedDict):
    """ASR hypothesis output."""
    type: str  # "hypothesis"
    segment_id: str
    revision: int
    text: str
    words: list[WordInfo]
    confidence: float
    start_time_ms: int
    end_time_ms: int


class ASREngine(ABC):
    """
    Abstract base class for ASR engines.
    
    All implementations must provide:
    - feed(audio_bytes) -> Hypothesis | None
    - reset() -> None
    
    The feed() method should:
    - Accept PCM16 audio bytes (16kHz, mono)
    - Return a Hypothesis dict with word-level timestamps
    - Return None if not enough audio to transcribe
    """
    
    @abstractmethod
    def feed(self, audio: bytes) -> Hypothesis | None:
        """
        Feed audio chunk and return hypothesis if available.
        
        Args:
            audio: PCM16 audio bytes (16kHz, mono)
            
        Returns:
            Hypothesis dict with word timestamps, or None
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the engine state for a new segment."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'whisper', 'google')."""
        pass
