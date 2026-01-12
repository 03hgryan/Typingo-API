# asr/adapter.py
"""
ASR Adapter Pattern

Provides a unified interface for different ASR engines to work with SegmentManager.
Each adapter normalizes its engine's output to the standard Hypothesis format.

Standard Hypothesis Format (expected by SegmentManager):
{
    "type": "hypothesis",
    "segment_id": str,
    "revision": int,
    "text": str,
    "words": [
        {
            "word": str,
            "start_ms": int,
            "end_ms": int,
            "probability": float,
        },
        ...
    ],
    "confidence": float,
    "start_time_ms": int,  # buffer start
    "end_time_ms": int,    # buffer end
}

Usage:
    from fap.asr.adapter import create_asr_adapter
    
    # Create adapter (auto-detects engine type)
    adapter = create_asr_adapter("whisper", model_size="medium")
    # or
    adapter = create_asr_adapter("google", language_code="en-US")
    
    # Feed audio - returns normalized hypothesis
    hypothesis = adapter.feed(audio_bytes)
    if hypothesis:
        segment_output = segment_manager.ingest(hypothesis)
"""

from abc import ABC, abstractmethod
from typing import Any
import time


class ASRAdapter(ABC):
    """
    Abstract base class for ASR adapters.
    
    Adapters wrap ASR engines and normalize their output to the standard
    Hypothesis format expected by SegmentManager.
    """
    
    @abstractmethod
    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio chunk and return normalized hypothesis.
        
        Args:
            audio: PCM16 audio bytes (16kHz, mono)
            
        Returns:
            Normalized hypothesis dict or None if no transcription available
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset the adapter and underlying engine state."""
        pass
    
    @abstractmethod
    def finalize(self) -> dict | None:
        """
        Finalize the current segment and return any remaining hypothesis.
        Called when stream ends or segment boundary detected.
        
        Returns:
            Final hypothesis dict or None
        """
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name (e.g., 'whisper', 'google')."""
        pass
    
    @property
    @abstractmethod
    def is_streaming(self) -> bool:
        """Return whether the adapter is currently streaming."""
        pass


class WhisperAdapter(ASRAdapter):
    """
    Adapter for faster-whisper ASR engine.
    
    Whisper uses a rolling buffer approach - it re-transcribes
    the entire buffer on each call, providing word-level timestamps.
    """
    
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
        model: Any = None,
        buffer_duration_ms: int = 2000,
    ):
        """
        Initialize Whisper adapter.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to run on (auto, cpu, cuda)
            model: Pre-loaded WhisperModel instance (optional)
            buffer_duration_ms: Rolling buffer duration
        """
        from .whisper_engine import WhisperASR
        
        self._engine = WhisperASR(
            model_size=model_size,
            device=device,
            model=model,
            buffer_duration_ms=buffer_duration_ms,
        )
        self._is_streaming = False
        self._last_hypothesis: dict | None = None
    
    @property
    def provider_name(self) -> str:
        return "whisper"
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
    
    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio to Whisper and return normalized hypothesis.
        
        Whisper already outputs in the expected format, so minimal
        normalization is needed.
        """
        self._is_streaming = True
        
        hypothesis = self._engine.feed(audio)
        
        if hypothesis:
            # Whisper output is already in the correct format
            # Just ensure all required fields are present
            normalized = self._normalize_hypothesis(hypothesis)
            self._last_hypothesis = normalized
            return normalized
        
        return None
    
    def _normalize_hypothesis(self, hypothesis: dict) -> dict:
        """Ensure hypothesis has all required fields."""
        return {
            "type": hypothesis.get("type", "hypothesis"),
            "segment_id": hypothesis.get("segment_id", f"whisper-{int(time.time() * 1000)}"),
            "revision": hypothesis.get("revision", 0),
            "text": hypothesis.get("text", ""),
            "words": hypothesis.get("words", []),
            "confidence": hypothesis.get("confidence", 0.0),
            "start_time_ms": hypothesis.get("start_time_ms", 0),
            "end_time_ms": hypothesis.get("end_time_ms", 0),
        }
    
    def reset(self) -> None:
        """Reset Whisper engine state."""
        self._engine.reset()
        self._is_streaming = False
        self._last_hypothesis = None
    
    def finalize(self) -> dict | None:
        """
        Finalize current segment.
        
        For Whisper, we return the last hypothesis as final.
        """
        self._is_streaming = False
        
        if self._last_hypothesis:
            # Mark as final
            final_hypothesis = self._last_hypothesis.copy()
            final_hypothesis["final"] = True
            self._last_hypothesis = None
            return final_hypothesis
        
        return None


class GoogleAdapter(ASRAdapter):
    """
    Adapter for Google Cloud Speech-to-Text ASR engine.
    
    Google uses true streaming - it returns interim and final results
    asynchronously as audio is processed.
    """
    
    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate_hz: int = 16000,
        credentials_path: str | None = None,
        model: str = "latest_long",
    ):
        """
        Initialize Google Cloud adapter.
        
        Args:
            language_code: BCP-47 language code
            sample_rate_hz: Audio sample rate
            credentials_path: Path to service account JSON
            model: Recognition model
        """
        from .google_engine import GoogleCloudASR
        
        self._engine = GoogleCloudASR(
            language_code=language_code,
            sample_rate_hz=sample_rate_hz,
            credentials_path=credentials_path,
            model=model,
        )
        self._is_streaming = False
        self._last_hypothesis: dict | None = None
        self._revision = 0
    
    @property
    def provider_name(self) -> str:
        return "google"
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming
    
    def feed(self, audio: bytes) -> dict | None:
        """
        Feed audio to Google Cloud and return normalized hypothesis.
        
        Google returns hypotheses asynchronously - we normalize them
        to match the expected format.
        """
        self._is_streaming = True
        
        hypothesis = self._engine.feed(audio)
        
        if hypothesis:
            normalized = self._normalize_hypothesis(hypothesis)
            self._last_hypothesis = normalized
            return normalized
        
        return None
    
    def _normalize_hypothesis(self, hypothesis: dict) -> dict:
        """
        Normalize Google hypothesis to standard format.
        
        Google's format is similar but may have slight differences.
        """
        self._revision += 1
        
        # Normalize words format
        words = []
        for word_info in hypothesis.get("words", []):
            words.append({
                "word": word_info.get("word", ""),
                "start_ms": word_info.get("start_ms", 0),
                "end_ms": word_info.get("end_ms", 0),
                "probability": word_info.get("probability", 0.9),
            })
        
        return {
            "type": "hypothesis",
            "segment_id": hypothesis.get("segment_id", f"google-{int(time.time() * 1000)}"),
            "revision": self._revision,
            "text": hypothesis.get("text", ""),
            "words": words,
            "confidence": hypothesis.get("confidence", 0.9),
            "start_time_ms": hypothesis.get("start_time_ms", 0),
            "end_time_ms": hypothesis.get("end_time_ms", 0),
        }
    
    def reset(self) -> None:
        """Reset Google engine state."""
        self._engine.reset()
        self._is_streaming = False
        self._last_hypothesis = None
        self._revision = 0
    
    def finalize(self) -> dict | None:
        """
        Finalize current segment.
        
        For Google, we return the last hypothesis as final.
        """
        self._is_streaming = False
        
        if self._last_hypothesis:
            final_hypothesis = self._last_hypothesis.copy()
            final_hypothesis["final"] = True
            self._last_hypothesis = None
            return final_hypothesis
        
        return None


def create_asr_adapter(
    provider: str = "whisper",
    **kwargs: Any,
) -> ASRAdapter:
    """
    Factory function to create an ASR adapter.
    
    Args:
        provider: ASR provider ("whisper" or "google")
        **kwargs: Provider-specific arguments
        
    Returns:
        ASRAdapter instance
        
    Examples:
        # Whisper adapter
        adapter = create_asr_adapter("whisper", model_size="medium")
        
        # Google adapter
        adapter = create_asr_adapter("google", language_code="ko-KR")
        
        # With pre-loaded model
        adapter = create_asr_adapter("whisper", model=shared_model)
    """
    import os
    
    # Get provider from env if not specified
    if provider is None:
        provider = os.getenv("ASR_PROVIDER", "whisper").lower()
    
    print(f"🔧 Creating ASR adapter: {provider}")
    
    if provider == "whisper":
        # Default kwargs for whisper
        whisper_defaults = {
            "model_size": os.getenv("WHISPER_MODEL_SIZE", "medium"),
            "device": os.getenv("WHISPER_DEVICE", "auto"),
        }
        whisper_defaults.update(kwargs)
        return WhisperAdapter(**whisper_defaults)
    
    elif provider == "google":
        # Default kwargs for google
        google_defaults = {
            "language_code": os.getenv("GOOGLE_ASR_LANGUAGE", "en-US"),
            "credentials_path": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            "model": os.getenv("GOOGLE_ASR_MODEL", "latest_long"),
        }
        google_defaults.update(kwargs)
        return GoogleAdapter(**google_defaults)
    
    else:
        raise ValueError(
            f"Unknown ASR provider: {provider}. "
            f"Supported providers: whisper, google"
        )