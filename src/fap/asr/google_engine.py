# asr/google_engine.py
"""
Google Cloud Speech-to-Text V1 ASR Engine implementation.

Cloud-based endless streaming speech recognition using Google Cloud.
Automatically handles stream restarts for unlimited duration transcription.

Based on Google's ResumableMicrophoneStream example for endless streaming.

Requires: pip install google-cloud-speech
"""

import os
import time
import queue
import threading
from typing import Generator

from .base import ASREngine, Hypothesis, WordInfo


# Google's streaming limit is ~305 seconds, we restart at 4 minutes to be safe
STREAMING_LIMIT_MS = 240000  # 4 minutes


class GoogleCloudASR(ASREngine):
    """
    Google Cloud Speech-to-Text V1 endless streaming ASR.
    
    Uses bidirectional streaming for real-time transcription.
    Automatically restarts stream every 4 minutes for unlimited duration.
    Maintains audio continuity across restarts using bridging buffer.
    """

    def __init__(
        self,
        language_code: str = "en-US",
        sample_rate_hz: int = 16000,
        credentials_path: str | None = None,
        interim_results: bool = True,
        model: str = "latest_long",
    ):
        """
        Initialize Google Cloud ASR V1 engine for endless streaming.

        Args:
            language_code: BCP-47 language code (e.g., "en-US", "ko-KR")
            sample_rate_hz: Audio sample rate (default 16000)
            credentials_path: Path to service account JSON (optional)
            interim_results: Whether to return partial results
            model: Recognition model (default, latest_long, latest_short, phone_call, video)
        """
        print(f"🌐 Initializing Google Cloud ASR V1 (Endless): {language_code}, model={model}")

        # Set credentials if provided
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            print(f"   Using credentials from: {credentials_path}")
        elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            print(f"   Using credentials from env")
        else:
            print("   ⚠️ No credentials set - using default application credentials")

        # Import Google Cloud Speech V1
        try:
            from google.cloud import speech
            self.speech = speech
        except ImportError:
            raise ImportError(
                "google-cloud-speech not installed. "
                "Run: pip install google-cloud-speech"
            )

        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.interim_results = interim_results
        self.model = model

        # Create client
        self.client = self.speech.SpeechClient()

        # Create config
        self.recognition_config = self.speech.RecognitionConfig(
            encoding=self.speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
            model=self.model,
            max_alternatives=1,
        )

        self.streaming_config = self.speech.StreamingRecognitionConfig(
            config=self.recognition_config,
            interim_results=self.interim_results,
        )

        # State
        self.segment_id = f"google-seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.session_start_time_ms = int(time.time() * 1000)  # Start of entire session
        
        # Streaming state
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._response_queue: queue.Queue[Hypothesis | None] = queue.Queue()
        self._finals_queue: queue.Queue[Hypothesis | None] = queue.Queue()  # Separate queue for finals!
        self._streaming_thread: threading.Thread | None = None
        self._is_streaming = False
        
        # Endless streaming state (from ResumableMicrophoneStream)
        self._restart_counter = 0
        self._stream_start_time_ms = 0
        self._audio_input: list[bytes] = []  # Current stream's audio
        self._last_audio_input: list[bytes] = []  # Previous stream's audio (for bridging)
        self._result_end_time_ms = 0
        self._is_final_end_time_ms = 0
        self._final_request_end_time_ms = 0
        self._bridging_offset_ms = 0
        self._new_stream = True
        
        # Lock for thread safety
        self._lock = threading.Lock()

        print("✅ Google Cloud ASR V1 (Endless) initialized")

    @property
    def provider_name(self) -> str:
        return "google"

    def _get_current_time_ms(self) -> int:
        """Get current time in milliseconds."""
        return int(time.time() * 1000)

    def _audio_generator(self) -> Generator[bytes, None, None]:
        """
        Generate audio chunks for streaming API.
        Handles bridging audio across stream restarts for continuity.
        """
        while self._is_streaming:
            # Check if we need to restart stream (approaching 4 minute limit)
            elapsed = self._get_current_time_ms() - self._stream_start_time_ms
            if elapsed > STREAMING_LIMIT_MS:
                print(f"⏰ Stream limit approaching, preparing restart...")
                break
            
            # Handle bridging audio from previous stream on restart
            if self._new_stream and self._last_audio_input:
                # Calculate how much audio to bridge
                chunk_time = STREAMING_LIMIT_MS / len(self._last_audio_input) if self._last_audio_input else 0
                
                if chunk_time > 0:
                    # Clamp bridging offset
                    if self._bridging_offset_ms < 0:
                        self._bridging_offset_ms = 0
                    if self._bridging_offset_ms > self._final_request_end_time_ms:
                        self._bridging_offset_ms = self._final_request_end_time_ms

                    # Calculate which chunks to replay
                    chunks_from_ms = round(
                        (self._final_request_end_time_ms - self._bridging_offset_ms) / chunk_time
                    )
                    
                    # Update bridging offset for next restart
                    self._bridging_offset_ms = round(
                        (len(self._last_audio_input) - chunks_from_ms) * chunk_time
                    )

                    # Yield bridging chunks
                    for i in range(chunks_from_ms, len(self._last_audio_input)):
                        if i < len(self._last_audio_input):
                            yield self._last_audio_input[i]

                self._new_stream = False

            # Get new audio chunk
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is None:  # Poison pill - stop streaming
                    break
                
                # Store for potential bridging on restart
                with self._lock:
                    self._audio_input.append(chunk)
                
                yield chunk
                
            except queue.Empty:
                continue

    def _stream_thread(self):
        """
        Background thread for handling endless streaming recognition.
        Automatically restarts stream when approaching time limit.
        """
        while self._is_streaming:
            try:
                # Reset for new stream
                self._stream_start_time_ms = self._get_current_time_ms()
                
                with self._lock:
                    self._audio_input = []
                
                print(f"🎙️ Starting stream #{self._restart_counter + 1} at {self._get_session_time_str()}")

                # Create request generator
                requests = (
                    self.speech.StreamingRecognizeRequest(audio_content=content)
                    for content in self._audio_generator()
                )

                # Start streaming recognition
                responses = self.client.streaming_recognize(
                    self.streaming_config,
                    requests,
                )

                # Process responses
                self._process_responses(responses)
                
                # Stream ended - prepare for restart if still active
                if self._is_streaming:
                    # Save state for bridging
                    if self._result_end_time_ms > 0:
                        self._final_request_end_time_ms = self._is_final_end_time_ms
                    
                    self._result_end_time_ms = 0
                    
                    with self._lock:
                        self._last_audio_input = self._audio_input.copy()
                        self._audio_input = []
                    
                    self._restart_counter += 1
                    self._new_stream = True
                    
                    print(f"🔄 Stream restart #{self._restart_counter} at {self._get_session_time_str()}")

            except Exception as e:
                error_str = str(e).lower()
                
                # Check if this is a normal termination condition
                is_normal_end = (
                    "iterating requests" in error_str or 
                    "cancelled" in error_str
                )
                
                # Audio timeout means no audio is being sent - likely stream ended
                is_audio_timeout = "audio timeout" in error_str
                
                if is_normal_end:
                    # Normal end of stream - continue only if still streaming
                    if self._is_streaming:
                        continue
                elif is_audio_timeout:
                    # Audio timeout - only restart if we're still actively streaming
                    # AND there's audio in the queue waiting
                    if self._is_streaming and not self._audio_queue.empty():
                        print(f"⚠️ Audio timeout, but queue has data - restarting...")
                        continue
                    else:
                        print(f"🛑 Audio timeout - stopping stream (no pending audio)")
                        self._is_streaming = False
                        break
                else:
                    print(f"❌ Google Cloud streaming error: {e}")
                
                if not self._is_streaming:
                    break
                    
                # Brief pause before retry on error
                time.sleep(0.5)
        
        print(f"🛑 Streaming thread ended. Total restarts: {self._restart_counter}")

    def _get_session_time_str(self) -> str:
        """Get formatted session elapsed time."""
        elapsed_ms = self._get_current_time_ms() - self.session_start_time_ms
        seconds = (elapsed_ms // 1000) % 60
        minutes = (elapsed_ms // 60000) % 60
        hours = elapsed_ms // 3600000
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def _calculate_corrected_time(self, result_time_ms: int) -> int:
        """
        Calculate corrected timestamp accounting for stream restarts.
        
        Args:
            result_time_ms: Time from current stream result
            
        Returns:
            Corrected time relative to session start
        """
        return (
            result_time_ms
            - self._bridging_offset_ms
            + (STREAMING_LIMIT_MS * self._restart_counter)
        )

    def _process_responses(self, responses):
        """Process streaming responses from Google Cloud."""
        for response in responses:
            if not self._is_streaming:
                break
            
            # Check if approaching stream limit
            elapsed = self._get_current_time_ms() - self._stream_start_time_ms
            if elapsed > STREAMING_LIMIT_MS:
                break
                
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            alternative = result.alternatives[0]
            transcript = alternative.transcript.strip()

            if not transcript:
                continue

            # Get result end time
            result_seconds = result.result_end_time.seconds if result.result_end_time.seconds else 0
            result_micros = result.result_end_time.microseconds if result.result_end_time.microseconds else 0
            self._result_end_time_ms = int((result_seconds * 1000) + (result_micros / 1000))

            # Extract word timestamps with correction for restarts
            words_with_timestamps: list[WordInfo] = []
            
            if alternative.words:
                for word_info in alternative.words:
                    start_ms = int(word_info.start_time.total_seconds() * 1000)
                    end_ms = int(word_info.end_time.total_seconds() * 1000)
                    
                    # Apply correction for stream restarts
                    corrected_start = self._calculate_corrected_time(start_ms)
                    corrected_end = self._calculate_corrected_time(end_ms)
                    
                    words_with_timestamps.append({
                        "word": word_info.word,
                        "start_ms": self.session_start_time_ms + corrected_start,
                        "end_ms": self.session_start_time_ms + corrected_end,
                        "probability": alternative.confidence if alternative.confidence else 0.9,
                    })

            # If no word timestamps, create approximate ones
            if not words_with_timestamps and transcript:
                words = transcript.split()
                word_duration_ms = 200
                corrected_time = self._calculate_corrected_time(self._result_end_time_ms - (len(words) * word_duration_ms))
                current_ms = self.session_start_time_ms + corrected_time
                
                for word in words:
                    words_with_timestamps.append({
                        "word": word,
                        "start_ms": current_ms,
                        "end_ms": current_ms + word_duration_ms,
                        "probability": 0.9,
                    })
                    current_ms += word_duration_ms

            # Get Google's is_final flag - THIS IS THE AUTHORITATIVE INDICATOR
            is_final = result.is_final

            self.revision += 1

            # Calculate corrected display time
            corrected_time_ms = self._calculate_corrected_time(self._result_end_time_ms)

            hypothesis: Hypothesis = {
                "type": "hypothesis",
                "segment_id": self.segment_id,
                "revision": self.revision,
                "text": transcript,
                "words": words_with_timestamps,
                "confidence": alternative.confidence if alternative.confidence else 0.9,
                "start_time_ms": self.session_start_time_ms,
                "end_time_ms": self.session_start_time_ms + corrected_time_ms,
                "is_final_from_google": is_final,  # CRITICAL: Pass Google's is_final flag
            }

            # Log
            if is_final:
                self._is_final_end_time_ms = self._result_end_time_ms
                
            prefix = "✅" if is_final else "⏳"
            time_str = self._get_session_time_str()
            print(f"{prefix} [{time_str}] Google ASR (rev {self.revision}): \"{transcript}\"")
            
            # Log word timestamps for final results
            if is_final and words_with_timestamps:
                print(f"📝 Words with timestamps (revision {self.revision}):")
                for w in words_with_timestamps:
                    print(f"   [{w['start_ms']:5d} - {w['end_ms']:5d}ms] \"{w['word']}\" (prob: {w['probability']:.2f})")

            # Put hypothesis in queues for main thread
            # Finals go to BOTH queues so they can't be missed
            self._response_queue.put(hypothesis)
            if is_final:
                self._finals_queue.put(hypothesis)

    def _start_streaming(self):
        """Start the streaming thread if not already running."""
        if not self._is_streaming:
            self._is_streaming = True
            self.session_start_time_ms = self._get_current_time_ms()
            self._streaming_thread = threading.Thread(target=self._stream_thread, daemon=True)
            self._streaming_thread.start()
            print("🎙️ Google Cloud endless streaming started")

    def feed(self, audio: bytes) -> Hypothesis | None:
        """
        Feed audio chunk and return hypothesis if available.

        Args:
            audio: PCM16 audio bytes (16kHz)

        Returns:
            Hypothesis dict if transcription available, None otherwise
        """
        # Start streaming on first audio
        if not self._is_streaming:
            self._start_streaming()

        # Add audio to queue
        self._audio_queue.put(audio)

        # Return latest hypothesis if available (non-blocking)
        try:
            return self._response_queue.get_nowait()
        except queue.Empty:
            return None

    def reset(self) -> None:
        """Reset the engine state for a new segment/session."""
        # Stop current stream
        self._is_streaming = False
        self._audio_queue.put(None)

        if self._streaming_thread and self._streaming_thread.is_alive():
            self._streaming_thread.join(timeout=2.0)

        # Clear queues
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self._response_queue.empty():
            try:
                self._response_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self._finals_queue.empty():
            try:
                self._finals_queue.get_nowait()
            except queue.Empty:
                break

        # Reset all state
        self._audio_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._finals_queue = queue.Queue()
        self.segment_id = f"google-seg-{int(time.time() * 1000)}"
        self.revision = 0
        self.session_start_time_ms = int(time.time() * 1000)
        
        # Reset endless streaming state
        self._restart_counter = 0
        self._stream_start_time_ms = 0
        self._audio_input = []
        self._last_audio_input = []
        self._result_end_time_ms = 0
        self._is_final_end_time_ms = 0
        self._final_request_end_time_ms = 0
        self._bridging_offset_ms = 0
        self._new_stream = True

        print("🔄 Google Cloud ASR reset (ready for new session)")

    def __del__(self):
        """Cleanup on destruction."""
        self._is_streaming = False
        try:
            self._audio_queue.put(None)
        except:
            pass