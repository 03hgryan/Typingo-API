# globalAccumulator.py
"""
Global transcript accumulator using word-level storage.

This is the append-only immutable history layer.

Key invariants:
- Only accepts WordInfo objects (not strings)
- Append-only (never modifies existing words)
- Can render text on demand, but words are the source of truth
"""

from typing import TypedDict


class WordInfo(TypedDict):
    """A single word with timing and confidence."""
    word: str
    start_ms: int
    end_ms: int
    probability: float


class SegmentInfo(TypedDict):
    """A finalized segment."""
    segment_id: str
    segment_index: int
    words: list[WordInfo]
    text: str
    start_ms: int
    end_ms: int


class GlobalAccumulator:
    """
    Accumulates finalized words across segments.
    
    This is the single source of truth for the transcript.
    Words are stored directly - no string diffing needed.
    """

    def __init__(self):
        # All finalized words in order
        self.words: list[WordInfo] = []
        
        # Segment boundaries for export
        self.segments: list[SegmentInfo] = []
        
        # Track current segment being accumulated
        self._current_segment_id: str | None = None
        self._current_segment_words: list[WordInfo] = []
        self._segment_count = 0

    def append_words(self, segment_id: str, words: list[WordInfo]) -> None:
        """
        Append finalized words to the accumulator.
        
        This is the ONLY way to add content. No diffing, no guessing.
        
        Args:
            segment_id: The segment these words belong to
            words: List of WordInfo to append
        """
        if not words:
            return
        
        # Handle segment transition
        if segment_id != self._current_segment_id:
            # Finalize previous segment if exists
            if self._current_segment_id is not None and self._current_segment_words:
                self._finalize_current_segment()
            
            # Start new segment
            self._current_segment_id = segment_id
            self._current_segment_words = []
        
        # Append words
        for word in words:
            self.words.append(word)
            self._current_segment_words.append(word)
        
        # Log
        text = " ".join(w["word"] for w in words)
        print(f"📝 Accumulated: \"{text}\" (total: {len(self.words)} words)")

    def _finalize_current_segment(self) -> None:
        """Finalize the current segment and add to segments list."""
        if not self._current_segment_words:
            return
        
        segment: SegmentInfo = {
            "segment_id": self._current_segment_id or "unknown",
            "segment_index": self._segment_count,
            "words": list(self._current_segment_words),
            "text": " ".join(w["word"] for w in self._current_segment_words),
            "start_ms": self._current_segment_words[0]["start_ms"],
            "end_ms": self._current_segment_words[-1]["end_ms"],
        }
        
        self.segments.append(segment)
        self._segment_count += 1
        
        print(f"📦 Segment {segment['segment_index']} finalized: \"{segment['text']}\"")

    def finalize(self) -> None:
        """Finalize any remaining segment (call on stream end)."""
        if self._current_segment_words:
            self._finalize_current_segment()
            self._current_segment_id = None
            self._current_segment_words = []

    def get_full_transcript(self) -> str:
        """Render the full transcript as a string."""
        return " ".join(w["word"] for w in self.words)

    def get_all_words(self) -> list[WordInfo]:
        """Get all accumulated words."""
        return list(self.words)

    def get_segments(self) -> list[SegmentInfo]:
        """
        Get all finalized segments.
        
        Note: Current in-progress segment is NOT included.
        Call finalize() first if you need everything.
        """
        # Include current segment if it has words
        segments = list(self.segments)
        
        if self._current_segment_words:
            current: SegmentInfo = {
                "segment_id": self._current_segment_id or "unknown",
                "segment_index": self._segment_count,
                "words": list(self._current_segment_words),
                "text": " ".join(w["word"] for w in self._current_segment_words),
                "start_ms": self._current_segment_words[0]["start_ms"],
                "end_ms": self._current_segment_words[-1]["end_ms"],
            }
            segments.append(current)
        
        return segments

    def get_word_count(self) -> int:
        """Get total word count."""
        return len(self.words)

    def clear(self) -> None:
        """Clear all accumulated data (for testing)."""
        self.words = []
        self.segments = []
        self._current_segment_id = None
        self._current_segment_words = []
        self._segment_count = 0