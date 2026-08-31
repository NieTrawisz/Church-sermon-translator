"""Shared, dependency-free data types and helpers.

Kept separate so that lightweight modules (display, pipeline, logger) can use
these without importing the heavy speech/LLM libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isnan


def format_timestamp(seconds: float) -> str:
    """Seconds -> mm:ss (recording position)."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


@dataclass
class TranscriptSegment:
    """One chunk of recognized speech."""

    start: float          # seconds from the beginning of the recording
    end: float            # seconds from the beginning of the recording
    text: str             # the recognized text
    language: str | None  # detected (or forced) language code, e.g. "pl"


@dataclass
class TranslationResult:
    """What the translator returns, including cost/latency info."""

    text: str                 # cleaned translation (what you display)
    raw: str                  # exact model output, before cleanup
    prompt_tokens: int        # tokens fed in (context + system + line)
    completion_tokens: int    # tokens generated (this is what costs time)
    thinking: bool            # did the model emit a non-empty <think> block?


@dataclass
class SegmentMetrics:
    """Everything measured for one speech -> translation cycle."""

    index: int
    rec_start: float          # audio-timeline start (s)
    rec_end: float            # audio-timeline end (s)
    text: str                 # source text
    translation: str
    language: str | None

    stt_s: float              # speech-to-text (whisper) time
    llm_s: float              # translation (llama.cpp) time
    overhead_s: float         # glue between stages
    render_s: float = 0.0     # terminal drawing time (logged, not in total)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    thinking: bool = False

    @property
    def audio_dur(self) -> float:
        return max(self.rec_end - self.rec_start, 0.0)

    @property
    def total_s(self) -> float:
        """Speech -> translated-text latency (excludes terminal drawing)."""
        return self.stt_s + self.llm_s + self.overhead_s

    @property
    def rtf(self) -> float:
        """Real-time factor: processing_time / audio_duration.

        < 1.0 means faster than real time (good). > 1.0 means you're falling
        behind the speaker.
        """
        d = self.audio_dur
        return self.total_s / d if d > 0 else float("nan")

    @property
    def tok_per_s(self) -> float:
        return self.completion_tokens / self.llm_s if self.llm_s > 0 else 0.0

    def to_dict(self) -> dict:
        rtf = self.rtf
        return {
            "index": self.index,
            "rec_start": round(self.rec_start, 3),
            "rec_end": round(self.rec_end, 3),
            "audio_dur": round(self.audio_dur, 3),
            "stt_s": round(self.stt_s, 4),
            "llm_s": round(self.llm_s, 4),
            "overhead_s": round(self.overhead_s, 4),
            "render_s": round(self.render_s, 4),
            "total_s": round(self.total_s, 4),
            "rtf": None if isnan(rtf) else round(rtf, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "tok_per_s": round(self.tok_per_s, 2),
            "thinking": self.thinking,
            "language": self.language,
            "source": self.text,
            "translation": self.translation,
        }
