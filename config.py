"""Central configuration for the sermon translation pipeline.

Everything that you might want to tune lives here, so the rest of the code
stays clean. You can also override any of these from the command line
(see main.py) without editing this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ISO-639-1 code -> human-readable name. Whisper reports codes like "en"/"pl";
# the translator prompt reads better with a real language name. Extend freely.
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "pl": "Polish",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "uk": "Ukrainian",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "sk": "Slovak",
    "ro": "Romanian",
    "hu": "Hungarian",
    "el": "Greek",
    "tr": "Turkish",
    "ar": "Arabic",
    "he": "Hebrew",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "sw": "Swahili",
}


def language_name(code: str | None) -> str:
    """Turn a whisper language code into a readable name (fallback: the code)."""
    if not code:
        return "the source language"
    return LANGUAGE_NAMES.get(code.lower(), code)


@dataclass
class WhisperConfig:
    """Settings for the faster-whisper transcription stage."""

    # "tiny", "base", "small", "medium", "large-v3", "distil-large-v3", or a path.
    # large-v3 is most accurate; distil-large-v3 is ~2x faster with minimal loss
    # and is a good default for near-real-time on a decent GPU.
    model_size: str = "large-v3"

    # "cuda" if you have an NVIDIA GPU, otherwise "cpu". "auto" lets CTranslate2
    # decide, but being explicit is safer.
    device: str = "cuda"

    # "float16" on GPU, "int8_float16" to save VRAM, "int8" on CPU.
    compute_type: str = "float16"

    # None = auto-detect the spoken language. Set e.g. "pl" to force it.
    language: str | None = None

    # Higher beam = slightly better accuracy, slower. 5 is a good balance;
    # drop to 1 for the lowest latency in real-time mode.
    beam_size: int = 5

    # Silero VAD trims silence so pauses between sentences don't waste compute
    # and don't get hallucinated into text. Recommended on for sermons.
    vad_filter: bool = True


@dataclass
class LlamaConfig:
    """Settings for the llama.cpp (Qwen3.5) translation stage."""

    # Path to your downloaded .gguf file. You MUST set this (or pass --model).
    # e.g. models/Qwen3.5-9B-Instruct-Q4_K_M.gguf
    model_path: str = "models/qwen3.5.gguf"

    # Context window. Sermons keep growing, so we only feed a rolling window of
    # recent lines (see context_turns), but the model still needs room for them
    # plus its own output.
    n_ctx: int = 8192

    # -1 = offload every layer to the GPU. If you run out of VRAM, lower this to
    # offload only some layers (the rest run on CPU).
    n_gpu_layers: int = -1

    # Low temperature keeps translations consistent line-to-line, which matters
    # for names and recurring terminology in a sermon.
    temperature: float = 0.3
    top_p: float = 0.9

    # Max tokens for a single translated line. Sermon segments are short.
    max_tokens: int = 512

    # Qwen3.x can "think" before answering. For live translation we want the
    # answer immediately, so we disable it. Leave True.
    disable_thinking: bool = True

    # How many previous (source -> translation) pairs to feed back in as context.
    # This is what keeps names/terms consistent across the service.
    context_turns: int = 6


@dataclass
class Config:
    """Top-level config passed around the whole pipeline."""

    # What language to translate INTO.
    target_language: str = "English"

    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    llama: LlamaConfig = field(default_factory=LlamaConfig)

    # Show the original text alongside the translation. Handy while testing.
    show_original: bool = True

    # Where per-session .log and .jsonl files are written.
    log_dir: str = "logs"
    logging_enabled: bool = True
