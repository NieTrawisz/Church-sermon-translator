"""Speech-to-text using faster-whisper.

Two entry points, both of which yield the SAME `TranscriptSegment` objects so
everything downstream (translation, display) is identical:

  * transcribe_file(path)   -> for your current test with a recorded video/audio
  * stream_microphone()     -> for the real-time target (live mic input)

Because both are generators, the pipeline can start translating segment #1 while
whisper is still working on segment #2.
"""

from __future__ import annotations

from typing import Iterator

from faster_whisper import WhisperModel

from config import WhisperConfig
from models import TranscriptSegment


class Transcriber:
    def __init__(self, cfg: WhisperConfig) -> None:
        self.cfg = cfg
        print(
            f"[whisper] loading model '{cfg.model_size}' on {cfg.device} "
            f"({cfg.compute_type}) ..."
        )
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )
        print("[whisper] ready.")

    # -- File / recorded input (what you're testing first) -------------------

    def transcribe_file(self, path: str) -> Iterator[TranscriptSegment]:
        """Transcribe an audio OR video file.

        faster-whisper decodes media through ffmpeg, so you can point this
        straight at an .mp4 / .mkv / .wav / .mp3 etc. Segments are yielded as
        soon as whisper produces them.
        """
        segments, info = self.model.transcribe(
            path,
            language=self.cfg.language,      # None => auto-detect
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
        )
        lang = info.language
        print(
            f"[whisper] detected language: {lang} "
            f"(p={info.language_probability:.2f})"
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                yield TranscriptSegment(seg.start, seg.end, text, lang)

    # -- Live microphone input (the real-time target) ------------------------

    def stream_microphone(
        self,
        samplerate: int = 16000,
        block_seconds: float = 0.5,
        silence_rms: float = 0.010,
        min_silence_blocks: int = 2,
        max_utterance_seconds: float = 20.0,
    ) -> Iterator[TranscriptSegment]:
        """Capture the mic and yield segments once a speaker pauses.

        This is a simple energy-based approach: we buffer audio until we hear a
        short silence (end of a phrase) or hit a max length, then transcribe
        that buffer. It's intentionally dependency-light so you can test live
        input quickly; swap in a proper VAD (e.g. webrtcvad / Silero) later for
        cleaner phrase boundaries.

        Requires: `pip install sounddevice numpy`
        """
        import queue

        import numpy as np
        import sounddevice as sd

        block_frames = int(samplerate * block_seconds)
        max_frames = int(samplerate * max_utterance_seconds)
        audio_q: "queue.Queue[np.ndarray]" = queue.Queue()

        def _callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                print(f"[mic] {status}")
            audio_q.put(indata[:, 0].copy())  # mono

        print("[mic] listening... (Ctrl+C to stop)")
        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            blocksize=block_frames,
            dtype="float32",
            callback=_callback,
        ):
            buffer = np.zeros(0, dtype=np.float32)
            silent_run = 0
            has_speech = False

            while True:
                block = audio_q.get()
                rms = float(np.sqrt(np.mean(block**2)) + 1e-9)
                is_silent = rms < silence_rms

                if not is_silent:
                    has_speech = True
                    silent_run = 0
                elif has_speech:
                    silent_run += 1

                buffer = np.concatenate([buffer, block])

                phrase_ended = has_speech and silent_run >= min_silence_blocks
                too_long = len(buffer) >= max_frames

                if phrase_ended or too_long:
                    yield from self._transcribe_array(buffer, samplerate)
                    buffer = np.zeros(0, dtype=np.float32)
                    silent_run = 0
                    has_speech = False

    def _transcribe_array(self, audio, samplerate: int) -> Iterator[TranscriptSegment]:
        """Transcribe a raw float32 numpy buffer (used by the mic path)."""
        # faster-whisper expects 16 kHz mono float32; the mic stream already is.
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language,
            beam_size=self.cfg.beam_size,
            vad_filter=self.cfg.vad_filter,
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                yield TranscriptSegment(seg.start, seg.end, text, info.language)
