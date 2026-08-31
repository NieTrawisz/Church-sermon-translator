"""Orchestration: wire transcriber -> translator -> display together, timing
each stage and logging everything to logs/.

The per-segment loop measures three buckets that make up the speech->text
latency:
  * whisper  (stt_s):  time to pull the next segment out of the transcriber
  * llm      (llm_s):  time for the translation call
  * overhead (overhead_s): the glue in between
Terminal drawing (render_s) is measured too, but kept out of the total since
it isn't part of the real-time latency you care about.
"""

from __future__ import annotations

import os
from statistics import mean
from time import perf_counter
from typing import Iterator

from config import Config
from display import Display
from models import SegmentMetrics, TranscriptSegment
from session_logger import SessionLogger
from transcriber import Transcriber
from translator import Translator


class TranslationPipeline:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.logger = SessionLogger(cfg.log_dir, cfg.logging_enabled)
        self.logger.info(
            f"=== session start | target={cfg.target_language} | "
            f"whisper={cfg.whisper.model_size} ({cfg.whisper.device}/"
            f"{cfg.whisper.compute_type}) | "
            f"llm={os.path.basename(cfg.llama.model_path)} "
            f"n_ctx={cfg.llama.n_ctx} n_gpu_layers={cfg.llama.n_gpu_layers} "
            f"context_turns={cfg.llama.context_turns} "
            f"no_think={cfg.llama.disable_thinking} ==="
        )

        t = perf_counter()
        self.transcriber = Transcriber(cfg.whisper)
        self.logger.info(f"whisper model loaded in {perf_counter() - t:.1f}s")

        t = perf_counter()
        self.translator = Translator(cfg)
        self.logger.info(f"llm model loaded in {perf_counter() - t:.1f}s")

        self.display = Display(cfg)

    # -- Shared inner loop ---------------------------------------------------

    def _consume(self, segments: Iterator[TranscriptSegment]) -> None:
        metrics: list[SegmentMetrics] = []
        it = iter(segments)
        index = 0

        while True:
            t0 = perf_counter()
            try:
                segment = next(it)          # whisper produces the next chunk here
            except StopIteration:
                break
            stt_s = perf_counter() - t0

            t1 = perf_counter()
            result = self.translator.translate(segment.text, segment.language)
            llm_s = perf_counter() - t1

            overhead_s = 0.0                 # (glue is negligible; measured below)
            t2 = perf_counter()
            m = SegmentMetrics(
                index=index,
                rec_start=segment.start,
                rec_end=segment.end,
                text=segment.text,
                translation=result.text,
                language=segment.language,
                stt_s=stt_s,
                llm_s=llm_s,
                overhead_s=perf_counter() - t2,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                thinking=result.thinking,
            )

            t3 = perf_counter()
            self.display.show(m)
            m.render_s = perf_counter() - t3

            self.logger.log_segment(m)
            metrics.append(m)
            index += 1

        self._finish(metrics)

    def _finish(self, metrics: list[SegmentMetrics]) -> None:
        if not metrics:
            self.logger.info("no segments produced.")
            self.logger.close()
            return

        rtfs = [m.rtf for m in metrics if m.rtf == m.rtf]  # drop NaN
        n_think = sum(1 for m in metrics if m.thinking)
        summary = (
            f"segments: {len(metrics)}\n"
            f"avg whisper: {mean(m.stt_s for m in metrics):.2f}s   "
            f"avg llm: {mean(m.llm_s for m in metrics):.2f}s\n"
            f"avg RTF: {mean(rtfs):.2f}" + ("  (<1 = real-time capable)" if rtfs else "") + "\n"
            f"avg llm speed: {mean(m.tok_per_s for m in metrics if m.tok_per_s):.0f} tok/s   "
            f"avg out tokens: {mean(m.completion_tokens for m in metrics):.0f}\n"
            f"lines where model was thinking: {n_think}/{len(metrics)}"
        )
        self.display.summary(summary)
        self.logger.log_summary("=== summary ===\n" + summary)
        if self.logger.log_path:
            print(f"\n[log] {self.logger.log_path}")
        self.logger.close()

    # -- Entry points --------------------------------------------------------

    def run_file(self, path: str) -> None:
        self.display.header(f"file: {path}")
        self._consume(self.transcriber.transcribe_file(path))

    def run_microphone(self) -> None:
        self.display.header("live microphone")
        try:
            self._consume(self.transcriber.stream_microphone())
        except KeyboardInterrupt:
            print("\n[pipeline] stopped.")
            self.logger.close()
