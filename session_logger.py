"""Per-session logging to the logs/ directory.

Writes two files per run:
  * logs/session_<timestamp>.log    human-readable, one block per line
  * logs/session_<timestamp>.jsonl  one JSON record per segment, for analysis

The console output (rich panels) stays clean; the detail goes to file.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from models import SegmentMetrics, format_timestamp


class SessionLogger:
    def __init__(self, log_dir: str = "logs", enabled: bool = True) -> None:
        self.enabled = enabled
        self._jsonl = None
        if not enabled:
            self.log_path = None
            self.jsonl_path = None
            return

        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"session_{stamp}.log")
        self.jsonl_path = os.path.join(log_dir, f"session_{stamp}.jsonl")

        self.logger = logging.getLogger(f"sermon.{stamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # don't spam the console
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))
        self.logger.addHandler(handler)

        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")

    def info(self, message: str) -> None:
        if self.enabled:
            self.logger.info(message)

    def log_segment(self, m: SegmentMetrics) -> None:
        if not self.enabled:
            return
        rtf = m.rtf
        rtf_txt = "n/a" if rtf != rtf else f"{rtf:.2f}"  # rtf != rtf catches NaN
        self.logger.info(
            f"[{format_timestamp(m.rec_start)}] "
            f"total={m.total_s:.2f}s  whisper={m.stt_s:.2f}s  llm={m.llm_s:.2f}s  "
            f"overhead={m.overhead_s * 1000:.0f}ms  render={m.render_s * 1000:.0f}ms  "
            f"RTF={rtf_txt}  tok={m.completion_tokens}  tok/s={m.tok_per_s:.1f}  "
            f"think={'YES' if m.thinking else 'no'}  lang={m.language}"
        )
        self.logger.info(f"    SRC: {m.text}")
        self.logger.info(f"    OUT: {m.translation}")
        self._jsonl.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")
        self._jsonl.flush()

    def log_summary(self, text: str) -> None:
        if self.enabled:
            for line in text.splitlines():
                self.logger.info(line)

    def close(self) -> None:
        if self.enabled and self._jsonl:
            self._jsonl.close()
