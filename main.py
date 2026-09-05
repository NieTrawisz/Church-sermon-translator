"""Command-line entry point.

Examples
--------
Test with a recorded video/audio file:
    python main.py --file service.mp4 --model models/qwen3.5-9b-Q4_K_M.gguf \
        --target-lang English

Force the spoken language (skip auto-detect) and translate into Polish:
    python main.py --file sermon.wav --source-lang en --target-lang Polish \
        --model models/qwen3.5-9b-Q4_K_M.gguf

Live microphone (real-time target):
    python main.py --mic --model models/qwen3.5-9b-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import sys

from config import Config
from pipeline import TranslationPipeline


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()

    cfg.target_language = args.target_lang
    cfg.show_original = not args.no_original

    cfg.whisper.model_size = args.whisper_model
    cfg.whisper.device = args.device
    cfg.whisper.compute_type = args.compute_type
    cfg.whisper.language = args.source_lang  # None => auto-detect

    cfg.llama.model_path = args.model
    if args.n_gpu_layers is not None:
        cfg.llama.n_gpu_layers = args.n_gpu_layers
    cfg.llama.n_ctx = args.n_ctx
    cfg.llama.context_turns = args.context_turns

    cfg.log_dir = args.log_dir
    cfg.logging_enabled = not args.no_log
    cfg.llama.verbose = args.verbose_llm

    return cfg


def main() -> int:
    p = argparse.ArgumentParser(description="Sermon / church service translator")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="Path to a recorded audio/video file to test with")
    src.add_argument("--mic", action="store_true", help="Use live microphone input")

    p.add_argument("--model", required=True, help="Path to the Qwen3.5 .gguf file")
    p.add_argument("--target-lang", default="English", help="Language to translate INTO")
    p.add_argument(
        "--source-lang",
        default=None,
        help="Spoken language code (e.g. pl, en). Omit to auto-detect.",
    )

    p.add_argument("--whisper-model", default="large-v3", help="faster-whisper model size/path")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "auto"])
    p.add_argument("--compute-type", default="float16", help="e.g. float16, int8_float16, int8")

    p.add_argument("--n-gpu-layers", type=int, default=None, help="LLM layers on GPU (-1 = all)")
    p.add_argument("--n-ctx", type=int, default=8192, help="LLM context window")
    p.add_argument("--context-turns", type=int, default=6, help="Previous lines fed back as context")

    p.add_argument("--no-original", action="store_true", help="Hide the source text, show only translation")
    p.add_argument("--log-dir", default="logs", help="Directory for session .log / .jsonl files")
    p.add_argument("--no-log", action="store_true", help="Disable writing log files")
    p.add_argument("--verbose-llm", action="store_true", help="Print llama.cpp load diagnostics (shows GPU offload)")

    args = p.parse_args()
    cfg = build_config(args)

    try:
        pipeline = TranslationPipeline(cfg)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.mic:
        pipeline.run_microphone()
    else:
        pipeline.run_file(args.file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())