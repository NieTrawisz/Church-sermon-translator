# Sermon Translator

Real-time-ready translation pipeline for church services:

```
speech ──▶ faster-whisper ──▶ text ──▶ Qwen3.5 (llama.cpp) ──▶ translation ──▶ screen
                                          (keeps rolling context)
```

You start by pointing it at a **recorded video/audio file** to test the whole
chain. The same code path also runs on a **live microphone** for the real-time
target, so nothing has to be rebuilt later — you just switch `--file` for `--mic`.

## How it fits together

| File | Job |
|------|-----|
| `config.py` | All tunable settings + language-code → name mapping |
| `models.py` | The `TranscriptSegment` data type (shared, no heavy deps) |
| `transcriber.py` | faster-whisper: `transcribe_file()` and `stream_microphone()` |
| `translator.py` | Qwen3.5 via llama.cpp, with rolling sermon context |
| `display.py` | Readable terminal output (uses `rich`, falls back to plain print) |
| `pipeline.py` | Wires the three stages together |
| `main.py` | Command-line entry point |

The key design choice: transcription **yields segments**, so translation and
display start on segment #1 while whisper is still working on #2. That streaming
shape is what makes the eventual real-time mode work without a rewrite.

## Install

```bash
pip install -r requirements.txt
```

**GPU notes (important for real-time):**

- **faster-whisper** uses CTranslate2. Install NVIDIA CUDA + cuDNN and it will
  use the GPU automatically when you pass `--device cuda`.
- **llama-cpp-python must be built with CUDA** or it runs the LLM on CPU (slow):
  ```bash
  CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
  ```

## Get a Qwen3.5 model

Download a **GGUF** build (the format llama.cpp uses). Qwen3.5 supports 201
languages, so it handles most source→target pairs a service needs. Pick the size
to fit your GPU — you said you'll decide once you know the card:

| VRAM you have | Suggested Qwen3.5 GGUF | Notes |
|---------------|------------------------|-------|
| 6–8 GB | `Qwen3.5-4B` Q4_K_M | Fast, fine for clean speech |
| 10–12 GB | `Qwen3.5-9B` Q4_K_M | **Good default** — quality + speed |
| 16–24 GB | `Qwen3.5-9B` Q8_0, or `Qwen3.5-27B` Q4_K_M | Best quality for one card |
| 24 GB+ (or split) | `Qwen3.5-35B-A3B` | MoE: only 3B active params, so it's fast for its quality |

Remember whisper also needs VRAM (`large-v3` ≈ 3–4 GB, or use `large-v3-turbo`
/ `medium` — both multilingual — to leave more room for the LLM). Download example:

```bash
pip install huggingface_hub
huggingface-cli download <repo>/Qwen3.5-9B-GGUF <file>.gguf --local-dir models
```

## Run

Test with a recorded service (auto-detects the spoken language):

```bash
python main.py --file service.mp4 \
    --model models/Qwen3.5-9B-Q4_K_M.gguf \
    --target-lang English
```

Force the source language and translate into Polish:

```bash
python main.py --file sermon.wav --source-lang en --target-lang Polish \
    --model models/Qwen3.5-9B-Q4_K_M.gguf
```

Live microphone (the real-time target — needs `sounddevice` + `numpy`):

```bash
python main.py --mic --model models/Qwen3.5-9B-Q4_K_M.gguf --target-lang English
```

Useful flags: `--whisper-model turbo` (large-v3-turbo: ~4x faster, still multilingual), `--device cpu`,
`--n-gpu-layers 20` (partial GPU offload if VRAM is tight), `--no-original`
(show only the translation), `--context-turns 8` (more context memory).

## How context is kept

Each translated line is stored as a `(source, translation)` pair. The last few
pairs (default 6, via `--context-turns`) are replayed to the model as prior
conversation turns before the new line. That's what keeps names, places, and
Scripture references consistent across the whole service instead of drifting
line by line.

## Tuning for real time

The `--mic` path uses simple energy-based silence detection to decide when a
phrase has ended. To lower latency and sharpen phrase boundaries later:

- Drop `--beam_size` toward 1 and use `turbo` (large-v3-turbo) for the transcriber. Do NOT use `distil-*` models unless your audio is English — they are English-only.
- Swap the energy detector in `transcriber.stream_microphone()` for a real VAD
  (webrtcvad or Silero) — the yield contract stays the same, so nothing else
  changes.
- Keep the LLM temperature low (already 0.3) for stable, consistent wording.

## Notes / limits

- Segment-by-segment LLM translation can occasionally split a sentence awkwardly
  at a pause; the rolling context reduces this a lot but doesn't eliminate it.
- `Qwen3.5` can "think" before answering; that's disabled here (`/no_think` +
  output cleanup) so translations appear immediately.
- This prints to the terminal for testing. Sending the text to a projector,
  browser overlay, or captions display is a natural next step — the `Display`
  class is the single place to change.