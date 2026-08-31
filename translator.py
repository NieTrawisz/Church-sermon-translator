"""Translation using a local Qwen3.5 model via llama-cpp-python.

The important idea is CONTEXT. A sermon is one continuous talk: names, Scripture
references, and recurring terms should be translated consistently. So instead of
translating each line in isolation, we feed the model the last few
(source -> translation) pairs as prior conversation turns.

translate() now returns a TranslationResult carrying token counts and whether
the model "thought", so the pipeline can report where time goes.
"""

from __future__ import annotations

import os
import re
from collections import deque

from llama_cpp import Llama

from config import Config, language_name
from models import TranslationResult

# Strip Qwen "thinking" blocks and common label prefixes from the output.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_LABEL_RE = re.compile(r"^(translation|译文|翻译)\s*[:：]\s*", re.IGNORECASE)


class Translator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.llama_cfg = cfg.llama

        path = self.llama_cfg.model_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Qwen model not found at '{path}'.\n"
                "Download a Qwen3.5 GGUF (see README) and set --model / "
                "LlamaConfig.model_path to point at the .gguf file."
            )

        print(f"[llama] loading '{os.path.basename(path)}' ...")
        self.llm = Llama(
            model_path=path,
            n_ctx=self.llama_cfg.n_ctx,
            n_gpu_layers=self.llama_cfg.n_gpu_layers,
            verbose=False,
        )
        print("[llama] ready.")

        # Rolling memory of recent (source, translation) pairs.
        self.history: deque[tuple[str, str]] = deque(maxlen=self.llama_cfg.context_turns)

    # -- Prompt construction -------------------------------------------------

    def _system_prompt(self, source_language: str) -> str:
        no_think = "\n/no_think" if self.llama_cfg.disable_thinking else ""
        return (
            "You are a professional simultaneous interpreter for a live church "
            f"service. Translate the speaker's words from {source_language} into "
            f"{self.cfg.target_language}.\n\n"
            "Rules:\n"
            "- Output ONLY the translation. No notes, no explanations, no "
            "quotation marks, and never repeat the original text.\n"
            "- Preserve meaning, tone, and register. Sermons can be emotional, "
            "poetic, or scriptural: keep that feeling.\n"
            "- Keep names of people and places, and Scripture references (book, "
            "chapter, verse), accurate and consistent with the earlier lines.\n"
            "- Translate fragments and incomplete sentences as they are. Do not "
            "invent, complete, or summarize.\n"
            f"- If a line is already in {self.cfg.target_language}, return it as "
            "the closest natural rendering." + no_think
        )

    def _build_messages(self, text: str, source_language: str) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt(source_language)}
        ]
        for src, dst in self.history:
            messages.append({"role": "user", "content": src})
            messages.append({"role": "assistant", "content": dst})
        # Belt-and-suspenders: also put the no-think switch on the live turn,
        # since some chat templates only honor it on the last user message.
        line = text + ("\n/no_think" if self.llama_cfg.disable_thinking else "")
        messages.append({"role": "user", "content": line})
        return messages

    # -- Output inspection / cleanup ----------------------------------------

    @staticmethod
    def _has_thinking(raw: str) -> bool:
        """True only if the model emitted a <think> block with real content."""
        m = _THINK_RE.search(raw)
        if not m:
            return False
        inner = re.sub(r"</?think>", "", m.group(0), flags=re.IGNORECASE).strip()
        return bool(inner)

    @staticmethod
    def _clean(raw: str) -> str:
        out = _THINK_RE.sub("", raw).strip()
        out = _LABEL_RE.sub("", out).strip()
        if len(out) >= 2 and out[0] in "\"'“”«»" and out[-1] in "\"'“”«»":
            out = out[1:-1].strip()
        return out

    # -- Public API ----------------------------------------------------------

    def translate(self, text: str, source_language_code: str | None = None) -> TranslationResult:
        """Translate one line, using and updating the rolling context."""
        source_language = language_name(source_language_code)

        result = self.llm.create_chat_completion(
            messages=self._build_messages(text, source_language),
            temperature=self.llama_cfg.temperature,
            top_p=self.llama_cfg.top_p,
            max_tokens=self.llama_cfg.max_tokens,
        )
        raw = result["choices"][0]["message"]["content"]
        usage = result.get("usage") or {}
        translation = self._clean(raw)

        # Remember this pair so the next line stays consistent.
        self.history.append((text, translation))

        return TranslationResult(
            text=translation,
            raw=raw,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            thinking=self._has_thinking(raw),
        )
