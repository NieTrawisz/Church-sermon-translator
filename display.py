"""Readable terminal output with per-segment timing.

Each panel shows: recording position, total processing time, real-time factor
(RTF), then the source/translation, then a breakdown line: whisper vs llm time,
tokens + tokens/sec, overhead, and whether the model "thought".

Uses `rich` if available; falls back to plain print so nothing crashes if it's
not installed.
"""

from __future__ import annotations

from config import Config
from models import SegmentMetrics, format_timestamp
from config import language_name

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    _RICH = True
    _console = Console()
except ImportError:  # pragma: no cover
    _RICH = False
    _console = None


def _rtf_style(rtf: float) -> str:
    if rtf != rtf:            # NaN
        return "dim"
    if rtf < 1.0:
        return "green"        # faster than real time
    if rtf < 1.5:
        return "yellow"
    return "red"              # falling behind


def _rtf_text(rtf: float) -> str:
    return "RTF n/a" if rtf != rtf else f"RTF {rtf:.2f}"


class Display:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def header(self, mode: str) -> None:
        title = f"Sermon Translator  →  {self.cfg.target_language}   [{mode}]"
        if _RICH:
            _console.rule(f"[bold cyan]{title}")
        else:
            print("=" * 70)
            print(title)
            print("=" * 70)

    def show(self, m: SegmentMetrics) -> None:
        rec = format_timestamp(m.rec_start)
        lang = language_name(m.language)
        stats = (
            f"whisper {m.stt_s:.2f}s · llm {m.llm_s:.2f}s "
            f"({m.completion_tokens} tok, {m.tok_per_s:.0f} tok/s) · "
            f"overhead {m.overhead_s * 1000:.0f}ms"
        )

        if _RICH:
            title = (
                f"[cyan]{rec}[/]  ·  [bold]{m.total_s:.2f}s[/]  "
                f"[{_rtf_style(m.rtf)}]{_rtf_text(m.rtf)}[/]"
            )
            subtitle = (
                "[red]● think: yes[/]" if m.thinking else "[dim]think: no[/]"
            )
            body = Text()
            if self.cfg.show_original:
                body.append(f"{lang}: ", style="dim")
                body.append(m.text + "\n", style="dim italic")
            body.append(m.translation + "\n", style="bold white")
            body.append(stats, style="dim")
            _console.print(
                Panel(
                    body,
                    title=title,
                    title_align="left",
                    subtitle=subtitle,
                    subtitle_align="right",
                    border_style="grey37",
                    padding=(0, 1),
                )
            )
        else:
            think = "THINK: yes" if m.thinking else "think: no"
            print(f"\n[{rec}]  {m.total_s:.2f}s  {_rtf_text(m.rtf)}  ({think})")
            if self.cfg.show_original:
                print(f"  ({lang}) {m.text}")
            print(f"  → {m.translation}")
            print(f"  {stats}")

    def summary(self, text: str) -> None:
        if not text:
            return
        if _RICH:
            _console.print(Panel(text, title="[bold cyan]session summary",
                                 title_align="left", border_style="cyan"))
        else:
            print("\n--- session summary ---")
            print(text)

    def error(self, message: str) -> None:
        if _RICH:
            _console.print(f"[bold red]Error:[/] {message}")
        else:
            print(f"Error: {message}")
