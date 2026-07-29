"""A dependency-free console progress reporter for evaluation runs.

Renders a live, single-line progress bar to a stream (stderr by default) as
cases complete, then clears it. It is driven entirely by the engine's existing
``on_case_start`` / ``on_case_complete`` hooks, so it adds no coupling to the
run logic. Nothing here is uploaded — it is purely local presentation.

The engine turns this on automatically when stdout is an interactive terminal
and off when output is piped/redirected (CI, the CLI runner, a subprocess), so
machine-readable channels stay clean. Callers can force it with
``evaluate(..., progress=True/False)``.
"""

from __future__ import annotations

import os
import sys
import time
from typing import TextIO

from traceroot.eval.results import EvalItemResult, case_status

# Eighth-block glyphs for a smooth sub-cell bar edge.
_BLOCKS = " ▏▎▍▌▋▊▉█"


def should_show_progress(explicit: bool | None) -> bool:
    """Resolve the effective progress setting.

    ``explicit`` wins when set. Otherwise auto-detect: on only for an
    interactive stdout, and suppressible via ``TRACEROOT_EVAL_PROGRESS=0``.
    """
    if explicit is not None:
        return explicit
    if os.environ.get("TRACEROOT_EVAL_PROGRESS") == "0":
        return False
    stream = sys.stdout
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class ConsoleProgress:
    """A single-line, in-place progress bar for an evaluation run."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        stream: TextIO | None = None,
        width: int = 24,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self.done = 0
        self.passed = 0
        self.failed = 0
        self.errored = 0
        self._t0 = time.monotonic()
        self._last_len = 0
        self._active = False

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._t0 = time.monotonic()
        self._active = True
        self._render()

    def on_case_complete(self, item: EvalItemResult, _duration_ms: float) -> None:
        self.done += 1
        status = case_status(item)
        if status == "passed":
            self.passed += 1
        elif status == "failed":
            self.failed += 1
        elif status == "errored":
            self.errored += 1
        self._render()

    def finish(self) -> None:
        """Erase the bar so the caller's own output starts on a clean line."""
        if not self._active:
            return
        self.stream.write("\r" + " " * self._last_len + "\r")
        self.stream.flush()
        self._active = False

    # -- rendering -------------------------------------------------------
    def _bar(self, frac: float) -> str:
        frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
        filled = frac * self.width
        full = int(filled)
        bar = "█" * full
        if full < self.width:
            bar += _BLOCKS[int((filled - full) * 8)]
            bar += " " * (self.width - full - 1)
        return bar

    def _render(self) -> None:
        if not self._active:
            return
        total = self.total or 1
        frac = self.done / total
        elapsed = time.monotonic() - self._t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        mm, ss = divmod(int(elapsed), 60)
        tail = f"  {self.failed + self.errored} off" if (self.failed or self.errored) else ""
        line = (
            f"  {self.label}  ▕{self._bar(frac)}▏ {self.done}/{self.total}"
            f"  ·  {rate:.1f}/s  ·  {mm:d}:{ss:02d}{tail}"
        )
        pad = " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + line + pad)
        self.stream.flush()
        self._last_len = len(line)
