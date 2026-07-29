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

    ``explicit`` wins when set. Otherwise auto-detect: on only when the bar's own
    stream (stderr) is an interactive terminal, and suppressible via
    ``TRACEROOT_EVAL_PROGRESS=0``. Gating on stderr (not stdout) means the bar shows
    even when stdout is piped, and is suppressed when stderr is captured (e.g. some
    IDE run panels / logs) so it can't stack.
    """
    if explicit is not None:
        return explicit
    if os.environ.get("TRACEROOT_EVAL_PROGRESS") == "0":
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def print_run_url(url: str, stream: TextIO | None = None) -> None:
    """Print a clickable run link on its own line (same stream as the bar)."""
    out = stream if stream is not None else sys.stderr
    out.write(f"  → {url}\n")
    out.flush()


def can_animate(stream: TextIO) -> bool:
    """Whether ``stream`` supports an in-place (``\\r``/ANSI) redraw.

    False for pipes, ``TERM=dumb``, and the VS Code Debug Console / Jupyter, which do
    NOT honor carriage returns even though ``isatty()`` is often spoofed to True there
    (so a ``\\r`` bar would stack). In those, the reporter falls back to plain newline
    progress instead of an animated single line.
    """
    try:
        if not stream.isatty():
            return False
    except Exception:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    # debugpy (VS Code "Run/Debug" → Debug Console) and ipykernel (Jupyter) don't process \r.
    return not ("debugpy" in sys.modules or "ipykernel" in sys.modules)


class ConsoleProgress:
    """Evaluation progress: an animated single-line bar where the terminal supports it,
    and clean (non-stacking) plain newline updates everywhere else."""

    def __init__(
        self,
        total: int,
        label: str,
        *,
        stream: TextIO | None = None,
        width: int = 24,
        animate: bool | None = None,
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
        self._active = False
        # Animate (in-place) only when the stream truly supports \r; else plain lines.
        self._animate = can_animate(self.stream) if animate is None else animate

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        self._t0 = time.monotonic()
        self._active = True
        if self._animate:
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
        if self._animate:
            self._render()
        else:
            self._plain()

    def finish(self) -> None:
        """Erase the animated bar so the caller's own output starts on a clean line.
        No-op in plain mode (its lines are already newline-terminated)."""
        if not self._active:
            return
        if self._animate:
            self.stream.write("\r\x1b[2K")  # CR + clear whole line
            self.stream.flush()
        self._active = False

    def _plain(self) -> None:
        """A clean newline-terminated progress line (no \\r/ANSI). Throttled to ~deciles
        for large runs so it never floods."""
        step = max(1, self.total // 10)
        if self.done == self.total or self.total <= 20 or self.done % step == 0:
            bad = self.failed + self.errored
            tail = f"  ({bad} off)" if bad else ""
            self.stream.write(f"  {self.label}  {self.done}/{self.total}{tail}\n")
            self.stream.flush()

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
        # \r returns to column 0; \x1b[2K erases the whole line -> a clean in-place
        # redraw regardless of the previous frame's length (no manual padding).
        self.stream.write("\r\x1b[2K" + line)
        self.stream.flush()
