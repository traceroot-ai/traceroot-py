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

from traceroot.eval.results import EvalItemResult, MainScore, case_status

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
    """Print a clickable run link on its own line, followed by a blank line to space it
    from the next evaluate()'s output (same stream as the bar)."""
    out = stream if stream is not None else sys.stderr
    out.write(f"  → {url}\n\n")
    out.flush()


def _term_cols(stream: TextIO) -> int:
    """Best-effort terminal width (columns) for ``stream``, default 80.

    The animated bar MUST fit on one physical row: if a frame is wider than the
    terminal it wraps, and ``\\r\\x1b[2K`` then only clears the last wrapped row —
    leaving the overflow behind on every frame (the "stacking" bug). We clamp the
    rendered line to this width so it never wraps.
    """
    try:
        cols = os.get_terminal_size(stream.fileno()).columns
        if cols > 0:
            return cols
    except Exception:
        pass
    try:
        cols = int(os.environ.get("COLUMNS", ""))
        if cols > 0:
            return cols
    except (TypeError, ValueError):
        pass
    return 80


def _fit(label: str, anchor: str, stats: str, limit: int) -> str:
    """Compose one progress line that fits within ``limit`` columns without wrapping.

    Keeps the bar + counts (``anchor``) visible at all costs — a progress bar with no
    progress is useless. Shedding order as space runs out: full line -> drop ``stats`` ->
    ellipsize ``label`` -> (last resort) hard-trim. ``label`` sits before the anchor,
    ``stats`` after it.
    """
    full = f"  {label}{anchor}{stats}"
    if len(full) <= limit:
        return full
    with_label = f"  {label}{anchor}"
    if len(with_label) <= limit:  # dropping stats is enough
        return with_label
    # Ellipsize the label to make room for the anchor (2 leading spaces + label + anchor).
    room = limit - len(anchor) - 2
    if room >= 1:
        lab = label if len(label) <= room else label[: max(room - 1, 0)] + "…"
        return f"  {lab}{anchor}"
    # Terminal too narrow for even the bare anchor: hard-trim so we still never wrap.
    return with_label[:limit]


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
        cols: int | None = None,
        main_score: MainScore | None = None,
    ) -> None:
        self.total = max(int(total), 0)
        self.label = label
        # The run's resolved scoring policy (threshold + direction), so live pass/fail matches
        # the final result. None -> the default policy (single-scorer default).
        self._main_score = main_score
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        # Terminal width to clamp each frame to (auto-detected when None).
        self._cols = cols
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
        status = case_status(item, self._main_score)
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
        """Persist the completed bar: redraw the final (100%) frame and end the line so it
        stays on screen, then subsequent output starts cleanly below it. No-op in plain mode
        (its last line already shows the final count and is newline-terminated)."""
        if not self._active:
            return
        if self._animate:
            self.stream.write("\r\x1b[2K" + self._frame() + "\n")
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

    def _frame(self) -> str:
        """The current progress line, clamped to one physical terminal row (no wrap)."""
        total = self.total or 1
        frac = self.done / total
        elapsed = time.monotonic() - self._t0
        rate = self.done / elapsed if elapsed > 0 else 0.0
        mm, ss = divmod(int(elapsed), 60)
        tail = f"  {self.failed + self.errored} off" if (self.failed or self.errored) else ""
        # Clamp to one physical row: a line wider than the terminal wraps, and then
        # \r\x1b[2K only clears the last wrapped row -> the overflow stacks. Trim to cols-1
        # (leave the last column free so an exactly-full line can't auto-wrap).
        cols = self._cols if self._cols is not None else _term_cols(self.stream)
        limit = max(cols - 1, 0)
        anchor = f"  ▕{self._bar(frac)}▏ {self.done}/{self.total}"  # bar + counts (kept)
        stats = f"  ·  {rate:.1f}/s  ·  {mm:d}:{ss:02d}{tail}"  # dropped first when tight
        return _fit(self.label, anchor, stats, limit)

    def _render(self) -> None:
        if not self._active:
            return
        # \r returns to column 0; \x1b[2K erases the whole line -> a clean in-place
        # redraw regardless of the previous frame's length (no manual padding).
        self.stream.write("\r\x1b[2K" + self._frame())
        self.stream.flush()
