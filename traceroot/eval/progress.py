"""A dependency-free console progress reporter for evaluation runs.

Renders a live, single-line progress bar to a stream (stderr by default) as
cases complete, then clears it. It is driven entirely by the engine's existing
``on_case_start`` / ``on_case_complete`` hooks, so it adds no coupling to the
run logic. Nothing here is uploaded — it is purely local presentation.

The engine turns this on automatically when stdout is an interactive terminal
and off when output is piped/redirected (CI, the runner, a subprocess), so
machine-readable channels stay clean. Callers can force it with
``evaluate(..., progress=True/False)``.
"""

from __future__ import annotations

import os
import re
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


def _char_width(cp: int) -> int:
    """Terminal columns for one code point: C0/C1 controls & zero-width/combining marks = 0,
    East Asian Wide/Fullwidth (incl. most emoji) = 2, else 1. Best-effort (no full Unicode DB),
    but enough to keep CJK/emoji/combining input from wrapping the single line."""
    if cp < 32 or 0x7F <= cp < 0xA0:  # C0 / C1 controls
        return 0
    if 0x0300 <= cp <= 0x036F or 0x200B <= cp <= 0x200F or cp == 0xFEFF or 0xFE00 <= cp <= 0xFE0F:
        return 0  # combining marks, zero-width, variation selectors
    if (
        0x1100 <= cp <= 0x115F
        or 0x2E80 <= cp <= 0xA4CF
        or 0xAC00 <= cp <= 0xD7A3
        or 0xF900 <= cp <= 0xFAFF
        or 0xFE30 <= cp <= 0xFE4F
        or 0xFF00 <= cp <= 0xFF60
        or 0xFFE0 <= cp <= 0xFFE6
        or 0x1F300 <= cp <= 0x1FAFF
        or 0x20000 <= cp <= 0x3FFFD
    ):
        return 2
    return 1


def _display_width(s: str) -> int:
    """Display width of a string in terminal columns."""
    return sum(_char_width(ord(ch)) for ch in s)


def _slice_to_width(s: str, max_cols: int) -> str:
    """Truncate to at most ``max_cols`` display columns."""
    if max_cols <= 0:
        return ""
    width = 0
    out: list[str] = []
    for ch in s:
        cw = _char_width(ord(ch))
        if width + cw > max_cols:
            break
        out.append(ch)
        width += cw
    return "".join(out)


_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")


def _sanitize_label(s: str) -> str:
    """Strip C0/C1 controls (newlines, ANSI escapes) so a run name can't break the single line."""
    return _CONTROL_RE.sub(" ", s).strip()


def _fit(label: str, anchor: str, stats: str, limit: int) -> str:
    """Compose one progress line that fits within ``limit`` columns without wrapping.

    Keeps the bar + counts (``anchor``) visible at all costs — a progress bar with no
    progress is useless. Shedding order as space runs out: full line -> drop ``stats`` ->
    ellipsize ``label`` -> (last resort) trim the anchor. Widths are display columns, so
    wide/zero-width characters can't sneak past the clamp. ``label`` sits before the anchor,
    ``stats`` after it.
    """
    full = f"  {label}{anchor}{stats}"
    if _display_width(full) <= limit:
        return full
    with_label = f"  {label}{anchor}"
    if _display_width(with_label) <= limit:  # dropping stats is enough
        return with_label
    # Ellipsize the label to make room for the anchor (2 leading spaces + label + anchor).
    room = limit - _display_width(anchor) - 2
    if room >= 1:
        lab = (
            label
            if _display_width(label) <= room
            else _slice_to_width(label, max(room - 1, 0)) + "…"
        )
        return f"  {lab}{anchor}"
    # Too narrow for even the bare anchor: keep the counts (drop the label) and trim the anchor
    # itself by display width so it still never wraps.
    return _slice_to_width(anchor, limit)


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
        self.label = _sanitize_label(label)
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

    def _safe_write(self, text: str) -> None:
        """Write+flush the frame, treating ANY output failure (e.g. a stderr consumer that closed
        early) as reporter shutdown. Progress is local presentation only and must never abort the
        evaluation it observes."""
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):  # broken pipe / closed or detached stream
            self._active = False

    def finish(self) -> None:
        """Persist the completed bar: redraw the final (100%) frame and end the line so it
        stays on screen, then subsequent output starts cleanly below it. No-op in plain mode
        (its last line already shows the final count and is newline-terminated)."""
        if not self._active:
            return
        if self._animate:
            self._safe_write("\r\x1b[2K" + self._frame() + "\n")
        self._active = False

    def _plain(self) -> None:
        """A clean newline-terminated progress line (no \\r/ANSI). Throttled to ~deciles
        for large runs so it never floods."""
        step = max(1, self.total // 10)
        if self.done == self.total or self.total <= 20 or self.done % step == 0:
            bad = self.failed + self.errored
            tail = f"  ({bad} off)" if bad else ""
            self._safe_write(f"  {self.label}  {self.done}/{self.total}{tail}\n")

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
        self._safe_write("\r\x1b[2K" + self._frame())
