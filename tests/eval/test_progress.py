"""Tests for the console progress reporter and its auto-detection."""

import io
import sys

from traceroot.eval.progress import ConsoleProgress, can_animate, should_show_progress
from traceroot.eval.results import EvalItemResult, Score


def _item(case_id: str, value):
    return EvalItemResult(
        case_id=case_id,
        input=None,
        output=None,
        expected=None,
        scores=[Score("acc", value)] if value is not None else [],
        scorer_errors={},
        error="boom" if value == "error" else None,
        trace_id=None,
    )


def test_should_show_progress_explicit_wins(monkeypatch):
    # Explicit True/False overrides everything, TTY or not.
    monkeypatch.setenv("TRACEROOT_EVAL_PROGRESS", "0")
    assert should_show_progress(True) is True
    assert should_show_progress(False) is False


def test_should_show_progress_env_disable(monkeypatch):
    monkeypatch.setenv("TRACEROOT_EVAL_PROGRESS", "0")
    assert should_show_progress(None) is False


def test_should_show_progress_auto_non_tty(monkeypatch):
    monkeypatch.delenv("TRACEROOT_EVAL_PROGRESS", raising=False)

    class _NoTTY:
        def isatty(self):
            return False

    monkeypatch.setattr("sys.stderr", _NoTTY())
    assert should_show_progress(None) is False


def test_should_show_progress_auto_tty(monkeypatch):
    monkeypatch.delenv("TRACEROOT_EVAL_PROGRESS", raising=False)

    class _TTY:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stderr", _TTY())
    assert should_show_progress(None) is True


def test_progress_counts_and_renders():
    buf = io.StringIO()
    bar = ConsoleProgress(3, "demo", stream=buf, width=10, animate=True)
    bar.start()
    bar.on_case_complete(_item("a", 1.0), 5.0)  # passed
    bar.on_case_complete(_item("b", 0.0), 5.0)  # failed
    bar.on_case_complete(_item("c", "error"), 5.0)  # errored
    bar.finish()

    assert (bar.passed, bar.failed, bar.errored, bar.done) == (1, 1, 1, 3)
    out = buf.getvalue()
    assert "demo" in out
    assert "3/3" in out
    # "off" tail appears once a case fails/errors.
    assert "off" in out
    # finish() clears the line with CR + the ANSI erase-line escape.
    assert out.endswith("\r\x1b[2K")


def test_render_never_exceeds_terminal_width():
    # A long run name in a narrow terminal must not wrap: each animated frame has to fit on
    # one physical row, or \r\x1b[2K can't clear it and the overflow stacks (the VS Code
    # terminal bug). Force a 40-column terminal and a label wide enough to overflow.
    buf = io.StringIO()
    bar = ConsoleProgress(6, "ticket-routing-quality-v2", stream=buf, animate=True, cols=40)
    bar.start()
    for i in range(6):
        bar.on_case_complete(_item(f"c{i}", 1.0), 5.0)
    # Each frame is "\r\x1b[2K" + line; split on the control prefix and check visible width.
    frames = [f for f in buf.getvalue().split("\r\x1b[2K") if f]
    assert frames, "expected at least one rendered frame"
    for line in frames:
        assert len(line) <= 39, f"frame wider than terminal (would wrap): {line!r}"
    # The bar + counts (the actual progress) stay visible even when the label is ellipsized.
    assert "…" in frames[-1] and "6/6" in frames[-1] and "█" in frames[-1]


def test_render_full_width_when_it_fits():
    # A wide terminal keeps the full line (label + bar + stats), untruncated.
    buf = io.StringIO()
    bar = ConsoleProgress(3, "demo", stream=buf, width=10, animate=True, cols=200)
    bar.start()
    bar.on_case_complete(_item("a", 1.0), 5.0)
    out = buf.getvalue()
    assert "demo" in out and "1/3" in out and "/s" in out  # stats survive, nothing trimmed


def test_progress_finish_is_idempotent_without_start():
    buf = io.StringIO()
    bar = ConsoleProgress(0, "empty", stream=buf)
    bar.finish()  # never started -> no-op, no crash
    assert buf.getvalue() == ""


def test_progress_plain_mode_is_clean_newlines_no_carriage_return():
    buf = io.StringIO()
    bar = ConsoleProgress(3, "demo", stream=buf, animate=False)  # e.g. VS Code Debug Console
    bar.start()
    bar.on_case_complete(_item("a", 1.0), 5.0)
    bar.on_case_complete(_item("b", 0.0), 5.0)
    bar.on_case_complete(_item("c", "error"), 5.0)
    bar.finish()
    out = buf.getvalue()
    assert "\r" not in out and "\x1b" not in out  # no CR/ANSI -> cannot stack anywhere
    assert "3/3" in out
    assert out.count("\n") == 3  # one clean line per case (small run)


def test_can_animate_false_under_debugpy(monkeypatch):
    class _TTY:
        def isatty(self):
            return True

    monkeypatch.delenv("TERM", raising=False)
    assert can_animate(_TTY()) is True
    monkeypatch.setitem(sys.modules, "debugpy", object())  # VS Code Debug Console
    assert can_animate(_TTY()) is False
