"""Ctrl+C while TraceRoot is slow to answer a cancelled run.

A result upload is a blocking call on a worker thread. Cancelling the run can't stop it, and
TraceRoot may already have saved the result when only its reply is late. A cancelled run keeps
that case, reported once and counted as unconfirmed when no reply came.

Closing the run on TraceRoot is the last step before a cancelled run is saved. It gets what is left
of ``_CANCEL_SAVE_S`` after the cancellation; a close TraceRoot hasn't acknowledged by then is left
behind, and the run is saved with ``close_confirmed`` false. Either way the run file is written
within traceroot-cli's 10 s grace.

These tests interrupt a run with SIGINT, as the terminal does, while a fake TraceRoot holds the
reply to one saved result or to closing the run. SIGTERM and SIGHUP, which traceroot-cli forwards
too, stop a run the same way.
"""

import asyncio
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

import traceroot
from traceroot.eval import Dataset, Evaluation, engine
from traceroot.eval.engine import _CANCEL_SAVE_S
from traceroot.eval.runner import (
    Emitter,
    _CancelledError,
    _partial_result,
    _run_one,
    _stop_signals_as_ctrl_c,
)
from traceroot.eval.transport import FakeTransport

# traceroot-cli sends SIGTERM to the runner this long after Ctrl+C, and SIGKILL 2 s later.
_CLI_GRACE_S = 10.0
# The result POST's socket timeout: the longest TraceRoot can hold a reply.
_HOLD_S = 30.0


@pytest.fixture(autouse=True)
def _isolate():
    yield
    try:
        traceroot.shutdown()
    except Exception:
        pass
    traceroot._client = None


class _HeldReply(FakeTransport):
    """A TraceRoot that saves each result as it arrives (``calls``, as FakeTransport records them)
    but holds its reply to ``held``'s until ``reply`` is set, for up to ``_HOLD_S``. A result for
    ``failing`` is rejected instead.

    ``close`` is how it answers closing the run: at once (None), ``"silent"`` (it takes the close
    but replies only once ``close_reply`` is set, for up to ``_HOLD_S``) or ``"failing"`` (it
    rejects the close)."""

    def __init__(
        self, held: str | None = None, failing: str | None = None, close: str | None = None
    ) -> None:
        super().__init__()
        self.held = held
        self.failing = failing
        self.close = close
        self.held_saved = threading.Event()
        self.reply = threading.Event()
        self.replied = threading.Event()
        self.close_reply = threading.Event()
        self.close_replied = threading.Event()

    def record_item_result(self, run, item_result):
        if item_result.case_id == self.failing:
            raise RuntimeError("HTTP 400: rejected")
        super().record_item_result(run, item_result)
        if item_result.case_id == self.held:
            self.held_saved.set()
            self.reply.wait(_HOLD_S)
            self.replied.set()

    def finish_run(self, run, status=None, emitted_metrics=None):
        state = super().finish_run(run, status, emitted_metrics)
        if self.close == "failing":
            raise RuntimeError("HTTP 503: unavailable")
        if self.close == "silent":
            self.close_reply.wait(_HOLD_S)
            self.close_replied.set()
        return state

    def saved(self) -> list[str]:
        return sorted(c[1] for c in self.calls if c[0] == "record_item_result")


async def _task(x):
    if x == "stalled":
        await asyncio.sleep(60)  # still computing when Ctrl+C arrives
    return x


def _ok(ctx):
    return 1.0


def _evaluation(transport: FakeTransport, case_ids: list[str]) -> Evaluation:
    ds = Dataset("cancel")
    for case_id in case_ids:
        ds.add(input=case_id, id=case_id, expected=case_id)
    return Evaluation(
        name="cancel",
        dataset=ds,
        task=_task,
        scorers=[_ok],
        report_to=transport,
        max_concurrency=len(case_ids),
    )


def _saved(events: list[dict]) -> tuple[dict, list[str]]:
    """The run file and the case ids in the cases file the run's terminal event points at."""
    done = next(e for e in events if e["type"] == "evaluation_completed")
    run_doc = json.loads(Path(done["artifact"]["run"]).read_text())
    lines = Path(done["artifact"]["cases"]).read_text().splitlines()
    return run_doc, sorted(json.loads(line)["case_id"] for line in lines)


def _completed(events: list[dict]) -> list[str]:
    return [e["case_id"] for e in events if e["type"] == "case_completed"]


def _started(events: list[dict], case_id: str) -> bool:
    return any(e["type"] == "case_started" and e["case_id"] == case_id for e in events)


def _ctrl_c_when(ready: Callable[[], bool], then: Callable[[], None] | None = None) -> dict:
    """Send SIGINT to the main thread, where asyncio.run handles it, once ``ready()`` holds; then
    call ``then``. Returns a dict that gets ``at``, the monotonic time it was sent."""
    sent: dict[str, float] = {}
    main = threading.main_thread().ident

    def interrupt() -> None:
        deadline = time.monotonic() + 30
        while not ready():
            if time.monotonic() > deadline:
                return  # never sent: the run's own assertions report what was missing
            time.sleep(0.01)
        sent["at"] = time.monotonic()
        signal.pthread_kill(main, signal.SIGINT)
        if then is not None:
            then()

    threading.Thread(target=interrupt, daemon=True).start()
    return sent


def _interrupted_run(transport, case_ids, ready, tmp_path, then=None):
    """Run the evaluation through the runner, Ctrl+C it once ``ready(events)`` holds, and return
    (events, run file, cases file case ids, seconds from SIGINT until the files were written)."""
    # asyncio.run turns SIGINT into a cancellation of the run only over Python's default handler.
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    events: list[dict] = []
    emitter = Emitter(lambda line: events.append(json.loads(line)))
    sent = _ctrl_c_when(lambda: ready(events), then)
    out = tmp_path / "runs"
    with pytest.raises(_CancelledError):
        _run_one(
            _evaluation(transport, case_ids),
            {"reporting": True, "out_dir": str(out)},
            emitter,
        )
    written_after = time.monotonic() - sent["at"]
    run_doc, case_ids_saved = _saved(events)
    return events, run_doc, case_ids_saved, written_after


class TestHeldReply:
    def test_a_saved_result_whose_reply_is_held_is_kept_once_and_counted_unconfirmed(
        self, tmp_path, monkeypatch
    ):
        thread_errors = []
        monkeypatch.setattr(threading, "excepthook", thread_errors.append)
        t = _HeldReply(held="slow")
        try:
            events, run_doc, case_ids, written_after = _interrupted_run(
                t,
                ["a", "b", "c", "slow", "stalled"],
                lambda events: (
                    t.held_saved.is_set()
                    and len(_completed(events)) == 3
                    and _started(events, "stalled")
                ),
                tmp_path,
            )
        finally:
            t.reply.set()  # TraceRoot replies only after the run was saved

        # Saved well within the CLI's grace, although the reply is held for up to _HOLD_S.
        assert written_after < _CLI_GRACE_S - 2, f"run file written {written_after:.1f}s after"

        # The case whose result TraceRoot saved is kept and reported once; the one still
        # computing is left out and never uploaded.
        assert sorted(_completed(events)) == ["a", "b", "c", "slow"]
        assert case_ids == ["a", "b", "c", "slow"]
        assert sorted(c["case_id"] for c in run_doc["cases"]) == ["a", "b", "c", "slow"]
        assert run_doc["status"] == "incomplete"
        assert run_doc["is_final"] is False
        assert run_doc["upload"]["unconfirmed_result_count"] == 1
        assert run_doc["upload"]["failed_result_count"] == 0
        assert run_doc["upload"]["close_confirmed"] is True
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["status"] == "incomplete"
        assert done["counts"]["cases"] == 4

        # TraceRoot holds the same cases, and the run was closed incomplete.
        assert t.saved() == case_ids
        assert ("finish_run", "incomplete") in t.calls

        # The late reply, after the run was saved, reports nothing a second time.
        assert t.replied.wait(5)
        time.sleep(0.2)  # let the upload thread finish with it
        assert sorted(_completed(events)) == ["a", "b", "c", "slow"]
        assert thread_errors == []

    def test_a_reply_that_comes_while_the_run_waits_is_confirmed(self, tmp_path):
        t = _HeldReply(held="slow")

        def reply_soon() -> None:
            time.sleep(0.5)
            t.reply.set()

        events, run_doc, case_ids, written_after = _interrupted_run(
            t,
            ["a", "b", "slow", "stalled"],
            lambda events: (
                t.held_saved.is_set()
                and len(_completed(events)) == 2
                and _started(events, "stalled")
            ),
            tmp_path,
            then=reply_soon,
        )
        # The run waited for the reply, not for a bound it no longer needed.
        assert written_after < 3, f"run file written {written_after:.1f}s after"
        assert sorted(_completed(events)) == ["a", "b", "slow"]
        assert case_ids == ["a", "b", "slow"]
        assert run_doc["upload"]["unconfirmed_result_count"] == 0
        assert run_doc["upload"]["failed_result_count"] == 0
        assert run_doc["upload"]["close_confirmed"] is True
        assert t.saved() == case_ids

    def test_an_interrupted_run_keeps_its_failed_result_count(self, tmp_path):
        t = _HeldReply(failing="bad")
        _events, run_doc, case_ids, _ = _interrupted_run(
            t,
            ["a", "bad", "stalled"],
            lambda events: len(_completed(events)) == 2 and _started(events, "stalled"),
            tmp_path,
        )
        # The case is kept, as on a completed run; its upload failure is counted.
        assert case_ids == ["a", "bad"]
        assert t.saved() == ["a"]
        assert run_doc["upload"]["failed_result_count"] == 1
        assert run_doc["upload"]["unconfirmed_result_count"] == 0
        assert run_doc["upload"]["close_confirmed"] is True


@pytest.mark.skipif(sys.platform == "win32", reason="signal.pthread_kill is POSIX")
class TestSlowClose:
    """Closing a cancelled run gets what is left of ``_CANCEL_SAVE_S`` after the cancellation.
    Shortened here; the end-to-end tests below use the real bounds."""

    def test_a_close_traceroot_never_answers_is_left_behind_and_the_run_saved(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(engine, "_CANCEL_SAVE_S", 1.0)
        thread_errors = []
        monkeypatch.setattr(threading, "excepthook", thread_errors.append)
        t = _HeldReply(close="silent")
        try:
            events, run_doc, case_ids, written_after = _interrupted_run(
                t,
                ["a", "b", "stalled"],
                lambda events: len(_completed(events)) == 2 and _started(events, "stalled"),
                tmp_path,
            )
        finally:
            t.close_reply.set()  # TraceRoot answers only after the run was saved

        # The close had the whole budget, and the run didn't wait past it.
        assert 0.9 <= written_after < 1.5, f"run file written {written_after:.1f}s after"
        assert case_ids == ["a", "b"]
        assert sorted(_completed(events)) == ["a", "b"]
        assert run_doc["status"] == "incomplete"
        assert run_doc["upload"]["close_confirmed"] is False
        assert ("finish_run", "incomplete") in t.calls

        # The late answer, after the run was saved, changes nothing and raises nowhere.
        assert t.close_replied.wait(5)
        time.sleep(0.2)  # let the close thread finish with it
        assert thread_errors == []

    def test_the_close_gets_only_what_the_drain_left_of_the_budget(self, tmp_path, monkeypatch):
        # Nothing else is running, so the run sees its cancellation only once the held result's
        # drain ends. The budget still counts from the cancellation.
        monkeypatch.setattr(engine, "_CANCEL_DRAIN_S", 1.5)
        monkeypatch.setattr(engine, "_CANCEL_SAVE_S", 3.0)
        t = _HeldReply(held="slow", close="silent")
        try:
            _events, run_doc, case_ids, written_after = _interrupted_run(
                t,
                ["a", "b", "slow"],
                lambda events: t.held_saved.is_set() and len(_completed(events)) == 2,
                tmp_path,
            )
        finally:
            t.reply.set()
            t.close_reply.set()

        assert 2.5 <= written_after < 3.75, f"run file written {written_after:.1f}s after"
        assert case_ids == ["a", "b", "slow"]
        assert run_doc["upload"]["unconfirmed_result_count"] == 1
        assert run_doc["upload"]["close_confirmed"] is False

    def test_a_close_traceroot_rejects_is_recorded_with_the_counts(self, tmp_path):
        t = _HeldReply(failing="bad", close="failing")
        _events, run_doc, case_ids, _ = _interrupted_run(
            t,
            ["a", "bad", "stalled"],
            lambda events: len(_completed(events)) == 2 and _started(events, "stalled"),
            tmp_path,
        )
        assert case_ids == ["a", "bad"]
        assert run_doc["status"] == "incomplete"
        assert run_doc["upload"]["close_confirmed"] is False
        assert run_doc["upload"]["failed_result_count"] == 1
        assert ("finish_run", "incomplete") in t.calls


class TestRunSavedBeforeItWasFinalized:
    """A run saved without the engine having reported how it was finalized (stopped again while
    it was being closed, say) can't claim that TraceRoot closed it."""

    def test_a_registered_run_is_saved_with_an_unconfirmed_close(self):
        evaluation = _evaluation(FakeTransport(), ["a"])
        partial = _partial_result(evaluation, [], None, "local_1", "run_1")
        assert partial.upload_state.close_confirmed is False

    def test_a_run_that_never_registered_has_nothing_to_close(self):
        evaluation = _evaluation(FakeTransport(), ["a"])
        assert _partial_result(evaluation, [], None).upload_state.close_confirmed is True


@pytest.mark.skipif(sys.platform == "win32", reason="SIGHUP and raise_signal semantics are POSIX")
class TestStopSignals:
    """SIGTERM and SIGHUP are handled exactly as SIGINT is at that moment."""

    @pytest.fixture(autouse=True)
    def _restore_handlers(self):
        saved = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        yield
        for signum, handler in saved.items():
            signal.signal(signum, handler)

    def test_they_reach_the_sigint_handler_and_are_restored_after(self):
        seen = []
        signal.signal(signal.SIGINT, lambda signum, frame: seen.append(signum))
        with _stop_signals_as_ctrl_c():
            signal.raise_signal(signal.SIGTERM)
            signal.raise_signal(signal.SIGHUP)
        assert seen == [signal.SIGINT, signal.SIGINT]
        assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL

    def test_the_first_cancels_a_run_and_the_next_is_a_second_ctrl_c(self):
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        second: list[type] = []

        async def run():
            signal.raise_signal(signal.SIGTERM)
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                try:
                    signal.raise_signal(signal.SIGHUP)
                except KeyboardInterrupt as exc:
                    second.append(type(exc))
                raise

        with _stop_signals_as_ctrl_c(), pytest.raises(KeyboardInterrupt):
            asyncio.run(run())
        assert second == [KeyboardInterrupt]

    def test_outside_a_run_they_raise_keyboard_interrupt(self):
        with _stop_signals_as_ctrl_c(), pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)

    def test_nothing_is_installed_when_sigint_is_ignored(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        with _stop_signals_as_ctrl_c():
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_DFL

    def test_an_ignored_signal_stays_ignored(self):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        with _stop_signals_as_ctrl_c():
            assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN
            assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
        assert signal.getsignal(signal.SIGHUP) is signal.SIG_IGN

    def test_nothing_is_installed_off_the_main_thread(self):
        seen = []

        def enter() -> None:
            with _stop_signals_as_ctrl_c():
                seen.append(signal.getsignal(signal.SIGTERM))

        thread = threading.Thread(target=enter)
        thread.start()
        thread.join()
        assert seen == [signal.SIG_DFL]


# ---------------------------------------------------------------------------
# The runner as traceroot-cli drives it: a process group of its own, SIGINT to the group, and the
# group killed if it is still alive when the CLI's grace runs out.
# ---------------------------------------------------------------------------
_EVAL_SRC = """
from traceroot import Dataset, Evaluation

ds = Dataset("held-reply")
for case_id in ("a", "b", "c", "slow"):
    ds.add(input=case_id, id=case_id, expected=case_id)

def task(x):
    return x

def scored(ctx):
    return 1.0

held_reply = Evaluation(
    name="held-reply",
    dataset=ds,
    task=task,
    scorers=[scored],
    dataset_id="ds_held",
    max_concurrency=4,
)
"""

# Three cases finish and one is still computing when the run is stopped.
_STALLED_EVAL_SRC = """
import asyncio

from traceroot import Dataset, Evaluation

ds = Dataset("stalled")
for case_id in ("a", "b", "c", "stalled"):
    ds.add(input=case_id, id=case_id, expected=case_id)

async def task(x):
    if x == "stalled":
        await asyncio.sleep(60)
    return x

def scored(ctx):
    return 1.0

stalled = Evaluation(
    name="stalled",
    dataset=ds,
    task=task,
    scorers=[scored],
    dataset_id="ds_stalled",
    max_concurrency=4,
)
"""

# How long the fake TraceRoot holds a reply it never sends: past the end of any test.
_NEVER_S = 120.0


class _FakeTraceRoot(http.server.ThreadingHTTPServer):
    """Just the endpoints an uploaded run calls. Saves each case result as it arrives, but holds
    its reply to ``held``'s for ``hold_s``. With ``silent_close`` it takes the request closing the
    run and never answers it. Serves while used as a context manager."""

    def __init__(
        self, held: str | None = None, hold_s: float = 0.0, silent_close: bool = False
    ) -> None:
        self.held = held
        self.hold_s = hold_s
        self.silent_close = silent_close
        self.results: list[str] = []
        self.completed_as: str | None = None
        self.held_saved = threading.Event()
        self.release = threading.Event()
        super().__init__(("127.0.0.1", 0), _FakeTraceRootHandler)

    def __enter__(self):
        threading.Thread(target=self.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.release.set()
        self.shutdown()
        self.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request, client_address):
        pass  # the runner exits without reading a held reply: a broken pipe is expected


class _FakeTraceRootHandler(http.server.BaseHTTPRequestHandler):
    server: _FakeTraceRoot

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/api/v1/public/evaluation-runs":
            return self._reply({"evaluation_run_id": "run_1"})
        if self.path.endswith("/results"):
            case_id = json.loads(body)["test_case_id"]
            self.server.results.append(case_id)
            if case_id == self.server.held:
                self.server.held_saved.set()
                self.server.release.wait(self.server.hold_s)
            return self._reply({})
        if self.path.endswith("/complete"):
            self.server.completed_as = json.loads(body)["status"]
            if self.server.silent_close:
                self.server.release.wait(_NEVER_S)
                return None  # the runner is gone by now
        return self._reply({})  # completion, span export

    def _reply(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _read_events(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _stopped_runner(tmp_path, host, src, ready, signum=signal.SIGINT):
    """Run the evaluation in ``src`` through the runner as traceroot-cli does for an uploaded run:
    in a process group of its own, ``signum`` sent to the group once ``ready(events)`` holds, and
    the group killed if it is still running ``_CLI_GRACE_S`` later.

    Returns (exit status, or None when it had to be killed; events; runner output; wall-clock time
    the signal was sent; seconds from the signal until the runner exited)."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "stopped_eval.py").write_text(src)
    event_file = tmp_path / "events.ndjson"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TRACEROOT_", "OTEL_"))}
    # What the CLI sets for an uploaded run.
    env.update(
        TRACEROOT_API_KEY="tr-test-key",
        TRACEROOT_HOST_URL=host.url,
        TRACEROOT_EVAL_EVENT_FILE=str(event_file),
        TRACEROOT_EVAL_OPTIONS=json.dumps({"reporting": True, "out_dir": str(tmp_path / "runs")}),
    )
    log_path = tmp_path / "runner.log"
    status: int | None = None
    with open(log_path, "w") as log_file:
        child = subprocess.Popen(
            [sys.executable, "-m", "traceroot.eval.runner", str(evals)],
            env=env,
            cwd=tmp_path,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # a process group of its own, as the CLI spawns it
        )
        try:
            deadline = time.monotonic() + 60
            while not ready(_read_events(event_file)):
                assert child.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.05)

            sent_at = time.time()
            sent = time.monotonic()
            os.killpg(child.pid, signum)
            try:
                status = child.wait(_CLI_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
            exited_after = time.monotonic() - sent
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return status, _read_events(event_file), log_path.read_text(), sent_at, exited_after


def _written_at(events: list[dict]) -> float:
    """Wall-clock time the run's terminal event's run file and cases file were last written."""
    done = next(e for e in events if e["type"] == "evaluation_completed")
    return max(Path(done["artifact"][k]).stat().st_mtime for k in ("run", "cases"))


@pytest.mark.skipif(sys.platform == "win32", reason="process groups and SIGINT are POSIX")
@pytest.mark.parametrize("hold_s", [8.0, _HOLD_S], ids=["held-8s", "held-to-timeout"])
def test_ctrl_c_keeps_a_result_traceroot_saved_but_had_not_acknowledged(tmp_path, hold_s):
    with _FakeTraceRoot(held="slow", hold_s=hold_s) as host:
        status, events, log, _, _ = _stopped_runner(
            tmp_path,
            host,
            _EVAL_SRC,
            lambda events: host.held_saved.is_set() and len(_completed(events)) == 3,
        )

    assert status is not None, f"the runner was still running {_CLI_GRACE_S}s after SIGINT\n{log}"
    assert status == 130, log

    assert events[-1] == {"type": "suite_completed", "evaluations": 1, "cancelled": True}
    done = next(e for e in events if e["type"] == "evaluation_completed")
    assert done["status"] == "incomplete"
    assert done["counts"]["cases"] == 4
    assert _completed(events).count("slow") == 1

    run_doc, case_ids = _saved(events)
    assert run_doc["status"] == "incomplete"
    assert sorted(c["case_id"] for c in run_doc["cases"]) == ["a", "b", "c", "slow"]
    assert case_ids == ["a", "b", "c", "slow"]
    assert run_doc["upload"]["unconfirmed_result_count"] == 1
    assert run_doc["upload"]["failed_result_count"] == 0
    assert run_doc["upload"]["close_confirmed"] is True
    # TraceRoot holds the same four, and the run was closed incomplete.
    assert sorted(host.results) == case_ids
    assert host.completed_as == "incomplete"


@pytest.mark.skipif(sys.platform == "win32", reason="process groups and SIGINT are POSIX")
@pytest.mark.parametrize("held", [None, "slow"], ids=["nothing-uploading", "after-a-held-result"])
def test_ctrl_c_saves_the_run_when_traceroot_never_answers_closing_it(tmp_path, held):
    with _FakeTraceRoot(held=held, hold_s=_NEVER_S, silent_close=True) as host:
        if held is None:
            src, expected = _STALLED_EVAL_SRC, ["a", "b", "c"]

            def ready(events):
                return len(_completed(events)) == 3 and _started(events, "stalled")

        else:
            # Nothing else is running, so the run sees its cancellation only once the held
            # result's drain ends; the close still gets only what is left of the budget.
            src, expected = _EVAL_SRC, ["a", "b", "c", "slow"]

            def ready(events):
                return host.held_saved.is_set() and len(_completed(events)) == 3

        status, events, log, sent_at, exited_after = _stopped_runner(tmp_path, host, src, ready)

    assert status is not None, f"the runner was still running {_CLI_GRACE_S}s after SIGINT\n{log}"
    assert status == 130, log
    # Saved once the budget ran out, not before and not after, and exited right after that.
    written_after = _written_at(events) - sent_at
    assert _CANCEL_SAVE_S - 1 <= written_after < _CANCEL_SAVE_S + 0.5, f"{written_after:.1f}s"
    assert exited_after < _CANCEL_SAVE_S + 1.5, f"exited {exited_after:.1f}s after SIGINT\n{log}"

    assert events[-1] == {"type": "suite_completed", "evaluations": 1, "cancelled": True}
    run_doc, case_ids = _saved(events)
    assert run_doc["status"] == "incomplete"
    assert case_ids == expected
    assert run_doc["upload"]["close_confirmed"] is False
    assert run_doc["upload"]["unconfirmed_result_count"] == (0 if held is None else 1)
    # The close was sent, as incomplete; TraceRoot just never answered it.
    assert host.completed_as == "incomplete"


@pytest.mark.skipif(sys.platform == "win32", reason="process groups, SIGTERM and SIGHUP are POSIX")
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_sigterm_and_sighup_stop_a_run_as_ctrl_c_does(tmp_path, signame):
    # The same run as the Ctrl+C test above: one saved result's reply comes 8 s later, past the
    # drain, so the run is saved without waiting for it.
    with _FakeTraceRoot(held="slow", hold_s=8.0) as host:
        status, events, log, _, _ = _stopped_runner(
            tmp_path,
            host,
            _EVAL_SRC,
            lambda events: host.held_saved.is_set() and len(_completed(events)) == 3,
            signum=getattr(signal, signame),
        )

    assert status is not None, f"the runner was still running {_CLI_GRACE_S}s after {signame}"
    assert status == 130, log

    assert events[-1] == {"type": "suite_completed", "evaluations": 1, "cancelled": True}
    run_doc, case_ids = _saved(events)
    assert run_doc["status"] == "incomplete"
    assert run_doc["is_final"] is False
    assert case_ids == ["a", "b", "c", "slow"]
    assert run_doc["upload"]["unconfirmed_result_count"] == 1
    assert run_doc["upload"]["close_confirmed"] is True
    assert sorted(host.results) == case_ids
    assert host.completed_as == "incomplete"
