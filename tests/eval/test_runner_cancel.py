"""Ctrl+C while a finished case's result is still waiting for TraceRoot's reply.

A result upload is a blocking call on a worker thread. Cancelling the run can't stop it, and
TraceRoot may already have saved the result when only its reply is late. A cancelled run keeps
that case, reported once and counted as unconfirmed when no reply came, and writes its run file
within traceroot-cli's 10 s grace however long the reply takes.

These tests interrupt a run with SIGINT, as the terminal does, while a fake TraceRoot holds the
reply to one saved result.
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
from traceroot.eval import Dataset, Evaluation
from traceroot.eval.runner import Emitter, _CancelledError, _run_one
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
    ``failing`` is rejected instead."""

    def __init__(self, held: str | None = None, failing: str | None = None) -> None:
        super().__init__()
        self.held = held
        self.failing = failing
        self.held_saved = threading.Event()
        self.reply = threading.Event()
        self.replied = threading.Event()

    def record_item_result(self, run, item_result):
        if item_result.case_id == self.failing:
            raise RuntimeError("HTTP 400: rejected")
        super().record_item_result(run, item_result)
        if item_result.case_id == self.held:
            self.held_saved.set()
            self.reply.wait(_HOLD_S)
            self.replied.set()

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


@pytest.mark.skipif(sys.platform == "win32", reason="signal.pthread_kill is POSIX")
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


class _FakeTraceRoot(http.server.ThreadingHTTPServer):
    """Just the endpoints an uploaded run calls. Saves each case result as it arrives, but holds
    its reply to ``held``'s for ``hold_s``."""

    def __init__(self, held: str, hold_s: float) -> None:
        self.held = held
        self.hold_s = hold_s
        self.results: list[str] = []
        self.completed_as: str | None = None
        self.held_saved = threading.Event()
        self.release = threading.Event()
        super().__init__(("127.0.0.1", 0), _FakeTraceRootHandler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def handle_error(self, request, client_address):
        pass  # the runner exits without reading the held reply: a broken pipe is expected


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


@pytest.mark.skipif(sys.platform == "win32", reason="process groups and SIGINT are POSIX")
@pytest.mark.parametrize("hold_s", [8.0, _HOLD_S], ids=["held-8s", "held-to-timeout"])
def test_ctrl_c_keeps_a_result_traceroot_saved_but_had_not_acknowledged(tmp_path, hold_s):
    host = _FakeTraceRoot(held="slow", hold_s=hold_s)
    threading.Thread(target=host.serve_forever, daemon=True).start()
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "held_reply_eval.py").write_text(_EVAL_SRC)
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
    killed = False
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
            while not (
                host.held_saved.is_set()
                and len([e for e in _read_events(event_file) if e["type"] == "case_completed"]) == 3
            ):
                assert child.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, log_path.read_text()
                time.sleep(0.05)

            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(_CLI_GRACE_S)
            except subprocess.TimeoutExpired:
                killed = True
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            host.release.set()
            host.shutdown()
            host.server_close()

    log = log_path.read_text()
    assert not killed, f"the runner was still running {_CLI_GRACE_S}s after SIGINT\n{log}"
    assert child.returncode == 130, log

    events = _read_events(event_file)
    assert events[-1] == {"type": "suite_completed", "evaluations": 1, "cancelled": True}
    done = next(e for e in events if e["type"] == "evaluation_completed")
    assert done["status"] == "incomplete"
    assert done["counts"]["cases"] == 4
    assert [e["case_id"] for e in events if e["type"] == "case_completed"].count("slow") == 1

    run_doc, case_ids = _saved(events)
    assert run_doc["status"] == "incomplete"
    assert sorted(c["case_id"] for c in run_doc["cases"]) == ["a", "b", "c", "slow"]
    assert case_ids == ["a", "b", "c", "slow"]
    assert run_doc["upload"]["unconfirmed_result_count"] == 1
    assert run_doc["upload"]["failed_result_count"] == 0
    # TraceRoot holds the same four, and the run was closed incomplete.
    assert sorted(host.results) == case_ids
    assert host.completed_as == "incomplete"
