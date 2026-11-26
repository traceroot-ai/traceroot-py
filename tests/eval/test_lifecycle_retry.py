"""The two lifecycle calls survive a blip; everything else is left alone.

Per-case result POSTs are already isolated -- one that fails costs one case, and the loss is
counted on the upload state. Run registration and completion are not: a single dropped packet on
register aborts an evaluation before it runs a case, and one on complete leaves a finished run
stuck in ``running`` on the platform forever. Those two get a small bounded retry -- transient
failures only, since a 400 will never get better. Parity with
traceroot-ts/tests/eval-lifecycle-retry.test.ts.
"""

from __future__ import annotations

import socket
import time

import pytest

from traceroot.eval.platform import PlatformTransport, _retry_delay_ms
from traceroot.eval.results import EvalItemResult
from traceroot.eval.transport import RunHandle

_RUN = RunHandle(name="e", dataset_name="d", metadata=None)
_TRANSIENT = RuntimeError("POST https://h/x -> HTTP 503: upstream unavailable")


class _Flaky(PlatformTransport):
    """Fails the first ``failures`` attempts of EVERY endpoint, then answers normally."""

    def __init__(self, *args, failures: int = 1, error: BaseException = _TRANSIENT, **kwargs):
        super().__init__(*args, **kwargs)
        self.retry_base_delay_ms = 0  # no real waiting in tests; backoff is tested separately
        self.attempts: list[str] = []
        self.bodies: list[tuple[str, dict | None]] = []
        self._remaining: dict[str, int] = {}
        self._failures = failures
        self._error = error

    def _request(self, method, path, body=None):
        self.attempts.append(path)
        self.bodies.append((path, body))
        left = self._remaining.get(path, self._failures)
        if left > 0:
            self._remaining[path] = left - 1
            raise self._error
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_run_id": "run_1"}
        return {}

    def tries(self, suffix: str) -> int:
        return sum(1 for p in self.attempts if p.endswith(suffix))


def _t(**kwargs) -> _Flaky:
    return _Flaky("ds_1", api_key="tr-x", host_url="https://h", **kwargs)


def _item() -> EvalItemResult:
    return EvalItemResult(
        case_id="c0",
        input=1,
        output=1,
        expected=1,
        scores=[],
        scorer_errors={},
        error=None,
        trace_id=None,
    )


class TestTransientFailuresAreRetried:
    def test_create_run_recovers(self):
        t = _t()
        t.create_run("e", "d", None)
        assert t.run_id == "run_1"
        assert t.tries("/evaluation-runs") == 2

    def test_finish_run_recovers(self):
        t = _t(failures=0)
        t.create_run("e", "d", None)
        t._remaining.clear()
        t._failures = 1
        state = t.finish_run(_RUN)
        assert state.status == "uploaded"
        assert t.tries("/complete") == 2

    def test_a_connection_error_is_retried_too(self):
        # No HTTP status at all -- a reset socket or a DNS hiccup, the classic transient failure.
        t = _t(error=TimeoutError("timed out"))
        t.create_run("e", "d", None)
        assert t.tries("/evaluation-runs") == 2

    def test_keyless_registration_sends_a_stable_idempotency_key(self):
        # A create_run with NO client_run_id must still send one (generated), and the SAME one on
        # every retry -- so a retry after a lost response can never register a second run.
        t = _t(failures=1)  # one transient failure -> one retry, i.e. two registration attempts
        t.create_run("e", "d", None)
        keys = [b.get("client_run_id") for (p, b) in t.bodies if p.endswith("/evaluation-runs")]
        assert len(keys) == 2  # attempted twice
        assert all(keys)  # a key was present on both attempts
        assert len(set(keys)) == 1  # the SAME key -> idempotent


class TestPermanentFailuresAreNot:
    def test_a_rejected_payload_is_not_retried(self):
        t = _t(failures=99, error=RuntimeError("POST https://h/x -> HTTP 400: bad field"))
        with pytest.raises(RuntimeError, match="HTTP 400"):
            t.create_run("e", "d", None)
        assert t.tries("/evaluation-runs") == 1  # a 400 will never get better

    def test_it_gives_up_and_raises_the_real_error(self):
        t = _t(failures=99)
        with pytest.raises(RuntimeError, match="HTTP 503"):
            t.create_run("e", "d", None)
        assert t.tries("/evaluation-runs") == t.retry_attempts

    def test_per_case_results_are_not_retried(self):
        """They are isolated by design: retrying every case's POST would multiply the load of a
        broken backend by the retry count, and a dropped result is already counted."""
        t = _t(failures=0)
        t.create_run("e", "d", None)
        t._remaining.clear()
        t._failures = 99
        with pytest.raises(RuntimeError):
            t.record_item_result(_RUN, _item())
        assert t.tries("/results") == 1


class _Bounded(_Flaky):
    """Also records the timeout each attempt is given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeouts: list[float] = []

    def _request(self, method, path, body=None, timeout=None):
        self.timeouts.append(timeout)
        return super()._request(method, path, body)


def _registered(**kwargs) -> _Bounded:
    t = _Bounded("ds_1", api_key="tr-x", host_url="https://h", failures=0)
    t.create_run("e", "d", None)
    t._remaining.clear()
    t.timeouts.clear()
    for name, value in kwargs.items():
        setattr(t, name, value)
    return t


class TestABoundedClose:
    """A cancelled run closes within the time it has left to be saved: ``finish_run(timeout=)``
    bounds the retries too. Parity with the TypeScript SDK's bounded ``finishRun``."""

    def test_a_retry_that_fits_in_the_bound_is_made(self):
        t = _registered(_failures=1, retry_base_delay_ms=100)
        assert t.finish_run(_RUN, "incomplete", timeout=2.0).status == "uploaded"
        assert t.tries("/complete") == 2
        # Each attempt gets the time left, never the full 30 s.
        assert 1.8 < t.timeouts[0] <= 2.0
        assert t.timeouts[1] <= t.timeouts[0] - 0.1

    def test_a_retry_whose_backoff_would_end_past_the_bound_is_not_started(self):
        t = _registered(_failures=99, retry_base_delay_ms=500)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="HTTP 503"):
            t.finish_run(_RUN, "incomplete", timeout=0.3)
        assert t.tries("/complete") == 1
        assert time.monotonic() - started < 0.3, "waited for a retry that could not fit"

    def test_no_attempt_is_made_with_no_time_left(self):
        t = _registered()
        with pytest.raises(TimeoutError):
            t.finish_run(_RUN, "incomplete", timeout=0.0)
        assert t.tries("/complete") == 0

    def test_without_a_bound_each_attempt_is_called_as_before(self):
        # No timeout argument at all, so an overridden request seam that doesn't take one works.
        t = _t(failures=0)
        t.create_run("e", "d", None)
        assert t.finish_run(_RUN).status == "uploaded"

    def test_a_request_seam_that_takes_no_timeout_is_called_without_one(self):
        # An override of the seam written without the parameter still closes the run...
        t = _t(failures=0)
        t.create_run("e", "d", None)
        assert t.finish_run(_RUN, "incomplete", timeout=1.0).status == "uploaded"
        # ... and the bound still applies to its retries.
        t._remaining.clear()
        t._failures = 99
        t.retry_base_delay_ms = 500
        with pytest.raises(RuntimeError, match="HTTP 503"):
            t.finish_run(_RUN, "incomplete", timeout=0.3)
        assert t.tries("/complete") == 2

    def test_an_attempt_still_waiting_at_the_bound_is_cut_off(self):
        # A TraceRoot that takes the request and never answers it.
        silent = socket.socket()
        silent.bind(("127.0.0.1", 0))
        silent.listen()
        try:
            t = PlatformTransport(
                "ds_1", api_key="tr-x", host_url=f"http://127.0.0.1:{silent.getsockname()[1]}"
            )
            t.run_id = "run_1"
            started = time.monotonic()
            with pytest.raises(OSError):
                t.finish_run(_RUN, "incomplete", timeout=0.3)
            waited = time.monotonic() - started
        finally:
            silent.close()
        assert 0.25 < waited < 1.0, f"gave up after {waited:.2f}s"


class TestBackoff:
    def test_the_delay_doubles_per_attempt(self):
        # Byte-identical to the TypeScript SDK's retryDelayMs.
        assert [_retry_delay_ms(n, 500) for n in (1, 2, 3)] == [500, 1000, 2000]
