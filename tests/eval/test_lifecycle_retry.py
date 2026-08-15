"""The two lifecycle calls survive a blip; everything else is left alone.

Per-case result POSTs are already isolated -- one that fails costs one case, and the loss is
counted on the upload state. Run registration and completion are not: a single dropped packet on
register aborts an evaluation before it runs a case, and one on complete leaves a finished run
stuck in ``running`` on the platform forever. Those two get a small bounded retry -- transient
failures only, since a 400 will never get better. Parity with
traceroot-ts/tests/eval-lifecycle-retry.test.ts.
"""

from __future__ import annotations

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


class TestBackoff:
    def test_the_delay_doubles_per_attempt(self):
        # Byte-identical to the TypeScript SDK's retryDelayMs.
        assert [_retry_delay_ms(n, 500) for n in (1, 2, 3)] == [500, 1000, 2000]
