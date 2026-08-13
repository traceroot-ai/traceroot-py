"""Event-loop hygiene: nothing blocks the loop, nothing is left dangling.

Two defects the async engine can hide, because both still produce correct RESULTS:

  1. the reporting transport is synchronous (urllib, 30s timeout). Called straight from a
     per-case coroutine it blocks the loop thread, so per-case POSTs serialize behind one
     another and one slow backend inflates every other case's wall clock -- and delays the
     ``asyncio.wait`` that enforces their timeouts. Task and scorer code is already dispatched
     to a worker; reporting must be too.
  2. a timed-out case cancels its future and walks away without awaiting it, so the cancelled
     task settles unobserved -- "Task exception was never retrieved" noise on every timeout.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from traceroot.eval import Dataset, EvalCase
from traceroot.eval.engine import _CaseTimeoutError, _bounded, _run
from traceroot.eval.transport import FakeTransport


def _ds(n: int) -> Dataset:
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
    return ds


class _SlowTransport(FakeTransport):
    """A reporting transport that BLOCKS, like the real urllib-backed one does."""

    def __init__(self, delay: float = 0.2) -> None:
        super().__init__()
        self.delay = delay
        self.threads: list[int] = []

    def _block(self) -> None:
        self.threads.append(threading.get_ident())
        time.sleep(self.delay)

    def create_run(self, *args, **kwargs):
        self._block()
        return super().create_run(*args, **kwargs)

    def record_item_result(self, run, item_result):
        self._block()
        return super().record_item_result(run, item_result)

    def finish_run(self, *args, **kwargs):
        self._block()
        return super().finish_run(*args, **kwargs)


class TestReportingRunsOffTheLoop:
    def test_per_case_reporting_does_not_serialize_the_run(self):
        cases = 4
        delay = 0.2
        t = _SlowTransport(delay)
        started = time.perf_counter()
        _run(
            name="r",
            data=_ds(cases),
            task=lambda x: x,
            scorers=[lambda ctx: 1.0],
            transport=t,
            max_concurrency=cases,
        )
        elapsed = time.perf_counter() - started
        # Serialized on the loop this is create + 4 * results + finish = 6 delays; concurrent it
        # is create + one round of results + finish = 3. The midpoint separates them with room to
        # spare on a loaded machine.
        assert elapsed < delay * 4.5, f"reporting serialized the run ({elapsed:.2f}s)"

    def test_reporting_never_runs_on_the_loop_thread(self):
        t = _SlowTransport(0.0)
        loop_thread = threading.get_ident()  # _run drives the loop on the calling thread
        _run(name="r", data=_ds(2), task=lambda x: x, scorers=[lambda ctx: 1.0], transport=t)
        assert t.threads, "the transport was never called"
        assert loop_thread not in t.threads


class TestTimedOutCasesLeaveNothingDangling:
    @pytest.mark.asyncio
    async def test_a_timed_out_future_is_settled_before_the_error_is_raised(self, monkeypatch):
        """``_bounded`` cancels the overrunning future; cancellation is only DELIVERED on a later
        tick, so walking away leaves a live task nobody ever observes."""
        captured: list[asyncio.Future] = []
        real_ensure_future = asyncio.ensure_future

        def _capture(coro, **kwargs):
            fut = real_ensure_future(coro, **kwargs)
            captured.append(fut)
            return fut

        monkeypatch.setattr(asyncio, "ensure_future", _capture)

        async def never() -> None:
            await asyncio.sleep(30)

        with pytest.raises(_CaseTimeoutError):
            await _bounded(never(), time.monotonic() + 0.01, 0.01)

        assert captured and captured[0].done()

    @pytest.mark.asyncio
    async def test_an_exception_raised_while_cancelling_is_retrieved(self, monkeypatch):
        """A coroutine that turns its cancellation into a different error is the case that
        actually emits 'exception was never retrieved'."""
        captured: list[asyncio.Future] = []
        real_ensure_future = asyncio.ensure_future

        def _capture(coro, **kwargs):
            fut = real_ensure_future(coro, **kwargs)
            captured.append(fut)
            return fut

        monkeypatch.setattr(asyncio, "ensure_future", _capture)

        async def rogue() -> None:
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                raise ValueError("cleanup blew up") from None

        with pytest.raises(_CaseTimeoutError):
            await _bounded(rogue(), time.monotonic() + 0.01, 0.01)

        fut = captured[0]
        assert fut.done() and not fut.cancelled()
        assert isinstance(fut.exception(), ValueError)  # observed, not left to warn at GC
