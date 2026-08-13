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

from traceroot.eval import Dataset, EvalCase, engine
from traceroot.eval.engine import _bounded, _CaseTimeoutError, _run
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
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0

    def _block(self) -> None:
        with self._lock:
            self.threads.append(threading.get_ident())
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        time.sleep(self.delay)
        with self._lock:
            self._in_flight -= 1

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
        t = _SlowTransport(0.2)
        _run(
            name="r",
            data=_ds(cases),
            task=lambda x: x,
            scorers=[lambda ctx: 1.0],
            transport=t,
            max_concurrency=cases,
        )
        # Overlap, not elapsed time, is the property: called on the loop thread the blocking POSTs
        # can only ever be one-in-flight, so any concurrency at all proves they were dispatched to
        # workers. Asserting a wall-clock budget instead is just a slow proxy that a loaded machine
        # can break.
        assert t.peak_in_flight > 1, "per-case reporting never overlapped -- it serialized"

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

    def test_a_case_that_ignores_its_cancellation_cannot_hold_the_run_open(self, monkeypatch):
        """The timed-out task is created on the RUN'S loop, and ``asyncio.run`` cancels and then
        AWAITS every remaining task before closing that loop. A task that swallows cancellation is
        therefore not merely orphaned -- it blocks loop shutdown, so ``evaluate()`` never returns
        even though the case already timed out."""
        monkeypatch.setattr(engine, "_CANCEL_GRACE_S", 0.05)
        work = 4.0  # how long the uncooperative case keeps running regardless of cancellation

        async def stubborn(x):
            end = time.monotonic() + work
            while time.monotonic() < end:
                try:
                    await asyncio.sleep(0.01)
                except asyncio.CancelledError:
                    pass  # ignores its cancellation entirely

        started = time.perf_counter()
        run = _run(
            name="r",
            data=_ds(1),
            task=stubborn,
            scorers=[lambda ctx: 1.0],
            transport=FakeTransport(),
            timeout=0.05,
        )
        elapsed = time.perf_counter() - started
        assert elapsed < work / 2, f"the run waited for the uncancellable case ({elapsed:.2f}s)"
        item = run.item_results[0]
        assert item.error is not None and "TimeoutError" in item.error
