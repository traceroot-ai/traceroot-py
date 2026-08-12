"""Finalization honesty.

The completion call runs in the engine's ``finally``, so it is the last thing that can go
wrong -- and the easiest place to lose the truth. Two rules:

  1. a completion failure never MASKS the error the run already failed with, and on its own
     it surfaces as ONE clear error (not a double traceback);
  2. per-case result POSTs that were dropped on the floor are COUNTED on the upload state,
     so a run that reports "uploaded" with silently-missing results is detectable.
"""

import pytest

from traceroot.eval import Dataset, EvalCase
from traceroot.eval.engine import _run
from traceroot.eval.transport import EvalCompletionError, FakeTransport


def _ds(n=1):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
    return ds


def echo(x):
    return x


def ok(ctx):
    return 1.0


class _BrokenFinish(FakeTransport):
    """The live failure: /complete rejects the payload (400) after the run is done."""

    def finish_run(self, run, status=None, emitted_metrics=None):
        raise RuntimeError("HTTP 400: /complete rejected the payload")


class _DroppingResults(FakeTransport):
    """A backend that has not deployed the new per-case fields: every result POST 400s."""

    def record_item_result(self, run, item_result):
        raise RuntimeError("HTTP 400: unknown field 'passed'")


class TestCompletionNeverMasksTheRealError:
    def test_run_failure_survives_a_failing_completion(self):
        # Body raises AND finish_run raises. The user must still see the real cause -- the
        # 400 buried a `RuntimeError: no running event loop` in the field.
        def boom(item, duration_ms):
            raise RuntimeError("no running event loop")

        with pytest.raises(RuntimeError) as excinfo:
            _run(
                name="r",
                data=_ds(1),
                task=echo,
                scorers=[ok],
                transport=_BrokenFinish(),
                on_case_complete=boom,
            )
        assert "no running event loop" in str(excinfo.value)
        assert not isinstance(excinfo.value, EvalCompletionError)
        # The completion failure is not lost either -- it rides along as secondary context.
        notes = " ".join(getattr(excinfo.value, "__notes__", []))
        assert "/complete rejected the payload" in notes

    def test_completion_failure_alone_is_one_clear_error(self):
        with pytest.raises(EvalCompletionError) as excinfo:
            _run(name="r", data=_ds(1), task=echo, scorers=[ok], transport=_BrokenFinish())
        err = excinfo.value
        assert "/complete rejected the payload" in str(err)  # names the completion failure
        assert "running" in str(err)  # says what it means for the run
        # One traceback, not two: the transport error is carried as data, not chained.
        assert err.__suppress_context__ is True
        assert err.__cause__ is None
        assert isinstance(err.completion_error, RuntimeError)


class TestDroppedResultsAreCounted:
    def test_dropped_result_posts_are_surfaced_on_the_upload_state(self):
        run = _run(name="r", data=_ds(3), task=echo, scorers=[ok], transport=_DroppingResults())
        assert run.upload_state.status == "uploaded"
        assert run.upload_state.failed_result_count == 3
        assert run.upload_state.partial is True
        assert run.upload_state.to_dict()["failed_result_count"] == 3

    def test_a_clean_run_reports_no_dropped_results(self):
        run = _run(name="r", data=_ds(2), task=echo, scorers=[ok], transport=FakeTransport())
        assert run.upload_state.failed_result_count == 0
        assert run.upload_state.partial is False
