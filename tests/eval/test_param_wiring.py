"""Every public Evaluation/evaluate argument is wired through or rejected,
never silently ignored (audit follow-up)."""

import asyncio
import time

import pytest

from traceroot.eval import Dataset, EvalCase, Evaluation, evaluate
from traceroot.eval.transport import FakeTransport, RunHandle


def _ds(n=2):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
    return ds


def echo(x):
    return x


def ok(ctx):
    return 1.0


async def slow(x):
    await asyncio.sleep(1.0)
    return x


async def slow_scorer(ctx):
    await asyncio.sleep(1.0)
    return 1.0


class TestTimeout:
    def test_constructor_timeout_is_applied(self):
        # Evaluation(timeout=...) must actually bound the task (was stored but dropped). An explicit
        # transport is supplied so the run doesn't depend on auto-transport behavior for an
        # unsynced inline dataset.
        run = Evaluation(
            name="r",
            dataset=_ds(1),
            task=slow,
            scorers=[ok],
            timeout=0.02,
            report_to=FakeTransport(),
        ).run()
        assert run.item_results[0].error is not None
        assert run.task_error_count == 1

    def test_evaluate_accepts_timeout(self):
        run = evaluate(
            name="r",
            dataset=_ds(1),
            task=slow,
            scorers=[ok],
            timeout=0.02,
            report_to=FakeTransport(),
        )
        assert run.task_error_count == 1

    def test_timeout_bounds_the_scorer_too(self):
        # One deadline covers task AND scorers (parity with the TS engine): a hung judge must
        # not outlive the budget, or `timeout=` silently doesn't bound the run at all.
        started = time.perf_counter()
        run = Evaluation(
            name="r",
            dataset=_ds(1),
            task=echo,
            scorers=[slow_scorer],
            timeout=0.05,
            report_to=FakeTransport(),
        ).run()
        assert time.perf_counter() - started < 0.9  # returned on the deadline, did not hang
        item = run.item_results[0]
        # A budget overrun errors the whole case (not an isolated per-scorer failure), and a
        # timed-out case is honestly "not scored".
        assert item.error is not None and "TimeoutError" in item.error
        assert item.scores == []

    def test_shared_deadline_is_not_restarted_per_scorer(self):
        # The budget is per CASE, not per call: a task that spends most of it leaves the
        # scorer only the remainder. The scorer's sleep is deliberately chosen to sit BETWEEN
        # the remaining budget (~0.05s) and a full fresh one (0.5s): it must time out under the
        # shared deadline, and would have completed had the deadline restarted per scorer.
        async def most_of_the_budget(x):
            await asyncio.sleep(0.45)
            return x

        async def scorer_within_a_fresh_budget(ctx):
            await asyncio.sleep(0.2)
            return 1.0

        run = Evaluation(
            name="r",
            dataset=_ds(1),
            task=most_of_the_budget,
            scorers=[scorer_within_a_fresh_budget],
            timeout=0.5,
            report_to=FakeTransport(),
        ).run()
        item = run.item_results[0]
        assert item.error is not None and "TimeoutError" in item.error
        assert item.scores == []

    def test_no_timeout_completes(self):
        run = Evaluation(
            name="r", dataset=_ds(1), task=echo, scorers=[ok], report_to=FakeTransport()
        ).run()
        assert run.task_error_count == 0


class TestMetadata:
    def test_metadata_on_run_result(self):
        # User metadata is preserved on the run result; auto provenance (git/ci) may add
        # more keys, so check the user key is present rather than exact-equality.
        run = evaluate(name="r", dataset=_ds(1), task=echo, scorers=[ok], metadata={"branch": "x"})
        assert run.metadata["branch"] == "x"

    def test_metadata_reaches_transport_create_run(self):
        seen = {}

        class CapturingTransport(FakeTransport):
            def create_run(self, name, dataset_name, metadata, client_run_id=None):
                seen["metadata"] = metadata
                return RunHandle(name=name, dataset_name=dataset_name, metadata=metadata)

        Evaluation(
            name="r",
            dataset=_ds(1),
            task=echo,
            scorers=[ok],
            metadata={"model": "gpt"},
            report_to=CapturingTransport(),
        ).run()
        # Free-form user metadata reaches registration (git/CI reproducibility provenance may ride
        # alongside as non-identity keys); SDK-language identity never does.
        assert seen["metadata"]["model"] == "gpt"
        assert "sdk" not in seen["metadata"] and "language" not in seen["metadata"]


class TestRetryRejected:
    def test_retry_is_explicitly_rejected_not_ignored(self):
        with pytest.raises(NotImplementedError):
            Evaluation(name="r", dataset=_ds(1), task=echo, scorers=[ok], retry=3)

    def test_retry_none_is_fine(self):
        Evaluation(name="r", dataset=_ds(1), task=echo, scorers=[ok], retry=None)  # no raise
