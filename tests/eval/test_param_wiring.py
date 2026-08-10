"""Every public Evaluation/evaluate argument is wired through or rejected,
never silently ignored (audit follow-up)."""

import asyncio

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
        # Free-form metadata reaches registration; no SDK-identity provenance rides along.
        assert seen["metadata"] == {"model": "gpt"}


class TestRetryRejected:
    def test_retry_is_explicitly_rejected_not_ignored(self):
        with pytest.raises(NotImplementedError):
            Evaluation(name="r", dataset=_ds(1), task=echo, scorers=[ok], retry=3)

    def test_retry_none_is_fine(self):
        Evaluation(name="r", dataset=_ds(1), task=echo, scorers=[ok], retry=None)  # no raise
