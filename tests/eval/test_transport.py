"""Transport seam - FakeTransport wiring + cloud-only default behavior."""

import pytest

from traceroot.eval import Dataset, EvalCase, evaluate
from traceroot.eval.transport import FakeTransport


def _ds(n):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
    return ds


def echo(x):
    return x


def exact(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


class TestCloudOnlyDefault:
    @pytest.mark.no_default_transport
    def test_default_evaluate_requires_reporting(self):
        # Cloud-only: with no credentials + an unsynced (inline) dataset there is nothing to
        # report to, so evaluate() raises rather than silently running locally.
        with pytest.raises(RuntimeError, match="reports to the TraceRoot platform"):
            evaluate(name="r", dataset=_ds(1), task=echo, scorers=[exact])


class TestFakeTransportWiring:
    def test_call_order_single_case(self):
        fake = FakeTransport()
        evaluate(name="r", dataset=_ds(1), task=echo, scorers=[exact], transport=fake)
        kinds = [c[0] for c in fake.calls]
        assert kinds[0] == "create_run"
        assert kinds[-1] == "finish_run"
        # register precedes both record calls for the case
        assert kinds.index("register_item") < kinds.index("record_item_result")
        assert kinds.index("register_item") < kinds.index("record_scores")

    def test_register_before_result_for_every_case(self):
        fake = FakeTransport()
        evaluate(
            name="r", dataset=_ds(4), task=echo, scorers=[exact], transport=fake, max_concurrency=1
        )
        for cid in ("c0", "c1", "c2", "c3"):
            reg = next(
                i for i, c in enumerate(fake.calls) if c[0] == "register_item" and c[1] == cid
            )
            rec = next(
                i for i, c in enumerate(fake.calls) if c[0] == "record_item_result" and c[1] == cid
            )
            assert reg < rec

    def test_register_fires_for_erroring_case(self):
        def boom(x):
            if x == 1:
                raise ValueError("no")
            return x

        fake = FakeTransport()
        evaluate(name="r", dataset=_ds(3), task=boom, scorers=[exact], transport=fake)
        registered = {c[1] for c in fake.calls if c[0] == "register_item"}
        assert registered == {"c0", "c1", "c2"}  # including the failing c1

    def test_single_finish_run(self):
        fake = FakeTransport()
        evaluate(name="r", dataset=_ds(3), task=echo, scorers=[exact], transport=fake)
        assert sum(1 for c in fake.calls if c[0] == "finish_run") == 1

    def test_explicit_transport_reports_uploaded(self):
        fake = FakeTransport()
        result = evaluate(name="r", dataset=_ds(1), task=echo, scorers=[exact], transport=fake)
        assert any(c[0] == "create_run" for c in fake.calls)
        assert result.upload_state.status == "uploaded"
