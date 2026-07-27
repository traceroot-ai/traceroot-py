"""DS-3: low-level RunSession lifecycle over the EvalTransport seam."""

from traceroot.eval import EvalCase, Score
from traceroot.eval.results import EvalItemResult
from traceroot.eval.session import RunSession
from traceroot.eval.transport import FakeTransport


def _item(case_id, scores=None, output=None, trace_id=None):
    return EvalItemResult(
        case_id=case_id,
        input={"m": case_id},
        output=output,
        expected=None,
        scores=scores or [],
        scorer_errors={},
        error=None,
        trace_id=trace_id,
    )


class TestLifecycle:
    def test_full_sequence(self):
        fake = FakeTransport()
        s = RunSession(fake, name="r", dataset_name="d").start()
        s.register(EvalCase(input=1, id="c0"))
        s.record(_item("c0", scores=[Score("acc", 1.0)], output={"r": 1}))
        s.complete()
        kinds = [c[0] for c in fake.calls]
        assert kinds[0] == "create_run"
        assert "register_item" in kinds
        assert "record_item_result" in kinds
        assert kinds[-1] == "finish_run"

    def test_client_run_id_stable(self):
        s = RunSession(FakeTransport(), name="r", dataset_name="d")
        assert s.client_run_id
        assert s.client_run_id == s.client_run_id  # stable across reads

    def test_client_run_id_reaches_create_run(self):
        fake = FakeTransport()
        s = RunSession(fake, name="r", dataset_name="d", client_run_id="crun_fixed").start()
        create = next(c for c in fake.calls if c[0] == "create_run")
        assert create[3] == "crun_fixed"  # idempotency key threaded to the transport
        assert s.client_run_id == "crun_fixed"

    def test_complete_returns_upload_state(self):
        s = RunSession(FakeTransport(), name="r", dataset_name="d").start()
        state = s.complete()
        assert state.status in ("local_only", "uploaded")


class TestPartialUpdates:
    def test_attach_trace_merges_without_clobbering_scores(self):
        fake = FakeTransport()
        s = RunSession(fake, name="r", dataset_name="d").start()
        s.record(_item("c0", scores=[Score("acc", 1.0)], output={"r": 1}))
        s.attach_trace("c0", "trace-xyz")
        # last recorded item for c0 keeps its scores and gains the trace id
        merged = s.item("c0")
        assert merged.trace_id == "trace-xyz"
        assert [sc.name for sc in merged.scores] == ["acc"]
        assert merged.output == {"r": 1}

    def test_score_merges_into_existing_item(self):
        s = RunSession(FakeTransport(), name="r", dataset_name="d").start()
        s.register(EvalCase(input=1, id="c0"))
        s.score("c0", [Score("human", 1.0, comment="looks good")])
        assert [sc.name for sc in s.item("c0").scores] == ["human"]


class TestTerminalStatuses:
    def test_fail_sends_failed(self):
        fake = FakeTransport()
        RunSession(fake, name="r", dataset_name="d").start().fail(reason="boom")
        finish = next(c for c in fake.calls if c[0] == "finish_run")
        assert finish[1] == "failed"

    def test_cancel_sends_incomplete_until_backend_supports_cancelled(self):
        fake = FakeTransport()
        RunSession(fake, name="r", dataset_name="d").start().cancel()
        finish = next(c for c in fake.calls if c[0] == "finish_run")
        assert finish[1] == "incomplete"

    def test_complete_default_status(self):
        fake = FakeTransport()
        RunSession(fake, name="r", dataset_name="d").start().complete()
        finish = next(c for c in fake.calls if c[0] == "finish_run")
        assert finish[1] in (None, "completed")
