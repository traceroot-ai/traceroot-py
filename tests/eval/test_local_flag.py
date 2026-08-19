"""``evaluate(..., local=True)``: run in full, report nothing.

``transport=`` is the power knob -- it names the sink a run reports to. ``local=True`` is the
convenience for the common "just run it, don't report anywhere" case: the task and scorers execute
in full and a complete ``EvalRunResult`` comes back, but nothing leaves the process. No credentials
are needed, no dataset is auto-published, no run is registered. It is exactly
``transport=FakeTransport()`` without making the user import a class named "Fake", so the two must
stay observationally identical. Parity with traceroot-ts/tests/eval-local-flag.test.ts.
"""

from __future__ import annotations

import contextlib
import contextvars
import urllib.request

import pytest

import traceroot.eval.dataset_sync as sync_mod
import traceroot.eval.engine as engine
import traceroot.eval.platform as platform
from traceroot.eval import Dataset, Evaluation, evaluate, evaluate_async
from traceroot.eval.transport import FakeTransport

# Every test here drives the real transport-resolution path, so the conftest default that
# hands a bare evaluate() a FakeTransport must stay out of the way.
pytestmark = pytest.mark.no_default_transport

_MUTUALLY_EXCLUSIVE = "local=True OR transport="


def _dataset(name: str = "local-flag") -> Dataset:
    d = Dataset(name)
    d.add(input={"m": 1}, expected={"m": 1})
    d.add(input={"m": 2}, expected={"m": 2})
    return d


def _synced_dataset() -> Dataset:
    """A dataset that was pulled/pushed -- the shape that WOULD report on the default path."""
    d = _dataset("already-synced")
    d.dataset_version_id = "dsv_9"
    d.base_version_id = "dsv_9"
    return d


def echo(x):
    return x


def exact(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


@contextlib.contextmanager
def _already_initialized_client(monkeypatch):
    """An app that ALREADY called ``initialize()``: a fresh, fully-initialized (enabled, exporting)
    provider with a real OpenAI instrumentation integration -- the starting point the origin gate
    must cover. Fake host; the export spy sits before export, so nothing touches the network.

    Don't let ``initialize()`` claim OTel's process-global tracer-provider slot (set-once): the run
    is driven through the client's own provider, and claiming the slot would poison sibling tests.
    Yields the client; undoes the real (global) OpenAI instrumentation on exit so it can't leak onto
    sibling tests (the rest of the suite mocks the instrumentor for this reason)."""
    import traceroot
    from traceroot import Integration

    monkeypatch.setattr("opentelemetry.trace.set_tracer_provider", lambda *a, **k: None)
    traceroot.shutdown()
    traceroot._client = None
    client = traceroot.initialize(
        api_key="k", host_url="https://host.invalid", integrations=[Integration.OPENAI]
    )
    try:
        yield client
    finally:
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().uninstrument()
        traceroot.shutdown()
        traceroot._client = None


def _summary(result) -> dict:
    """The observable shape of a run, minus the per-run ids and timings."""
    return {
        "name": result.name,
        "counts": (result.case_count, result.errored, result.not_scored),
        "scores": {k: (v.mean, v.count) for k, v in result.score_summary.items()},
        "upload": (result.upload_state.status, result.upload_state.dashboard_url),
        "run_id": result.run_id,
    }


@pytest.fixture
def no_exfiltration(monkeypatch):
    """Fail loudly on every route out of the process: HTTP, the reporting-transport builder,
    and the dataset-publish client."""

    def _boom(*_a, **_k):
        raise AssertionError("local=True must not reach the platform")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(engine, "_auto_transport", _boom)
    monkeypatch.setattr(sync_mod, "PlatformDatasetSync", _boom)


@pytest.fixture
def no_credentials(monkeypatch):
    monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("", None))


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))


@pytest.fixture
def spy_transport(monkeypatch):
    """Capture the in-memory transport ``local=True`` selects, to prove the run really was
    recorded (locally) rather than skipped."""
    made: list[FakeTransport] = []

    class Spy(FakeTransport):
        def __init__(self) -> None:
            super().__init__()
            made.append(self)

    monkeypatch.setattr(engine, "FakeTransport", Spy)
    return made


@pytest.fixture
def export_spy(monkeypatch):
    """Spy at the exporter boundary. ``TracerootSpanProcessor.on_end`` delegates to
    ``BatchSpanProcessor.on_end`` (queue-for-export) only for spans it does NOT drop, so recording
    the name of every span that reaches it is exactly what makes "exported vs dropped" observable.
    Restored automatically in teardown by monkeypatch."""
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    exported: list[str] = []

    def _record(self, span):
        exported.append(span.name)

    monkeypatch.setattr(BatchSpanProcessor, "on_end", _record)
    return exported


class TestLocalRunsFullyAndReportsNothing:
    def test_completes_without_credentials(self, no_credentials, no_exfiltration):
        result = evaluate(name="r", data=_dataset(), task=echo, scorers=[exact], local=True)

        assert result.case_count == 2
        assert result.errored == 0
        assert result.score_summary["exact"].mean == 1.0
        assert result.score_summary["exact"].count == 2
        # A non-uploaded run: no server run id and no dashboard to link to.
        assert result.run_id is None
        assert result.upload_state.dashboard_url is None

    async def test_evaluate_async_completes_without_credentials(
        self, no_credentials, no_exfiltration
    ):
        result = await evaluate_async(
            name="r", data=_dataset(), task=echo, scorers=[exact], local=True
        )
        assert result.case_count == 2
        assert result.score_summary["exact"].mean == 1.0

    def test_the_run_is_recorded_in_memory(self, no_credentials, no_exfiltration, spy_transport):
        evaluate(name="r", data=_dataset(), task=echo, scorers=[exact], local=True)

        assert len(spy_transport) == 1
        kinds = [c[0] for c in spy_transport[0].calls]
        assert kinds.count("create_run") == 1
        assert kinds.count("register_item") == 2
        assert kinds.count("record_item_result") == 2
        assert kinds.count("finish_run") == 1
        assert "publish_dataset" not in kinds

    def test_evaluation_definition_honors_local(self, no_credentials, no_exfiltration):
        result = Evaluation(
            name="r", dataset=_dataset(), task=echo, scorers=[exact], local=True
        ).run()
        assert result.case_count == 2


class TestLocalOnASyncedDataset:
    """A dataset with a version id is exactly what the default path reports against; ``local=True``
    still runs it, and still reports nothing and re-publishes nothing."""

    def test_synced_dataset_runs_local(self, credentials, no_exfiltration):
        d = _synced_dataset()

        result = evaluate(name="r", data=d, task=echo, scorers=[exact], local=True)

        assert result.case_count == 2
        assert result.run_id is None
        assert d.dataset_version_id == "dsv_9"  # untouched: no re-publish


class TestMutualExclusion:
    """``local=True`` already answers "where does this report" -- passing a sink too is a
    contradiction, not a precedence puzzle."""

    def test_local_with_transport_raises(self, no_credentials):
        with pytest.raises(ValueError, match=_MUTUALLY_EXCLUSIVE):
            evaluate(
                name="r",
                data=_dataset(),
                task=echo,
                scorers=[exact],
                local=True,
                transport=FakeTransport(),
            )

    def test_local_with_report_to_raises(self, no_credentials):
        with pytest.raises(ValueError, match=_MUTUALLY_EXCLUSIVE):
            evaluate(
                name="r",
                data=_dataset(),
                task=echo,
                scorers=[exact],
                local=True,
                report_to=FakeTransport(),
            )

    def test_evaluation_definition_with_report_to_raises(self, no_credentials):
        with pytest.raises(ValueError, match=_MUTUALLY_EXCLUSIVE):
            Evaluation(
                name="r",
                dataset=_dataset(),
                task=echo,
                scorers=[exact],
                local=True,
                report_to=FakeTransport(),
            )


class TestEquivalentToAnExplicitFakeTransport:
    def test_same_result_shape_and_same_recorded_calls(
        self, no_credentials, no_exfiltration, spy_transport
    ):
        explicit = FakeTransport()
        via_transport = evaluate(
            name="r", data=_dataset(), task=echo, scorers=[exact], transport=explicit
        )
        via_local = evaluate(name="r", data=_dataset(), task=echo, scorers=[exact], local=True)

        assert _summary(via_local) == _summary(via_transport)
        assert len(spy_transport) == 1  # only the local run selected one for itself
        # The same set of calls reached both. Compared as a multiset: per-case results are
        # recorded in completion order, which concurrency leaves free to differ between runs;
        # the per-run idempotency key on create_run is dropped for the same reason.
        assert sorted(c[:3] for c in spy_transport[0].calls) == sorted(
            c[:3] for c in explicit.calls
        )


class TestDefaultPathUnchanged:
    """``local`` defaults to False and only adds a branch: with credentials and a locally-authored
    dataset, an unset ``local`` still auto-publishes and still reports."""

    def test_unset_local_still_auto_publishes_and_reports(self, monkeypatch, credentials):
        published: list[str] = []

        class RecordingSync:
            def __init__(self, *_a, **_k) -> None:
                pass

        def _fake_push(self, _sync, *_a, **_k):
            published.append(self.name)
            self.dataset_version_id = "dsv_10"
            self.base_version_id = "dsv_10"

        monkeypatch.setattr(sync_mod, "PlatformDatasetSync", RecordingSync)
        monkeypatch.setattr(Dataset, "push", _fake_push)
        reported = FakeTransport()
        monkeypatch.setattr(engine, "_auto_transport", lambda *a, **k: reported)

        result = evaluate(name="r", data=_dataset(), task=echo, scorers=[exact])

        assert published == ["local-flag"]
        assert [c[0] for c in reported.calls].count("create_run") == 1
        assert result.dataset.dataset_version_id == "dsv_10"


class TestLocalSuppressesGlobalAutoInit:
    """A local run must not let a task/judge ``@observe`` (or auto-instrumentation) lazily bring up
    an exporting provider and ship case data off-process: the lazy-init seam is suppressed for the
    whole run and released after. Parity with the TS ``_suppressGlobalAutoInit`` seam.

    A ``local=True`` run also drops export from a provider the app ALREADY initialized: an
    ``initialize()`` call before the eval leaves a live exporting provider whose auto-instrumented
    spans (OpenAI/Anthropic/...) would otherwise ship despite ``local=True``. The drop is
    ORIGIN-scoped (traceroot-ai/traceroot#1969): a span born inside the local run is stamped at
    creation and dropped at the exporter boundary no matter when it ends, while a span born OUTSIDE
    the run (e.g. a concurrent reported run) carries no stamp and exports normally."""

    def test_local_run_suppresses_lazy_auto_init(
        self, no_credentials, no_exfiltration, monkeypatch
    ):
        from traceroot import decorators

        brought_up: list[int] = []
        # get_client() is the lazy bring-up _ensure_initialized() would perform; it must NOT run
        # while a local eval is in flight.
        monkeypatch.setattr("traceroot.get_client", lambda: brought_up.append(1))

        seen_suppressed: list[bool] = []

        def task(x):
            decorators._ensure_initialized()  # what a task-side @observe does first
            seen_suppressed.append(decorators._is_global_auto_init_suppressed())
            return x

        evaluate(name="r", data=_dataset(), task=task, scorers=[exact], local=True)

        assert seen_suppressed and all(seen_suppressed)  # suppressed during every case
        assert brought_up == []  # lazy bring-up never happened
        assert decorators._is_global_auto_init_suppressed() is False  # released after the run

    def test_reported_run_does_not_suppress(self, credentials, monkeypatch):
        """The default (reporting) path is unchanged: suppression stays off, lazy init still allowed."""
        from traceroot import decorators

        monkeypatch.setattr(engine, "_auto_transport", lambda *a, **k: FakeTransport())

        seen_suppressed: list[bool] = []

        def task(x):
            seen_suppressed.append(decorators._is_global_auto_init_suppressed())
            return x

        evaluate(name="r", data=_synced_dataset(), task=task, scorers=[exact])

        assert seen_suppressed and not any(seen_suppressed)  # never suppressed on the reported path

    def test_already_initialized_provider_exports_nothing_during_local_run(
        self, no_credentials, no_exfiltration, export_spy, monkeypatch
    ):
        """The fix (sync path): an app that already called ``initialize()`` has a live exporting
        provider; a ``local=True`` run drops the spans it originates at the exporter boundary. A
        task span named like an auto-instrumented LLM call ("messages.create"), created through that
        already-initialized provider, forwards NOTHING to export."""
        with _already_initialized_client(monkeypatch) as client:

            def task(x):
                client._provider.get_tracer("app").start_span("messages.create").end()
                return x

            evaluate(name="r", data=_dataset(), task=task, scorers=[exact], local=True)

            assert export_spy == []  # nothing the local run originated left the process

    async def test_already_initialized_provider_exports_nothing_during_local_run_async(
        self, no_credentials, no_exfiltration, export_spy, monkeypatch
    ):
        """Same guarantee on the async entry path (``evaluate_async``): the guard rides the marker
        contextvar into every per-case asyncio task, so the task's instrumented span is dropped."""
        with _already_initialized_client(monkeypatch) as client:

            def task(x):
                client._provider.get_tracer("app").start_span("messages.create").end()
                return x

            await evaluate_async(name="r", data=_dataset(), task=task, scorers=[exact], local=True)

            assert export_spy == []

    async def test_already_initialized_provider_exports_nothing_from_worker_thread(
        self, no_credentials, no_exfiltration, export_spy, monkeypatch
    ):
        """The worker-thread path: a synchronous ``evaluate(local=True)`` called from INSIDE a
        running loop runs to completion in a worker thread (``_run`` -> ``pool.submit`` ->
        ``asyncio.run``). The marker is (re)established inside ``_run_async`` in that worker and the
        copied context carries it across the hop, so the task's instrumented span is still dropped."""
        with _already_initialized_client(monkeypatch) as client:

            def task(x):
                client._provider.get_tracer("app").start_span("messages.create").end()
                return x

            # This test is async, so a loop is already running on this thread -> the worker path.
            evaluate(name="r", data=_dataset(), task=task, scorers=[exact], local=True)

            assert export_spy == []

    def test_unrelated_reported_span_still_exports_during_local_run(self, export_spy):
        """Origin-scoping (#1969): a local run being active must not reach an unrelated, concurrently
        reporting provider. The local marker is set in one context; a reported span born in a
        SEPARATE context (its own run) is never stamped, so it still exports -- the process-global
        gate this replaced dropped it too."""
        from opentelemetry.sdk.trace import TracerProvider

        from traceroot.transport.span_processor import (
            TracerootSpanProcessor,
            mark_local_eval_run,
        )

        reported_provider = TracerProvider()
        reported_proc = TracerootSpanProcessor(api_key="k2", host_url="https://reported.invalid")
        reported_provider.add_span_processor(reported_proc)
        # The reported run's context, captured independently of any local run (marker unset in it).
        reported_ctx = contextvars.copy_context()
        try:
            with mark_local_eval_run():  # a local run is active in THIS context
                # The reported span is born in the reported run's own context -> no stamp -> exports.
                reported_ctx.run(
                    lambda: (
                        reported_provider.get_tracer("reported-app")
                        .start_span("reported.work")
                        .end()
                    )
                )

            assert "reported.work" in export_spy  # the unrelated reported run still exported
        finally:
            reported_proc.shutdown()

    def test_span_started_in_local_run_but_ended_after_is_dropped(self, export_spy):
        """A span BORN inside the local run whose ``on_end`` fires AFTER the run's context has
        exited is still dropped: the decision is stamped at birth (origin), not read at end. This is
        exactly the case the process-global gate got wrong (it exported spans that outlived the run)."""
        from opentelemetry.sdk.trace import TracerProvider

        from traceroot.transport.span_processor import (
            TracerootSpanProcessor,
            mark_local_eval_run,
        )

        provider = TracerProvider()
        proc = TracerootSpanProcessor(api_key="k", host_url="https://host.invalid")
        provider.add_span_processor(proc)
        try:
            with mark_local_eval_run():
                span = provider.get_tracer("app").start_span("late.work")  # on_start stamps it
            # The local run's context has now exited (marker reset); the span ends only now.
            span.end()

            assert "late.work" not in export_spy  # dropped despite ending after the run
        finally:
            proc.shutdown()
