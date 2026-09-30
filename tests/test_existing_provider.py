"""Tests for initialize() when a TracerProvider is already installed globally (e.g. `adk web`).

The `memory_exporter` fixture installs such a host provider.
"""

import logging
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traceroot
from tests.utils import reset_traceroot
from traceroot import observe
from traceroot.transport.span_processor import LocalEvalSampler, mark_local_eval_run


@pytest.fixture
def traceroot_exported(monkeypatch):
    """Spans handed to TraceRoot's OTLP exporter (stubbed: nothing leaves the process)."""
    exported = []

    def capture(self, spans):
        exported.extend(spans)
        return SpanExportResult.SUCCESS

    monkeypatch.setattr(OTLPSpanExporter, "export", capture)
    return exported


def _init():
    return traceroot.initialize(api_key="test-key", host_url="http://127.0.0.1:9")


def test_joins_existing_global_provider(memory_exporter, traceroot_exported, caplog):
    host_provider = trace.get_tracer_provider()

    with (
        patch("opentelemetry.trace.set_tracer_provider") as mock_set,
        caplog.at_level(logging.INFO, logger="traceroot.client"),
    ):
        client = _init()

    assert client._provider is host_provider
    mock_set.assert_not_called()
    assert "already installed globally" in caplog.text

    @observe(name="joined-op")
    def op():
        return "ok"

    op()
    traceroot.flush()

    assert [s.name for s in traceroot_exported] == ["joined-op"]
    assert traceroot_exported[0].attributes["traceroot.sdk.name"]
    assert [s.name for s in memory_exporter.get_finished_spans()] == ["joined-op"]


def test_local_run_spans_skip_traceroot_but_reach_host_for_early_and_late_tracers(
    memory_exporter, traceroot_exported
):
    host_provider = trace.get_tracer_provider()
    early_tracer = host_provider.get_tracer("host-early")
    _init()
    late_tracer = host_provider.get_tracer("host-late")

    with mark_local_eval_run():
        with early_tracer.start_as_current_span("early-local"):
            pass
        with late_tracer.start_as_current_span("late-local"):
            pass
    with late_tracer.start_as_current_span("control"):
        pass
    traceroot.flush()

    host_spans = [s.name for s in memory_exporter.get_finished_spans()]
    assert host_spans == ["early-local", "late-local", "control"]
    assert [s.name for s in traceroot_exported] == ["control"]


def test_joined_provider_sampler_is_left_alone(memory_exporter, traceroot_exported):
    host_provider = trace.get_tracer_provider()
    original = host_provider.sampler
    _init()
    reset_traceroot()
    _init()
    assert host_provider.sampler is original
    assert not isinstance(original, LocalEvalSampler)


def test_shutdown_leaves_host_provider_running(memory_exporter, traceroot_exported):
    _init()
    traceroot.shutdown()
    traceroot_exported.clear()

    with trace.get_tracer("host-app").start_as_current_span("after-shutdown"):
        pass

    spans = memory_exporter.get_finished_spans()
    assert [s.name for s in spans] == ["after-shutdown"]
    assert "traceroot.sdk.name" not in spans[0].attributes
    assert traceroot_exported == []


def test_shutdown_processor_does_not_block_host_force_flush(traceroot_exported):
    # The provider stops at the first processor whose force_flush() returns False.
    reset_traceroot()
    host_provider = TracerProvider()
    with patch("opentelemetry.trace.get_tracer_provider", return_value=host_provider):
        _init()
    traceroot.shutdown()
    host_exporter = InMemorySpanExporter()
    host_provider.add_span_processor(BatchSpanProcessor(host_exporter, schedule_delay_millis=60000))

    with host_provider.get_tracer("host-app").start_as_current_span("after-shutdown"):
        pass

    assert host_provider.force_flush() is True
    assert [s.name for s in host_exporter.get_finished_spans()] == ["after-shutdown"]
    reset_traceroot()
    host_provider.shutdown()


@pytest.mark.parametrize(
    ("existing", "warning"),
    [(trace.ProxyTracerProvider(), None), (trace.NoOpTracerProvider(), "NoOpTracerProvider")],
    ids=["no-global-provider", "non-sdk-provider"],
)
def test_creates_and_sets_own_provider_without_a_global_sdk_provider(existing, warning, caplog):
    reset_traceroot()
    with (
        patch("opentelemetry.trace.get_tracer_provider", return_value=existing),
        patch("opentelemetry.trace.set_tracer_provider") as mock_set,
        caplog.at_level(logging.WARNING, logger="traceroot.client"),
    ):
        client = _init()

    mock_set.assert_called_once_with(client._provider)
    assert isinstance(client._provider.sampler, LocalEvalSampler)
    assert ("cannot be shared" in caplog.text) == bool(warning)
    assert (warning or "") in caplog.text
    reset_traceroot()
