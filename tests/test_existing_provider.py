"""Tests for initialize() when a TracerProvider is already installed globally (e.g. `adk web`).

The `memory_exporter` fixture installs such a host provider.
"""

import logging
from unittest.mock import patch

import pytest
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

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


def test_local_eval_gate_wraps_joined_provider_once(memory_exporter, traceroot_exported):
    _init()
    reset_traceroot()
    client = _init()
    sampler = client._provider.sampler
    assert isinstance(sampler, LocalEvalSampler)
    assert not isinstance(sampler._inner, LocalEvalSampler)

    @observe(name="local-only")
    def op():
        return "ok"

    with mark_local_eval_run():
        op()
    traceroot.flush()

    assert traceroot_exported == []
    assert memory_exporter.get_finished_spans() == ()


def test_local_eval_gate_covers_host_tracer_created_before_initialize(
    memory_exporter, traceroot_exported, monkeypatch
):
    # A host tracer made before initialize() captured the provider's original sampler, so the
    # LocalEvalSampler wrap never sees its spans; TraceRoot's processor must drop them itself.
    host_provider = trace.get_tracer_provider()
    original = host_provider.sampler
    if isinstance(original, LocalEvalSampler):
        original = original._inner
    monkeypatch.setattr(host_provider, "sampler", original)
    early_tracer = host_provider.get_tracer("host-app")
    _init()

    with mark_local_eval_run(), early_tracer.start_as_current_span("local-run"):
        pass
    with early_tracer.start_as_current_span("reported-run"):
        pass
    traceroot.flush()

    assert [s.name for s in traceroot_exported] == ["reported-run"]


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

    assert client._owns_provider is True
    mock_set.assert_called_once_with(client._provider)
    assert ("cannot be shared" in caplog.text) == bool(warning)
    assert (warning or "") in caplog.text
    reset_traceroot()
