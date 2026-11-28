"""Global attributes: a correlation value stamped on every span's metadata.

A tool that owns a child process's environment can set
``TRACEROOT_GLOBAL_ATTRIBUTES`` (or pass ``global_attributes=`` to
``initialize()``) and then find the spans it produced with the backend's
existing metadata filter.
"""

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traceroot
import traceroot.client as client_mod
import traceroot.transport.span_processor as span_processor_mod
from traceroot import observe
from traceroot.span_attributes import SpanAttributes
from traceroot.transport.span_processor import TracerootSpanProcessor


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Keep the processor's own exporter off the network; the test exporter is local."""

    class _NoopExporter(SpanExporter):
        def export(self, spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    monkeypatch.setattr(span_processor_mod, "OTLPSpanExporter", lambda **kwargs: _NoopExporter())


def _provider_with_processor(**kwargs):
    exporter = InMemorySpanExporter()
    processor = TracerootSpanProcessor(api_key="test", host_url="http://localhost:9999", **kwargs)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _metadata(span):
    raw = span.attributes.get(SpanAttributes.SPAN_METADATA)
    assert raw is not None
    return json.loads(raw)


def test_processor_stamps_global_attributes_on_root_and_child(memory_exporter):
    provider, exporter = _provider_with_processor(global_attributes={"traceroot.run_id": "run-1"})
    tracer = provider.get_tracer("test")
    try:
        with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
            pass
    finally:
        provider.shutdown()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert _metadata(spans["root"])["traceroot.run_id"] == "run-1"
    assert _metadata(spans["child"])["traceroot.run_id"] == "run-1"


def test_observe_and_update_merge_around_global_attributes(monkeypatch, memory_exporter):
    provider, exporter = _provider_with_processor(
        global_attributes={"traceroot.run_id": "run-1", "source": "tool"}
    )
    tracer = provider.get_tracer("test")
    # Point the decorator at this test's provider without touching OTel's set-once global.
    monkeypatch.setattr("traceroot.decorators.trace.get_tracer", lambda *a, **k: tracer)

    @observe(metadata={"phase": "task", "traceroot.run_id": "per-call-wins"})
    def work():
        traceroot.update_current_span(metadata={"step": 2})
        return "done"

    try:
        assert work() == "done"
    finally:
        provider.shutdown()

    metadata = _metadata(exporter.get_finished_spans()[0])
    assert metadata["traceroot.run_id"] == "per-call-wins"  # per-call wins
    assert metadata["source"] == "tool"  # global is kept
    assert metadata["phase"] == "task"
    assert metadata["step"] == 2


def test_client_reads_env_and_explicit_wins(monkeypatch):
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    monkeypatch.setenv("TRACEROOT_GLOBAL_ATTRIBUTES", '{"traceroot.run_id": "env"}')

    from_env = client_mod.TracerootClient(api_key="test")
    assert from_env.global_attributes == {"traceroot.run_id": "env"}

    explicit = client_mod.TracerootClient(
        api_key="test", global_attributes={"traceroot.run_id": "explicit"}
    )
    assert explicit.global_attributes == {"traceroot.run_id": "explicit"}


def test_malformed_env_is_ignored(monkeypatch, caplog):
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    monkeypatch.setenv("TRACEROOT_GLOBAL_ATTRIBUTES", "not json")
    with caplog.at_level("WARNING"):
        client = client_mod.TracerootClient(api_key="test")
    assert client.global_attributes is None
    assert any("not valid JSON" in r.message for r in caplog.records)


def test_unsupported_values_are_skipped(monkeypatch):
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    monkeypatch.setenv(
        "TRACEROOT_GLOBAL_ATTRIBUTES",
        '{"traceroot.run_id": "run-1", "nested": {"not": "a primitive"}, "n": 3}',
    )
    client = client_mod.TracerootClient(api_key="test")
    assert client.global_attributes == {"traceroot.run_id": "run-1", "n": 3}


def test_initialize_passes_global_attributes(monkeypatch, memory_exporter):
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    client = traceroot.initialize(api_key="test", global_attributes={"traceroot.run_id": "run-2"})
    assert client.global_attributes == {"traceroot.run_id": "run-2"}
