"""Tests for TRACEROOT_ENVIRONMENT / TRACEROOT_RELEASE stamping.

Covers:
  - ``traceroot.environment`` and ``traceroot.release`` span attributes are
    stamped on every recording span when configured on the processor
  - The processor reads TRACEROOT_ENVIRONMENT / TRACEROOT_RELEASE env vars
    when no explicit value is given (issue #200)
  - TracerootClient resolves explicit args > env vars into self.environment/self.release
  - traceroot.initialize() passes environment/release through
  - Nothing is stamped when the values are unset
"""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import traceroot
from traceroot import client as client_mod
from traceroot.span_attributes import SpanAttributes
from traceroot.transport.span_processor import TracerootSpanProcessor


@pytest.fixture
def proc_setup():
    """Local TracerProvider with TracerootSpanProcessor + InMemorySpanExporter."""
    exporter = InMemorySpanExporter()
    processor = TracerootSpanProcessor(
        api_key="test-key",
        host_url="http://localhost:9999",
        environment="staging",
        release="v1.2.3",
    )
    provider = TracerProvider()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    yield tracer, exporter, processor
    provider.shutdown()
    exporter.clear()


def test_environment_and_release_stamped_on_every_span(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    for s in spans:
        assert s.attributes[SpanAttributes.ENVIRONMENT] == "staging"
        assert s.attributes[SpanAttributes.RELEASE] == "v1.2.3"


def test_environment_and_release_not_stamped_when_unset(monkeypatch):
    for v in ("TRACEROOT_ENVIRONMENT", "TRACEROOT_RELEASE"):
        monkeypatch.delenv(v, raising=False)
    exporter = InMemorySpanExporter()
    processor = TracerootSpanProcessor(api_key="test-key", host_url="http://localhost:9999")
    assert processor._environment is None
    assert processor._release is None
    provider = TracerProvider()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("s"):
        pass
    spans = exporter.get_finished_spans()
    assert spans[0].attributes.get(SpanAttributes.ENVIRONMENT) is None
    assert spans[0].attributes.get(SpanAttributes.RELEASE) is None
    provider.shutdown()


def test_processor_reads_env_vars(monkeypatch):
    monkeypatch.setenv("TRACEROOT_ENVIRONMENT", "production")
    monkeypatch.setenv("TRACEROOT_RELEASE", "abc123")
    processor = TracerootSpanProcessor(api_key="test-key", host_url="http://localhost:9999")
    assert processor._environment == "production"
    assert processor._release == "abc123"


def test_client_reads_env_vars_and_explicit_overrides(monkeypatch):
    monkeypatch.setenv("TRACEROOT_ENVIRONMENT", "staging")
    monkeypatch.setenv("TRACEROOT_RELEASE", "v0.9")
    monkeypatch.setattr(client_mod.TracerootClient, "_initialize", lambda self: None)
    c = client_mod.TracerootClient(api_key="test", environment="prod", release="v1")
    assert c.environment == "prod"
    assert c.release == "v1"
    c2 = client_mod.TracerootClient(api_key="test")
    assert c2.environment == "staging"
    assert c2.release == "v0.9"


def test_initialize_passes_environment_and_release(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(traceroot, "TracerootClient", FakeClient)
    monkeypatch.setattr(traceroot, "_client", None, raising=False)
    client = traceroot.initialize(environment="prod", release="v2")
    assert captured["environment"] == "prod"
    assert captured["release"] == "v2"
    assert isinstance(client, FakeClient)
    traceroot._client = None
