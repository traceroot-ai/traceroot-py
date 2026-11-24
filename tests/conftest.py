"""Shared test fixtures for the traceroot SDK test suite."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.utils import reset_traceroot

# ---------------------------------------------------------------------------
# Single shared OTel provider / exporter
#
# OTel's global tracer provider can only be set once per process.  All test
# files must share the same InMemorySpanExporter so spans end up in the right
# place regardless of test execution order.
# ---------------------------------------------------------------------------
_shared_exporter = InMemorySpanExporter()
_shared_provider = TracerProvider()
_shared_provider.add_span_processor(SimpleSpanProcessor(_shared_exporter))
_provider_registered = False


@pytest.fixture
def memory_exporter():
    """Provide the shared InMemorySpanExporter, cleared before and after each test."""
    global _provider_registered
    reset_traceroot()
    if not _provider_registered:
        trace.set_tracer_provider(_shared_provider)
        _provider_registered = True
    _shared_exporter.clear()
    yield _shared_exporter
    _shared_exporter.clear()
    reset_traceroot()
