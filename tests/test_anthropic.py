"""End-to-end tests for Integration.ANTHROPIC: a real anthropic client behind a fake HTTP transport.

The `memory_exporter` fixture installs the global provider that initialize() joins, so the
instrumentor's spans and the @observe spans land in the same exporter.
"""

import httpx2
import pytest
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

import traceroot
from traceroot import Integration, observe

anthropic = pytest.importorskip("anthropic")
from openinference.instrumentation.anthropic import AnthropicInstrumentor  # noqa: E402

MODEL = "claude-opus-4-5"
REPLY = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "Paris"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 3},
}


@pytest.fixture
def instrumented(monkeypatch, memory_exporter):
    """initialize() with the Anthropic integration; nothing leaves the process."""
    monkeypatch.setattr(OTLPSpanExporter, "export", lambda self, spans: SpanExportResult.SUCCESS)
    traceroot.initialize(
        api_key="test-key", host_url="http://127.0.0.1:9", integrations=[Integration.ANTHROPIC]
    )
    yield memory_exporter
    AnthropicInstrumentor().uninstrument()


def _ask():
    transport = httpx2.MockTransport(lambda request: httpx2.Response(200, json=REPLY))
    client = anthropic.Anthropic(api_key="fake", http_client=httpx2.Client(transport=transport))

    @observe(name="solve")
    def solve():
        return client.messages.create(
            model=MODEL,
            max_tokens=64,
            messages=[{"role": "user", "content": "Capital of France?"}],
        )

    solve()


def _spans_by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def test_llm_span_nests_under_observe(instrumented):
    """The model call must sit under the function that made it, or the trace tells the wrong story."""
    _ask()

    spans = _spans_by_name(instrumented)
    llm = spans["messages.create"]
    assert llm.parent.span_id == spans["solve"].context.span_id


def test_llm_span_records_model_and_tokens(instrumented):
    """Cost is computed from these attributes, so a span without them has no cost."""
    _ask()

    llm = _spans_by_name(instrumented)["messages.create"]
    assert {
        key: llm.attributes.get(key)
        for key in ("llm.model_name", "llm.token_count.prompt", "llm.token_count.completion")
    } == {
        "llm.model_name": MODEL,
        "llm.token_count.prompt": 12,
        "llm.token_count.completion": 3,
    }
