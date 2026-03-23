"""Integration tests for Integration.ANTHROPIC / AnthropicInstrumentor span emission.

Strategy:
- AnthropicInstrumentor monkey-patches anthropic SDK classes at the class level.
- A fresh TracerProvider + InMemorySpanExporter per fixture isolates these tests
  from test_otel.py's global OTel provider.
- respx intercepts the underlying httpx calls so no real API traffic is sent.

Requires: anthropic (added as a dev dependency in pyproject.toml).
"""

import pytest

anthropic = pytest.importorskip("anthropic")  # skip entire module if not installed

import respx  # noqa: E402
from httpx import Response  # noqa: E402
from openinference.instrumentation.anthropic import AnthropicInstrumentor  # noqa: E402
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

# ---------------------------------------------------------------------------
# Attribute key constants (matching what AnthropicInstrumentor actually emits)
# ---------------------------------------------------------------------------
OI_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND   # "openinference.span.kind"
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME          # "llm.model_name"
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER              # "llm.provider"
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT          # "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION  # "llm.token_count.completion"
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES  # "llm.output_messages"
LLM_TOOLS = SpanAttributes.LLM_TOOLS                    # "llm.tools"
INPUT_VALUE = SpanAttributes.INPUT_VALUE                 # "input.value"
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE               # "output.value"

LLM_KIND = OpenInferenceSpanKindValues.LLM.value         # "LLM"

# ---------------------------------------------------------------------------
# Fake API responses
# ---------------------------------------------------------------------------
_SIMPLE_RESPONSE = {
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Hello! How can I help you?"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 8,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    },
}


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def anthropic_exporter():
    """Fresh TracerProvider + InMemoryExporter with AnthropicInstrumentor applied."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=provider)

    yield exporter

    instrumentor.uninstrument()  # restore patched methods — critical for test isolation
    exporter.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_TOOL_RESPONSE = {
    "id": "msg_01aBcDeFgH",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "get_weather",
            "input": {"city": "San Francisco"},
        }
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 30,
        "output_tokens": 15,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    },
}

_TOOL_SCHEMA = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

# Minimal SSE streaming response for messages.create(stream=True)
_STREAM_EVENTS = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_stream01","type":"message",'
    '"role":"assistant","model":"claude-3-5-sonnet-20241022","content":[],'
    '"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}}\n\n'
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi!"}}\n\n'
    "event: content_block_stop\n"
    'data: {"type":"content_block_stop","index":0}\n\n'
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},'
    '"usage":{"output_tokens":3}}\n\n'
    "event: message_stop\n"
    'data: {"type":"message_stop"}\n\n'
)


@respx.mock
def test_basic_message_creates_span(anthropic_exporter):
    """messages.create emits exactly one span named 'messages.create' with kind=LLM."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json=_SIMPLE_RESPONSE)
    )

    client = anthropic.Anthropic(api_key="test-key")
    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )

    finished = anthropic_exporter.get_finished_spans()
    assert len(finished) == 1

    span = finished[0]
    assert span.name == "messages.create"
    assert span.attributes.get(OI_SPAN_KIND) == LLM_KIND
    assert span.attributes.get(LLM_PROVIDER) == "anthropic"


@respx.mock
def test_message_captures_model_and_tokens(anthropic_exporter):
    """Span attributes include the model name, token counts, and raw input/output values."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json=_SIMPLE_RESPONSE)
    )

    client = anthropic.Anthropic(api_key="test-key")
    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )

    span = anthropic_exporter.get_finished_spans()[0]
    assert span.attributes.get(LLM_MODEL_NAME) == "claude-3-5-sonnet-20241022"
    assert span.attributes.get(LLM_TOKEN_COUNT_PROMPT) == 10
    assert span.attributes.get(LLM_TOKEN_COUNT_COMPLETION) == 8
    assert "Hello" in span.attributes.get(INPUT_VALUE, "")
    assert "Hello! How can I help you?" in span.attributes.get(OUTPUT_VALUE, "")


@respx.mock
def test_tool_use_captured_in_span(anthropic_exporter):
    """Tool definitions and tool_use response blocks appear in span attributes."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json=_TOOL_RESPONSE)
    )

    client = anthropic.Anthropic(api_key="test-key")
    client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        tools=_TOOL_SCHEMA,
        messages=[{"role": "user", "content": "What's the weather in SF?"}],
    )

    span = anthropic_exporter.get_finished_spans()[0]

    # Tool schema should be captured on the input side
    tool_schema_key = f"{LLM_TOOLS}.0.tool.json_schema"
    assert span.attributes.get(tool_schema_key) is not None

    # Tool call name should appear in the output messages
    tool_name_key = f"{LLM_OUTPUT_MESSAGES}.0.message.tool_calls.0.tool_call.function.name"
    assert span.attributes.get(tool_name_key) == "get_weather"


@respx.mock
def test_streaming_message_creates_span(anthropic_exporter):
    """messages.create(stream=True) still emits a completed span after the stream closes."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            200,
            content=_STREAM_EVENTS.encode(),
            headers={"content-type": "text/event-stream"},
        )
    )

    client = anthropic.Anthropic(api_key="test-key")
    with client.messages.stream(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    ) as stream:
        stream.get_final_message()

    finished = anthropic_exporter.get_finished_spans()
    assert len(finished) == 1

    span = finished[0]
    assert span.name == "messages.stream"
    assert span.attributes.get(OI_SPAN_KIND) == LLM_KIND


@pytest.mark.asyncio
@respx.mock
async def test_async_message_creates_span(anthropic_exporter):
    """AsyncAnthropic.messages.create emits a span identical to the sync path."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(200, json=_SIMPLE_RESPONSE)
    )

    client = anthropic.AsyncAnthropic(api_key="test-key")
    await client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
    )

    finished = anthropic_exporter.get_finished_spans()
    assert len(finished) == 1

    span = finished[0]
    assert span.name == "messages.create"
    assert span.attributes.get(OI_SPAN_KIND) == LLM_KIND
    assert span.attributes.get(LLM_MODEL_NAME) == "claude-3-5-sonnet-20241022"
