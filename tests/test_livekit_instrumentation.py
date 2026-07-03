from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot.instrumentation.livekit import LiveKitSpanProcessor


def _provider_with_livekit_processor():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(LiveKitSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_livekit_agent_session_remains_generic_span():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("agent_session") as span:
        span.set_attribute("lk.room_name", "room-123")
        span.set_attribute("lk.agent_name", "support-agent")

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "SPAN"
    assert span.attributes.get("traceroot.span.type") == "span"
    assert span.attributes.get("lk.room_name") == "room-123"
    assert span.attributes.get("lk.agent_name") == "support-agent"


def test_livekit_agent_turn_maps_to_agent_kind_and_io():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("agent_turn") as span:
        span.set_attribute("lk.user_input", "Tell me a joke.")
        span.set_attribute("lk.response.text", "Why did the test pass?")

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "AGENT"
    assert span.attributes.get("traceroot.span.type") == "agent"
    assert span.attributes.get("input.value") == "Tell me a joke."
    assert span.attributes.get("output.value") == "Why did the test pass?"


def test_livekit_llm_request_maps_only_llm_model_tokens_and_io():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("llm_request") as span:
        span.set_attribute("lk.chat_ctx", "user: hello")
        span.set_attribute("lk.response.text", "assistant: hi")
        span.set_attribute("gen_ai.request.model", "openai/gpt-5.2-chat-latest")
        span.set_attribute("gen_ai.usage.input_tokens", 11)
        span.set_attribute("gen_ai.usage.output_tokens", 7)

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "LLM"
    assert span.attributes.get("traceroot.span.type") == "llm"
    assert span.attributes.get("input.value") == "user: hello"
    assert span.attributes.get("output.value") == "assistant: hi"
    assert span.attributes.get("llm.model_name") == "openai/gpt-5.2-chat-latest"
    assert span.attributes.get("llm.token_count.prompt") == 11
    assert span.attributes.get("llm.token_count.completion") == 7


def test_livekit_non_llm_model_spans_remain_generic():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("user_turn") as span:
        span.set_attribute("gen_ai.request.model", "deepgram/nova-3")
        span.set_attribute("lk.user_transcript", "Hello?")
        span.set_attribute("lk.transcript_confidence", 0.99)

    with tracer.start_as_current_span("tts_node") as span:
        span.set_attribute("gen_ai.request.model", "cartesia/sonic-3")
        span.set_attribute("lk.response.ttfb", 0.25)

    spans = exporter.get_finished_spans()
    assert [span.name for span in spans] == ["user_turn", "tts_node"]
    assert [span.attributes.get("openinference.span.kind") for span in spans] == ["SPAN", "SPAN"]
    assert [span.attributes.get("traceroot.span.type") for span in spans] == ["span", "span"]
    assert spans[0].attributes.get("input.value") == "Hello?"
    assert spans[0].attributes.get("lk.transcript_confidence") == 0.99
    assert spans[1].attributes.get("lk.response.ttfb") == 0.25


def test_livekit_llm_node_remains_generic_even_with_model_and_tokens():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("llm_node") as span:
        span.set_attribute("gen_ai.request.model", "openai/chat-latest")
        span.set_attribute("gen_ai.usage.input_tokens", 12)
        span.set_attribute("gen_ai.usage.output_tokens", 3)
        span.set_attribute("lk.chat_ctx", "user: add 12 and 30")
        span.set_attribute("lk.response.text", "tool call pending")

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "SPAN"
    assert span.attributes.get("traceroot.span.type") == "span"
    assert span.attributes.get("input.value") == "user: add 12 and 30"
    assert span.attributes.get("output.value") == "tool call pending"
    assert span.attributes.get("llm.model_name") is None
    assert span.attributes.get("llm.token_count.prompt") is None
    assert span.attributes.get("llm.token_count.completion") is None


def test_livekit_late_attributes_are_mirrored_while_span_is_writable():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("agent_turn") as span:
        span.set_attribute("lk.response.text", "assistant: ready")
        assert span.attributes.get("output.value") == "assistant: ready"

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("output.value") == "assistant: ready"


def test_livekit_processor_ignores_immutable_readable_span_on_end():
    class ImmutableAttributes(dict):
        def __setitem__(self, key, value):
            raise TypeError

    class ReadableSpanLike:
        def __init__(self):
            self.name = "agent_session"
            self.attributes = {"lk.response.text": "assistant: ready"}
            self._attributes = ImmutableAttributes(self.attributes)

    processor = LiveKitSpanProcessor()
    processor.on_end(ReadableSpanLike())


def test_livekit_function_tool_maps_tool_kind_name_and_output():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("function_tool") as span:
        span.set_attribute("lk.function_tool.name", "lookup_customer")
        span.set_attribute("lk.function_tool.arguments", '{"user_id":"user-123"}')
        span.set_attribute("lk.function_tool.output", '{"plan":"pro"}')

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "TOOL"
    assert span.attributes.get("traceroot.span.type") == "tool"
    assert span.attributes.get("gen_ai.tool.name") == "lookup_customer"
    assert span.attributes.get("input.value") == '{"user_id":"user-123"}'
    assert span.attributes.get("output.value") == '{"plan":"pro"}'


def test_livekit_lifecycle_span_maps_to_generic_span():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("tts_request"):
        pass

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "SPAN"
    assert span.attributes.get("traceroot.span.type") == "span"
