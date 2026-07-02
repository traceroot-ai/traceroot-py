from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot.instrumentation.livekit import LiveKitToOpenInferenceProcessor


def _provider_with_livekit_processor():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(LiveKitToOpenInferenceProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_livekit_agent_session_maps_to_agent_kind():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("agent_session") as span:
        span.set_attribute("lk.room_name", "room-123")
        span.set_attribute("lk.agent_name", "support-agent")

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "AGENT"
    assert span.attributes.get("lk.room_name") == "room-123"
    assert span.attributes.get("lk.agent_name") == "support-agent"


def test_livekit_llm_request_maps_model_tokens_and_io():
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
    assert span.attributes.get("input.value") == "user: hello"
    assert span.attributes.get("output.value") == "assistant: hi"
    assert span.attributes.get("llm.model_name") == "openai/gpt-5.2-chat-latest"
    assert span.attributes.get("llm.token_count.prompt") == 11
    assert span.attributes.get("llm.token_count.completion") == 7


def test_livekit_function_tool_maps_tool_kind_name_and_output():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("function_tool") as span:
        span.set_attribute("lk.function_tool.name", "lookup_customer")
        span.set_attribute("lk.function_tool.arguments", '{"user_id":"user-123"}')
        span.set_attribute("lk.function_tool.output", '{"plan":"pro"}')

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "TOOL"
    assert span.attributes.get("gen_ai.tool.name") == "lookup_customer"
    assert span.attributes.get("input.value") == '{"user_id":"user-123"}'
    assert span.attributes.get("output.value") == '{"plan":"pro"}'


def test_livekit_lifecycle_span_maps_to_chain_kind():
    provider, exporter = _provider_with_livekit_processor()
    tracer = provider.get_tracer("livekit-agents")

    with tracer.start_as_current_span("tts_request"):
        pass

    [span] = exporter.get_finished_spans()
    assert span.attributes.get("openinference.span.kind") == "CHAIN"
