"""Regression tests: ChatAnthropic calls counted by both LangChain and Anthropic's own tracing.

Each scenario runs in a fresh process, because tracer providers and instrumentation are process-wide.
"""

import asyncio
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("langchain_anthropic") is None,
    reason="langchain-anthropic not installed",
)

LANGCHAIN_SCOPE = "openinference.instrumentation.langchain"
NATIVE_SCOPE = "com.anthropic.sdk.python"
INPUT_TOKENS = 598
OUTPUT_TOKENS = 53


def run_scenario(name):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(
            ("ANTHROPIC_", "TRACEROOT_", "OTEL_", "OPENINFERENCE_", "LANGSMITH_", "LANGCHAIN_")
        )
    }
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), name],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])


TWO_CALL_SCENARIOS = [
    "invoke",
    "ainvoke",
    "stream",
    "astream",
    "invoke_streaming",
    "ainvoke_streaming",
    "batch",
    "generate",
]


@pytest.mark.parametrize("scenario", TWO_CALL_SCENARIOS)
def test_one_usage_per_chat_anthropic_call(scenario):
    data = run_scenario(scenario)
    assert data["requests"] == 2
    assert data["usage_scopes"] == [LANGCHAIN_SCOPE, LANGCHAIN_SCOPE]
    assert data["input_tokens"] == 2 * INPUT_TOKENS
    assert data["output_tokens"] == 2 * OUTPUT_TOKENS
    assert data["native_spans"] == 0
    # The surviving span keeps OpenInference I/O and user/session context.
    assert data["content_spans"] == 2
    assert data["context_spans"] == 2


@pytest.mark.parametrize("scenario", ["stream_without_usage", "astream_without_usage"])
def test_stream_without_usage_keeps_the_native_span(scenario):
    # With stream_usage=False the LangChain span has no token counts; the native span is the record.
    data = run_scenario(scenario)
    assert data["requests"] == 1
    assert data["usage_scopes"] == [NATIVE_SCOPE]
    assert data["input_tokens"] == INPUT_TOKENS
    assert data["output_tokens"] == OUTPUT_TOKENS


def test_failure_to_wrap_chat_anthropic_does_not_break_the_import(caplog):
    from traceroot.instrumentation import langchain

    with patch.object(langchain, "wrap_function_wrapper", side_effect=AttributeError("_create")):
        langchain._wrap_chat_anthropic(None)
    assert "ANTHROPIC_OPEN_TELEMETRY=false" in caplog.text


def test_call_in_langgraph_node_counts_once():
    data = run_scenario("langgraph")
    assert data["requests"] == 1
    assert data["usage_scopes"] == [LANGCHAIN_SCOPE]
    assert data["native_spans"] == 0


def test_raw_call_between_stream_chunks_is_not_mistaken_for_the_stream():
    data = run_scenario("raw_call_between_chunks")
    assert data["requests"] == 2
    assert sorted(data["usage_scopes"]) == [NATIVE_SCOPE, LANGCHAIN_SCOPE]


def test_native_usage_kept_when_the_langchain_span_is_not_recording():
    data = run_scenario("langchain_not_recording")
    assert data["requests"] == 1
    assert data["usage_scopes"] == [NATIVE_SCOPE]


def test_host_exporters_still_see_the_native_span():
    # On a host provider TraceRoot only filters its own export.
    data = run_scenario("host_provider")
    assert data["usage_scopes"] == [LANGCHAIN_SCOPE]
    assert data["native_spans"] == 0
    assert data["host_native_spans"] == 1


def test_model_built_before_initialize_counts_once():
    data = run_scenario("model_before_initialize")
    assert data["requests"] == 1
    assert data["usage_scopes"] == [LANGCHAIN_SCOPE]
    assert data["native_spans"] == 0


def test_chat_anthropic_imported_after_initialize_is_covered():
    data = run_scenario("import_after_initialize")
    assert data["requests"] == 1
    assert data["usage_scopes"] == [LANGCHAIN_SCOPE]
    assert data["native_spans"] == 0


# --- Runs in the fresh process --------------------------------------------------------------------

MODEL = "claude-sonnet-4-6"
MESSAGE = {
    "id": "msg_offline",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "offline mock"}],
    "model": MODEL,
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS},
}
EVENTS = [
    {
        "type": "message_start",
        "message": {
            **MESSAGE,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": 0},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "offline mock"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        # ChatAnthropic reads streamed usage from here, input tokens included.
        "usage": {"input_tokens": INPUT_TOKENS, "output_tokens": OUTPUT_TOKENS},
    },
    {"type": "message_stop"},
]
SSE_BODY = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in EVENTS)
RAW_CALL = {
    "model": MODEL,
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "offline mock request"}],
}


class Offline:
    """Anthropic clients and ChatAnthropic models on a mocked transport."""

    def __init__(self):
        self.requests = []

    def respond(self, request, *, is_async=False):
        import httpx2

        self.requests.append(request)
        if json.loads(request.content).get("stream"):
            # An unread body, as from the network: Anthropic's span records a stream's usage only
            # while it is read, so a body that is already in memory would leave the usage out.
            async def chunks():
                yield SSE_BODY.encode()

            body = chunks() if is_async else iter([SSE_BODY.encode()])
            return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=MESSAGE)

    def client(self):
        import anthropic
        import httpx2

        return anthropic.Anthropic(
            api_key="offline-placeholder",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.respond)),
        )

    def async_client(self):
        import anthropic
        import httpx2

        return anthropic.AsyncAnthropic(
            api_key="offline-placeholder",
            max_retries=0,
            http_client=httpx2.AsyncClient(
                transport=httpx2.MockTransport(lambda request: self.respond(request, is_async=True))
            ),
        )

    def chat(self, **options):
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model=MODEL, api_key="offline-placeholder", max_tokens=100, **options)
        # Clients are made with the model: before initialize() in model_before_initialize.
        chat.__dict__["_client"] = self.client()
        chat.__dict__["_async_client"] = self.async_client()
        return chat


def summarize(offline, exporter, **extra):
    spans = exporter.get_finished_spans()

    def tokens(span, *keys):
        return next((span.attributes[k] for k in keys if k in span.attributes), 0)

    usage = [
        s
        for s in spans
        if any(k in s.attributes for k in ("llm.token_count.prompt", "gen_ai.usage.input_tokens"))
    ]
    return {
        "requests": len(offline.requests),
        "usage_scopes": [s.instrumentation_scope.name for s in usage],
        "input_tokens": sum(
            tokens(s, "llm.token_count.prompt", "gen_ai.usage.input_tokens") for s in usage
        ),
        "output_tokens": sum(
            tokens(s, "llm.token_count.completion", "gen_ai.usage.output_tokens") for s in usage
        ),
        "native_spans": sum(s.instrumentation_scope.name.startswith(NATIVE_SCOPE) for s in spans),
        "content_spans": sum(
            "input.value" in s.attributes and "output.value" in s.attributes for s in usage
        ),
        "context_spans": sum(
            s.attributes.get("user.id") == "offline-user"
            and s.attributes.get("session.id") == "offline-session"
            for s in usage
        ),
        **extra,
    }


def twice(call):
    """Two calls with identical usage, flushed in between to catch batch-dependent deduplication."""
    import traceroot

    call()
    traceroot.flush()
    call()


def run_two_calls(scenario, offline):
    # Only when set: an explicit streaming=False turns stream() into invoke().
    chat = offline.chat(**({"streaming": True} if scenario.endswith("_streaming") else {}))

    def drain():
        return list(chat.stream("hi"))

    async def adrain():
        return [chunk async for chunk in chat.astream("hi")]

    if scenario.startswith("ainvoke"):
        asyncio.run(_gather(chat.ainvoke("hi"), chat.ainvoke("hi")))
    elif scenario == "batch":
        chat.batch(["hi", "hi"])
    elif scenario == "generate":
        from langchain_core.messages import HumanMessage

        chat.generate([[HumanMessage("hi")], [HumanMessage("hi")]])
    elif scenario == "astream":
        asyncio.run(_gather(adrain(), adrain()))
    elif scenario == "stream":
        twice(drain)
    else:
        twice(lambda: chat.invoke("hi"))


async def _gather(*calls):
    await asyncio.gather(*calls)


def run_langgraph(offline):
    from langgraph.graph import END, START, MessagesState, StateGraph

    chat = offline.chat()
    graph = StateGraph(MessagesState)
    graph.add_node("model", lambda state: {"messages": [chat.invoke(state["messages"])]})
    graph.add_edge(START, "model")
    graph.add_edge("model", END)
    graph.compile().invoke({"messages": [("user", "hi")]})


def run_mock(scenario):
    from openinference.instrumentation import using_attributes
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import traceroot

    exporter = InMemorySpanExporter()
    host_exporter = InMemorySpanExporter()
    offline = Offline()
    extra = {}
    if scenario == "host_provider":
        host_provider = TracerProvider()
        host_provider.add_span_processor(SimpleSpanProcessor(host_exporter))
        trace.set_tracer_provider(host_provider)
    if scenario != "import_after_initialize":
        import langchain_anthropic  # noqa: F401
    # An app that builds its model, and so its clients, at import time.
    early_chat = offline.chat() if scenario == "model_before_initialize" else None
    # Unlike test_anthropic_usage.py, no socket.connect guard: on Windows asyncio's event loop
    # connects a socket pair at startup, so the guard would fail every async scenario. Every
    # client here uses a MockTransport, and the exporter is in memory.
    with patch("traceroot.transport.span_processor.OTLPSpanExporter", return_value=exporter):
        traceroot.initialize(
            api_key="offline-placeholder",
            host_url="http://offline.invalid",
            git_repo="offline/mock",
            git_ref="offline",
            integrations=[traceroot.Integration.LANGCHAIN],
        )
        if scenario == "langchain_not_recording":
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

            LangChainInstrumentor().uninstrument()
            LangChainInstrumentor().instrument(tracer_provider=TracerProvider(sampler=ALWAYS_OFF))

        with using_attributes(user_id="offline-user", session_id="offline-session"):
            if scenario in TWO_CALL_SCENARIOS:
                run_two_calls(scenario, offline)
            elif scenario in (
                "host_provider",
                "import_after_initialize",
                "langchain_not_recording",
            ):
                offline.chat().invoke("hi")
            elif scenario == "model_before_initialize":
                early_chat.invoke("hi")
            elif scenario == "langgraph":
                run_langgraph(offline)
            elif scenario == "stream_without_usage":
                list(offline.chat(stream_usage=False).stream("hi"))
            elif scenario == "astream_without_usage":
                chat = offline.chat(stream_usage=False)

                async def adrain():
                    return [chunk async for chunk in chat.astream("hi")]

                asyncio.run(adrain())
            elif scenario == "raw_call_between_chunks":
                client = offline.client()
                for index, _ in enumerate(offline.chat().stream("hi")):
                    if index == 0:
                        client.messages.create(**RAW_CALL)
            else:
                raise AssertionError(f"unknown scenario {scenario}")

        traceroot.flush()
        if scenario == "host_provider":
            extra["host_native_spans"] = sum(
                s.instrumentation_scope.name.startswith(NATIVE_SCOPE)
                for s in host_exporter.get_finished_spans()
            )
        print(json.dumps(summarize(offline, exporter, **extra)))
        traceroot.shutdown()


if __name__ == "__main__":
    run_mock(sys.argv[1])
