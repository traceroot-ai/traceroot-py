"""Offline, fresh-process regressions for native Anthropic / OpenInference ownership.

Each case uses a fresh process because OTel providers and instrumentor wrappers
are process-wide. Responses come from MockTransport and spans stay in memory.
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


@pytest.mark.skipif(importlib.util.find_spec("anthropic") is None, reason="anthropic not installed")
@pytest.mark.parametrize("mode", ["both", "native_only", "native_fallback", "openinference_only"])
@pytest.mark.parametrize("api", ["sync", "async"])
@pytest.mark.parametrize("style", ["create", "create_stream", "stream"])
@pytest.mark.parametrize("beta", [False, True])
def test_one_usage_per_response(mode, api, style, beta):
    # Global providers and instrumentor singletons cannot be reset reliably in
    # process. Also reproduce clients constructed BEFORE initialize().
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ANTHROPIC_", "TRACEROOT_", "OTEL_", "OPENINFERENCE_"))
    }
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), mode, api, style, str(int(beta))],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(result.stdout)
    assert data["requests"] == 2
    assert data["usage_spans"] == 2
    assert data["input_tokens"] == 1196
    assert data["output_tokens"] == 106
    assert data["llm_spans"] == 2
    native = mode in ("native_only", "native_fallback")
    assert data["content_spans"] == (0 if native else 2)
    scope = "com.anthropic.sdk.python" if native else "openinference.instrumentation.anthropic"
    assert data["scopes"] == [scope, scope]


def run_mock(mode, api, style, beta):
    import anthropic
    import httpx2
    from openinference.instrumentation import using_attributes
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import traceroot

    requests = []
    model = "claude-sonnet-4-6"
    message = {
        "id": "msg_offline",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "offline mock"}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 598, "output_tokens": 53},
    }
    events = [
        {
            "type": "message_start",
            "message": {
                **message,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 598, "output_tokens": 0},
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
            "usage": {"output_tokens": 53},
        },
        {"type": "message_stop"},
    ]
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)

    def respond(request):
        requests.append(request)
        if json.loads(request.content).get("stream"):
            return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=message)

    http_client = (
        httpx2.Client(transport=httpx2.MockTransport(respond))
        if api == "sync"
        else httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
    )
    client_cls = anthropic.Anthropic if api == "sync" else anthropic.AsyncAnthropic
    options = {"open_telemetry": False} if mode == "openinference_only" else {}
    exporter = InMemorySpanExporter()
    native_provider = None
    if mode == "native_fallback":
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON

        native_provider = TracerProvider(sampler=ALWAYS_ON)
        native_provider.add_span_processor(SimpleSpanProcessor(exporter))
        options["open_telemetry"] = {"tracer_provider": native_provider}
    client = client_cls(api_key="offline-placeholder", http_client=http_client, **options)
    # Force the cached resource to exist before instrumentation too.
    resource = client.beta.messages if beta else client.messages
    with (
        patch("traceroot.transport.span_processor.OTLPSpanExporter", return_value=exporter),
        patch("socket.socket.connect", side_effect=AssertionError("Unexpected network access")),
        patch("socket.socket.connect_ex", side_effect=AssertionError("Unexpected network access")),
    ):
        traceroot.initialize(
            api_key="offline-placeholder",
            host_url="http://offline.invalid",
            git_repo="offline/mock",
            git_ref="offline",
            integrations=[]
            if mode in ("native_only", "native_fallback")
            else [traceroot.Integration.ANTHROPIC],
        )
        if mode == "native_fallback":
            # If OI's provider declines to record, the native call still owns
            # its legitimate usage on its explicitly configured provider.
            from openinference.instrumentation.anthropic import AnthropicInstrumentor
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

            AnthropicInstrumentor().instrument(tracer_provider=TracerProvider(sampler=ALWAYS_OFF))
        kwargs = {
            "model": model,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "offline mock request"}],
        }

        def sync_call():
            if style == "create":
                assert resource.create(**kwargs).content[0].text == "offline mock"
            elif style == "create_stream":
                with resource.create(**kwargs, stream=True) as stream:
                    assert list(stream)[-1].type == "message_stop"
            else:
                with resource.stream(**kwargs) as stream:
                    assert stream.get_final_message().content[0].text == "offline mock"

        async def async_call():
            if style == "create":
                assert (await resource.create(**kwargs)).content[0].text == "offline mock"
            elif style == "create_stream":
                async with await resource.create(**kwargs, stream=True) as stream:
                    assert [event async for event in stream][-1].type == "message_stop"
            else:
                async with resource.stream(**kwargs) as stream:
                    assert (await stream.get_final_message()).content[0].text == "offline mock"

        async def run_async():
            await asyncio.gather(async_call(), async_call())
            await client.close()

        with (
            trace.get_tracer("offline.test").start_as_current_span("two_equal_usage_calls"),
            using_attributes(user_id="offline-user", session_id="offline-session"),
        ):
            if api == "sync":
                sync_call()
                # Export the first call before the second exists: correctness
                # must not rely on matching spans within one export batch.
                traceroot.flush()
                first_call_spans = [
                    s for s in exporter.get_finished_spans() if s.name != "two_equal_usage_calls"
                ]
                assert len(first_call_spans) == 1, [s.name for s in first_call_spans]
                sync_call()
                client.close()
            else:
                asyncio.run(run_async())
        traceroot.flush()
        spans = exporter.get_finished_spans()
        usage = [
            s
            for s in spans
            if any(
                k in s.attributes for k in ("llm.token_count.prompt", "gen_ai.usage.input_tokens")
            )
        ]
        oi = [
            s
            for s in usage
            if s.instrumentation_scope.name == "openinference.instrumentation.anthropic"
        ]
        for span in oi:
            assert span.attributes.get("user.id") == "offline-user"
            assert span.attributes.get("session.id") == "offline-session"
        result = {
            "requests": len(requests),
            "usage_spans": len(usage),
            "llm_spans": sum(s.name != "two_equal_usage_calls" for s in spans),
            "input_tokens": sum(
                s.attributes.get(
                    "llm.token_count.prompt", s.attributes.get("gen_ai.usage.input_tokens", 0)
                )
                for s in usage
            ),
            "output_tokens": sum(
                s.attributes.get(
                    "llm.token_count.completion", s.attributes.get("gen_ai.usage.output_tokens", 0)
                )
                for s in usage
            ),
            "content_spans": sum(
                "input.value" in s.attributes and "output.value" in s.attributes for s in usage
            ),
            "scopes": [s.instrumentation_scope.name for s in usage],
        }
        print(json.dumps(result))
        if native_provider is not None:
            native_provider.shutdown()
        traceroot.shutdown()


if __name__ == "__main__":
    run_mock(sys.argv[1], sys.argv[2], sys.argv[3], bool(int(sys.argv[4])))
