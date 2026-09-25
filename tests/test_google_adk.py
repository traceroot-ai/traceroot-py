"""Tests for the Google ADK integration's ADK 2.x workflow/node spans.

End-to-end tests run a real ADK Workflow with a fake LLM in a fresh subprocess each, because the
behavior depends on whether ``node_tracing`` is first imported before or after instrumentation.
"""

import importlib.util
import json
import os
import subprocess
import sys

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider

from traceroot.constants import SDK_NAME
from traceroot.instrumentation.google_adk import (
    _NodeSpanTracer,
    _should_replace,
    instrument_adk_node_spans,
)

requires_adk = pytest.mark.skipif(
    importlib.util.find_spec("google.adk") is None, reason="google-adk is not installed"
)

_PRELUDE = """
import asyncio, json, sys
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SpanExportResult

EXPORTED = []

def _capture(self, spans):
    EXPORTED.extend(spans)
    return SpanExportResult.SUCCESS

OTLPSpanExporter.export = _capture
"""

_WORKFLOW = """
from typing import AsyncGenerator
from google.adk import Agent, Event, Workflow
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types


class FakeLlm(BaseLlm):
    model: str = "fake"
    calls: int = 0

    async def generate_content_async(self, llm_request, stream=False) -> AsyncGenerator[LlmResponse, None]:
        self.calls += 1
        if self.calls == 1:
            part = types.Part(function_call=types.FunctionCall(name="lookup", args={"q": "x"}))
        else:
            part = types.Part(text="final answer")
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def lookup(q: str) -> str:
    \"\"\"Look something up.\"\"\"
    return observed_lookup(q)


def a(node_input: str):
    return "x"


def b(node_input: str):
    return "y"


def c(node_input):
    return Event(message="done")


agent = Agent(name="helper", model=FakeLlm(), tools=[lookup], instruction="help")
inner = Workflow(name="inner_wf", edges=[("START", b)])
root = Workflow(name="probe_wf", edges=[("START", a, inner, agent, c)])


async def _run():
    runner = InMemoryRunner(node=root, app_name="probe")
    session = await runner.session_service.create_session(app_name="probe", user_id="u")
    message = types.Content(role="user", parts=[types.Part(text="go")])
    with traceroot.using_attributes(session_id="adk-test-session"):
        async for _ in runner.run_async(user_id="u", session_id=session.id, new_message=message):
            pass


asyncio.run(_run())
traceroot.flush()
by_id = {s.context.span_id: s for s in EXPORTED}
print(json.dumps([
    {
        "name": s.name,
        "kind": s.attributes.get("openinference.span.kind"),
        "parent": by_id[s.parent.span_id].name if s.parent and s.parent.span_id in by_id else None,
        "trace_id": s.context.trace_id,
        "session_id": s.attributes.get("session.id"),
        "traceroot_sdk": s.attributes.get("traceroot.sdk.name"),
    }
    for s in EXPORTED
]))
"""

_INIT = """
import traceroot
from traceroot import Integration, observe

traceroot.initialize(
    api_key="test-key", host_url="http://127.0.0.1:9", integrations=[Integration.GOOGLE_ADK]
)


@observe(name="observed_lookup", type="tool")
def observed_lookup(q):
    return "found " + q
"""

# `adk web` / `adk api_server` shape: the host installs a global SDK provider and loads the ADK
# runtime before initialize(); node_tracing is still imported after. TraceRoot must join the host
# provider, and the host must see the same spans.
_HOST_PROVIDER = """
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

HOST_EXPORTER = InMemorySpanExporter()
HOST_PROVIDER = TracerProvider()
HOST_PROVIDER.add_span_processor(SimpleSpanProcessor(HOST_EXPORTER))
trace.set_tracer_provider(HOST_PROVIDER)
import google.adk.runners
"""
_HOST_CHECK = """
assert trace.get_tracer_provider() is HOST_PROVIDER
assert traceroot.get_client()._provider is HOST_PROVIDER
host_names = sorted(s.name for s in HOST_EXPORTER.get_finished_spans())
assert host_names == sorted(s.name for s in EXPORTED), host_names
"""

_NODE_TRACING_FIRST = "import google.adk.telemetry.node_tracing\n"

# Each order reaches a different binding in _should_replace: OpenInference's proxy (initialize
# first), ADK's raw tracer (node_tracing first), and the proxy on a joined host provider.
_SCRIPTS = {
    "initialize_first": _PRELUDE + _INIT + _WORKFLOW,
    "node_tracing_first": _PRELUDE + _NODE_TRACING_FIRST + _INIT + _WORKFLOW,
    "host_provider_adk_runtime_first": _PRELUDE + _HOST_PROVIDER + _INIT + _WORKFLOW + _HOST_CHECK,
}


def _run_script(script: str) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("TRACEROOT_", "OTEL_"))}
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


# =============================================================================
# End-to-end: a real ADK 2.x Workflow, in every import order
# =============================================================================


@requires_adk
@pytest.mark.parametrize("order", sorted(_SCRIPTS))
def test_workflow_and_node_spans_are_chain_spans_nested_once(order):
    spans = _run_script(_SCRIPTS[order])
    names = [s["name"] for s in spans]
    by_name = {s["name"]: s for s in spans}

    # Exactly once each: ADK's native agent/LLM spans must not duplicate OpenInference's.
    assert sorted(names) == sorted(
        [
            "invocation [probe]",
            "invoke_workflow probe_wf",
            "invoke_node a",
            "invoke_workflow inner_wf",
            "invoke_node b",
            "agent_run [helper]",
            "call_llm",
            "call_llm",
            "execute_tool lookup",
            "observed_lookup",
            "invoke_node c",
        ]
    ), names
    assert not [n for n in names if n.startswith(("invoke_agent", "generate_content"))]

    for name in (
        "invoke_workflow probe_wf",
        "invoke_workflow inner_wf",
        "invoke_node a",
        "invoke_node b",
        "invoke_node c",
    ):
        assert by_name[name]["kind"] == "CHAIN", name

    parents = {s["name"]: s["parent"] for s in spans}
    assert parents["invocation [probe]"] is None
    assert parents["invoke_workflow probe_wf"] == "invocation [probe]"
    for child in (
        "invoke_node a",
        "invoke_workflow inner_wf",
        "agent_run [helper]",
        "invoke_node c",
    ):
        assert parents[child] == "invoke_workflow probe_wf", child
    assert parents["invoke_node b"] == "invoke_workflow inner_wf"
    assert parents["execute_tool lookup"] == "call_llm"
    assert parents["observed_lookup"] == "execute_tool lookup"

    assert len({s["trace_id"] for s in spans}) == 1
    assert {s["session_id"] for s in spans} == {"adk-test-session"}
    assert {s["traceroot_sdk"] for s in spans} == {SDK_NAME}


@requires_adk
def test_repeat_instrument_and_uninstrument_do_not_stack_wrappers():
    from google.adk.telemetry import node_tracing

    from traceroot.instrumentation._instrumentors import GoogleADKInstrumentor

    original = node_tracing.tracer
    provider = TracerProvider()
    try:
        GoogleADKInstrumentor().instrument(tracer_provider=provider)
        ours = node_tracing.tracer
        assert isinstance(ours, _NodeSpanTracer)
        GoogleADKInstrumentor().instrument(tracer_provider=provider)
        assert node_tracing.tracer is ours
        assert ours.wrapped is original
    finally:
        GoogleADKInstrumentor().uninstrument()
    assert node_tracing.tracer is original


# =============================================================================
# Deciding whether to take over node_tracing.tracer
# =============================================================================


def test_should_replace_only_tracers_that_lose_node_spans():
    from openinference.instrumentation.google_adk import _SelectiveExecuteToolTracer

    assert _should_replace(TracerProvider().get_tracer("gcp.vertex.agent")) is True
    assert _should_replace(trace_api.ProxyTracer("gcp.vertex.agent")) is True
    proxy = _SelectiveExecuteToolTracer(
        TracerProvider().get_tracer("adk"), TracerProvider().get_tracer("oi")
    )
    assert _should_replace(proxy) is True
    assert _should_replace(trace_api.NoOpTracer()) is False
    assert _should_replace(object()) is False


def test_defers_to_openinference_once_it_emits_node_spans(monkeypatch):
    """If OpenInference starts forwarding invoke_node itself, don't emit a second copy."""
    from openinference.instrumentation.google_adk import _SelectiveExecuteToolTracer

    def forward_everything(self, name, *args, **kwargs):
        return self._self_oi_tracer.start_as_current_span(name, *args, **kwargs)

    monkeypatch.setattr(_SelectiveExecuteToolTracer, "start_as_current_span", forward_everything)
    proxy = _SelectiveExecuteToolTracer(
        TracerProvider().get_tracer("adk"), TracerProvider().get_tracer("oi")
    )
    assert _should_replace(proxy) is False


def test_node_span_tracer_delegates_other_span_names():
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    adk_exporter, oi_exporter = InMemorySpanExporter(), InMemorySpanExporter()
    adk_provider, oi_provider = TracerProvider(), TracerProvider()
    adk_provider.add_span_processor(SimpleSpanProcessor(adk_exporter))
    oi_provider.add_span_processor(SimpleSpanProcessor(oi_exporter))
    tracer = _NodeSpanTracer(adk_provider.get_tracer("adk"), oi_provider.get_tracer("oi"))

    with (
        tracer.start_as_current_span(name="invoke_workflow wf", attributes={"k": "v"}),
        tracer.start_as_current_span("invoke_node n"),
    ):
        pass
    with tracer.start_as_current_span("invoke_workflow"):
        pass
    with tracer.start_as_current_span("send_data"):
        pass

    oi_spans = {s.name: s for s in oi_exporter.get_finished_spans()}
    assert set(oi_spans) == {"invoke_workflow wf", "invoke_node n", "invoke_workflow"}
    assert all(s.attributes["openinference.span.kind"] == "CHAIN" for s in oi_spans.values())
    assert oi_spans["invoke_workflow wf"].attributes["k"] == "v"
    assert (
        oi_spans["invoke_node n"].parent.span_id == oi_spans["invoke_workflow wf"].context.span_id
    )
    assert [s.name for s in adk_exporter.get_finished_spans()] == ["send_data"]


def test_no_op_without_adk_node_tracing(monkeypatch):
    """google-adk 1.x has no workflow runtime (no telemetry.node_tracing): nothing to patch."""
    monkeypatch.setitem(sys.modules, "google.adk.telemetry", None)
    assert instrument_adk_node_spans(TracerProvider()) is False
