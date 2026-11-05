"""Manual tracing inside LangChain / LangGraph runs traced by OpenInference.

OpenInference's LangChain instrumentor never makes a run's span the current OpenTelemetry span, so
code inside a LangGraph node used to see no current span: ``@observe`` started a separate trace
(without the run's session), ``update_current_span`` / ``update_current_trace`` did nothing, and the
current trace/span ids were ``None``. With an ``@observe`` entrypoint around the graph, they hit the
entrypoint span instead of the node.

These tests run real langchain-core / LangGraph runs under the real instrumentor.
"""

import asyncio
import importlib.util
import time

import pytest
from opentelemetry import trace

import traceroot
from traceroot import observe
from traceroot.span_attributes import SpanAttributes

requires_langchain = pytest.mark.skipif(
    importlib.util.find_spec("langchain_core") is None, reason="langchain-core is not installed"
)
requires_langgraph = pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is None, reason="langgraph is not installed"
)

pytestmark = requires_langchain


@pytest.fixture
def lc_spans(memory_exporter):
    """Instrument LangChain on the shared test provider; yield the exporter."""
    from openinference.instrumentation.langchain import LangChainInstrumentor

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(tracer_provider=trace.get_tracer_provider())
    try:
        yield memory_exporter
    finally:
        instrumentor.uninstrument()


def _by_name(exporter) -> dict:
    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    assert len(names) == len(set(names)), f"duplicate span names: {names}"
    return {s.name: s for s in spans}


def _parent_of(span, spans: dict):
    for candidate in spans.values():
        if span.parent is not None and candidate.context.span_id == span.parent.span_id:
            return candidate
    return None


def _graph(node):
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(dict)
    g.add_node("work", node)
    g.add_edge(START, "work")
    g.add_edge("work", END)
    return g.compile()


THREAD = {"configurable": {"thread_id": "thread-1"}}


@observe(name="effect", type="tool")
def effect(x: int) -> int:
    return x + 1


@requires_langgraph
def test_observe_inside_langgraph_node_nests_under_the_node(lc_spans):
    graph = _graph(lambda state: {"x": effect(state["x"])})
    assert graph.invoke({"x": 1}, THREAD) == {"x": 2}

    spans = _by_name(lc_spans)
    node, root, child = spans["work"], spans["LangGraph"], spans["effect"]
    assert _parent_of(child, spans) is node
    assert _parent_of(node, spans) is root
    assert {s.context.trace_id for s in spans.values()} == {root.context.trace_id}


def test_observe_inside_runnable_lambda_nests_under_the_run(lc_spans):
    from langchain_core.runnables import RunnableLambda

    chain = RunnableLambda(effect).with_config(run_name="step")
    assert chain.invoke(1) == 2

    spans = _by_name(lc_spans)
    assert _parent_of(spans["effect"], spans) is spans["step"]


@requires_langgraph
def test_update_current_span_reaches_the_node_and_update_current_trace_the_root(lc_spans):
    def node(state):
        traceroot.update_current_span(metadata={"charge_key": "thread-1:charge"})
        traceroot.update_current_trace(user_id="user-7", metadata={"plan": "pro"})
        return state

    _graph(node).invoke({"x": 1}, THREAD)

    spans = _by_name(lc_spans)
    work, root = spans["work"], spans["LangGraph"]
    assert "thread-1:charge" in work.attributes[SpanAttributes.SPAN_METADATA]
    # Trace-level attributes go to the trace's root span, where ingest reads trace metadata.
    assert root.attributes[SpanAttributes.TRACE_USER_ID] == "user-7"
    assert "pro" in root.attributes[SpanAttributes.TRACE_METADATA]
    assert SpanAttributes.TRACE_USER_ID not in work.attributes


@requires_langgraph
def test_current_ids_inside_node_identify_the_node_span(lc_spans):
    seen = {}

    def node(state):
        seen["trace"] = traceroot.get_current_trace_id()
        seen["span"] = traceroot.get_current_span_id()
        return state

    _graph(node).invoke({"x": 1}, THREAD)

    work = _by_name(lc_spans)["work"]
    assert seen["trace"] == format(work.context.trace_id, "032x")
    assert seen["span"] == format(work.context.span_id, "016x")


@requires_langgraph
def test_with_an_observe_entrypoint_node_work_still_lands_on_the_node(lc_spans):
    def node(state):
        traceroot.update_current_span(metadata={"where": "node"})
        return {"x": effect(state["x"])}

    graph = _graph(node)

    @observe(name="entrypoint", type="agent")
    def entrypoint():
        return graph.invoke({"x": 1}, THREAD)

    entrypoint()

    spans = _by_name(lc_spans)
    assert _parent_of(spans["LangGraph"], spans) is spans["entrypoint"]
    assert _parent_of(spans["effect"], spans) is spans["work"]
    assert '"where": "node"' in spans["work"].attributes[SpanAttributes.SPAN_METADATA]
    assert SpanAttributes.SPAN_METADATA not in spans["entrypoint"].attributes


@requires_langgraph
def test_with_an_observe_entrypoint_trace_updates_still_land_on_the_entrypoint(lc_spans):
    """The entrypoint is the trace root and is recording, as it was before this fix."""

    def node(state):
        traceroot.update_current_trace(metadata={"plan": "pro"})
        return state

    graph = _graph(node)

    @observe(name="entrypoint", type="agent")
    def entrypoint():
        return graph.invoke({"x": 1}, THREAD)

    entrypoint()

    spans = _by_name(lc_spans)
    assert "pro" in spans["entrypoint"].attributes[SpanAttributes.TRACE_METADATA]
    assert SpanAttributes.TRACE_METADATA not in spans["work"].attributes


@requires_langgraph
def test_nesting_does_not_depend_on_span_start_times(lc_spans, monkeypatch):
    """Clocks can tie (coarse Windows timers) or skew; the choice must not read them.

    Spans the SDK times itself (the @observe entrypoint) start an hour in the future here, so they
    look later than the node, whose start time OpenInference takes from the LangChain run.
    """
    import opentelemetry.sdk.trace as sdk_trace

    monkeypatch.setattr(sdk_trace, "time_ns", lambda: time.time_ns() + 3600 * 10**9)

    def node(state):
        traceroot.update_current_span(metadata={"where": "node"})
        return {"x": effect(state["x"])}

    graph = _graph(node)

    @observe(name="entrypoint", type="agent")
    def entrypoint():
        return graph.invoke({"x": 1}, THREAD)

    entrypoint()

    spans = _by_name(lc_spans)
    assert spans["entrypoint"].start_time > spans["work"].start_time
    assert _parent_of(spans["effect"], spans) is spans["work"]
    assert '"where": "node"' in spans["work"].attributes[SpanAttributes.SPAN_METADATA]


@requires_langgraph
def test_suppressed_work_inside_a_node_never_writes_to_the_node(lc_spans):
    """A local=True evaluation inside a node marks its context suppressed; nothing it records may
    reach the exported node span, and it sees no trace id."""
    from traceroot.transport.span_processor import mark_local_eval_run

    seen = {}

    def node(state):
        with mark_local_eval_run():
            traceroot.update_current_span(metadata={"case": "SECRET"})
            traceroot.update_current_trace(user_id="SECRET")
            seen["trace"] = traceroot.get_current_trace_id()
        return state

    _graph(node).invoke({"x": 1}, THREAD)

    for span in lc_spans.get_finished_spans():
        assert "SECRET" not in str(dict(span.attributes)), span.name
    assert seen["trace"] is None


@requires_langgraph
def test_a_span_made_current_inside_a_node_wins_even_if_it_started_earlier(lc_spans):
    tracer = trace.get_tracer("test")
    held = {}

    def a(state):
        held["conversation"] = tracer.start_span("conversation")
        return state

    def b(state):
        with trace.use_span(held["conversation"], end_on_exit=True):
            traceroot.update_current_span(metadata={"where": "conversation"})
            return {"x": effect(state["x"])}

    from langgraph.graph import END, START, StateGraph

    g = StateGraph(dict)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.compile().invoke({"x": 1}, THREAD)

    spans = _by_name(lc_spans)
    assert _parent_of(spans["effect"], spans) is spans["conversation"]
    assert (
        '"where": "conversation"' in spans["conversation"].attributes[SpanAttributes.SPAN_METADATA]
    )
    assert SpanAttributes.SPAN_METADATA not in spans["b"].attributes


@requires_langgraph
def test_observe_inside_observe_inside_node_keeps_the_inner_observe_parent(lc_spans):
    @observe(name="helper")
    def helper(x: int) -> int:
        traceroot.update_current_span(metadata={"where": "helper"})
        return effect(x)

    _graph(lambda state: {"x": helper(state["x"])}).invoke({"x": 1}, THREAD)

    spans = _by_name(lc_spans)
    assert _parent_of(spans["helper"], spans) is spans["work"]
    assert _parent_of(spans["effect"], spans) is spans["helper"]
    assert '"where": "helper"' in spans["helper"].attributes[SpanAttributes.SPAN_METADATA]
    assert SpanAttributes.SPAN_METADATA not in spans["work"].attributes


@requires_langgraph
def test_async_node_and_generator_observe_nest_under_the_node(lc_spans):
    @observe(name="aeffect", type="tool")
    async def aeffect(x: int) -> int:
        await asyncio.sleep(0)
        return x + 1

    @observe(name="gen")
    def gen(x: int):
        yield x

    async def node(state):
        return {"x": await aeffect(state["x"]) + sum(gen(0))}

    assert asyncio.run(_graph(node).ainvoke({"x": 1}, THREAD)) == {"x": 2}

    spans = _by_name(lc_spans)
    assert _parent_of(spans["aeffect"], spans) is spans["work"]
    assert _parent_of(spans["gen"], spans) is spans["work"]


@requires_langgraph
def test_separate_trace_mode_keeps_an_entrypoint_as_the_current_trace(memory_exporter):
    """With separate_trace_from_runtime_context, the graph is its own trace and does not run under
    the entrypoint, so the entrypoint stays the current span (as before this fix)."""
    from openinference.instrumentation.langchain import LangChainInstrumentor

    instrumentor = LangChainInstrumentor()
    instrumentor.instrument(
        tracer_provider=trace.get_tracer_provider(), separate_trace_from_runtime_context=True
    )
    try:
        seen = {}

        def node(state):
            seen["trace"] = traceroot.get_current_trace_id()
            return state

        graph = _graph(node)

        @observe(name="entrypoint", type="agent")
        def entrypoint():
            return graph.invoke({"x": 1}, THREAD)

        entrypoint()
    finally:
        instrumentor.uninstrument()

    spans = _by_name(memory_exporter)
    assert spans["LangGraph"].parent is None
    assert seen["trace"] == format(spans["entrypoint"].context.trace_id, "032x")


def test_outside_any_run_observe_nesting_is_unchanged(lc_spans):
    @observe(name="outer")
    def outer():
        return effect(1)

    outer()

    spans = _by_name(lc_spans)
    assert _parent_of(spans["effect"], spans) is spans["outer"]
    assert spans["outer"].parent is None


def test_uninstrumented_langchain_is_ignored_quietly(memory_exporter, capsys):
    """With langchain-core imported but not instrumented, nothing changes and nothing prints
    (OpenInference's lookup prints a warning when it has no tracer)."""
    from langchain_core.runnables import RunnableLambda

    from traceroot.context import get_active_span

    seen = {}

    def step(x):
        seen["active"] = get_active_span()
        return effect(x)

    RunnableLambda(step).invoke(1)

    assert not seen["active"].get_span_context().is_valid
    assert [s.name for s in memory_exporter.get_finished_spans()] == ["effect"]
    assert capsys.readouterr().out == ""
