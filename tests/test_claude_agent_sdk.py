"""Tests for the in-house Claude Agent SDK instrumentation.

Focus areas:
  1. Token/cost: authoritative token usage comes from the ResultMessage and lands
     on an LLM span (not the parent agent/query span); cost is not computed here.
  2. Timeline: subagent task spans are created from task system messages
     (including TaskProgress, not only TaskStarted) and parents cover children.

Messages are lightweight stand-ins whose class names match what the real SDK
emits (the instrumentor dispatches on ``type(message).__name__``).
"""

from __future__ import annotations

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from traceroot.instrumentation.claude_agent_sdk import wrap_client, wrap_query

LLM_COMPLETION = "llm.token_count.completion"
LLM_PROMPT = "llm.token_count.prompt"
SPAN_KIND = "openinference.span.kind"


# --- synthetic message / block stand-ins -----------------------------------


class TextBlock:
    def __init__(self, text: str):
        self.text = text


class ToolUseBlock:
    def __init__(self, id: str, name: str, input=None):
        self.id = id
        self.name = name
        self.input = input or {}


class ToolResultBlock:
    def __init__(self, tool_use_id: str, content=None, is_error: bool = False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class AssistantMessage:
    def __init__(self, content, model=None, message_id=None, usage=None, parent_tool_use_id=None):
        self.content = content
        self.model = model
        self.message_id = message_id
        self.usage = usage
        self.parent_tool_use_id = parent_tool_use_id


class UserMessage:
    def __init__(self, content, parent_tool_use_id=None):
        self.content = content
        self.parent_tool_use_id = parent_tool_use_id


class ResultMessage:
    def __init__(
        self, result=None, usage=None, total_cost_usd=None, num_turns=None, session_id=None
    ):
        self.result = result
        self.usage = usage
        self.total_cost_usd = total_cost_usd
        self.num_turns = num_turns
        self.session_id = session_id


class TaskStartedMessage:
    def __init__(self, task_id, description=None, tool_use_id=None, task_type=None):
        self.task_id = task_id
        self.description = description
        self.tool_use_id = tool_use_id
        self.task_type = task_type


class TaskProgressMessage:
    def __init__(self, task_id, description=None, usage=None, tool_use_id=None):
        self.task_id = task_id
        self.description = description
        self.usage = usage
        self.tool_use_id = tool_use_id


class TaskNotificationMessage:
    def __init__(self, task_id, tool_use_id=None, usage=None):
        self.task_id = task_id
        self.tool_use_id = tool_use_id
        self.usage = usage


# --- helpers ----------------------------------------------------------------


def _provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


async def _run(provider, messages):
    async def fake_query(*, prompt, options=None, transport=None):
        for m in messages:
            yield m

    wrapped = wrap_query(fake_query, provider)
    async for _ in wrapped(prompt="hello", options=None):
        pass


def _llm_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.name == "anthropic.messages.create"]


# --- Step 1: token / cost ---------------------------------------------------


@pytest.mark.asyncio
async def test_output_tokens_reconciled_to_result_message():
    """Total completion tokens on LLM spans must equal the ResultMessage total,
    not the tiny per-streamed-message values."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [TextBlock("thinking")],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 10, "output_tokens": 3},
        ),
        AssistantMessage(
            [TextBlock("answer")],
            model="claude-sonnet-4",
            message_id="m2",
            usage={"input_tokens": 10, "output_tokens": 3},
        ),
        ResultMessage(
            result="done",
            usage={"input_tokens": 10, "output_tokens": 1060},
            total_cost_usd=1.23,
            num_turns=1,
        ),
    ]
    await _run(provider, messages)

    llm = _llm_spans(exporter)
    assert llm, "expected at least one anthropic.messages.create LLM span"
    total_completion = sum((s.attributes.get(LLM_COMPLETION) or 0) for s in llm)
    assert total_completion == 1060, f"expected 1060 (ResultMessage total), got {total_completion}"


@pytest.mark.asyncio
async def test_tokens_on_llm_spans_not_on_query_or_task():
    """Token counts belong on LLM spans, never on the query/agent parent."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [TextBlock("a")],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 5, "output_tokens": 2},
        ),
        ResultMessage(
            result="ok", usage={"input_tokens": 5, "output_tokens": 50}, total_cost_usd=0.1
        ),
    ]
    await _run(provider, messages)

    for s in exporter.get_finished_spans():
        if s.name != "anthropic.messages.create":
            assert s.attributes.get(LLM_COMPLETION) is None, (
                f"{s.name} should not carry completion tokens"
            )
            assert s.attributes.get(LLM_PROMPT) is None, f"{s.name} should not carry prompt tokens"


@pytest.mark.asyncio
async def test_1h_cache_write_split_surfaces_as_its_own_attribute():
    """Anthropic prices 1h-retention cache writes at 2.0x input vs 1.25x for 5m
    writes and reports the split under usage.cache_creation. Usage lands from
    the ResultMessage (streamed per-message counts are ignored), so the split
    must ride from there as an additive attribute next to the collapsed total."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [TextBlock("a")],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 5, "output_tokens": 2},
        ),
        ResultMessage(
            result="ok",
            usage={
                "input_tokens": 10,
                "output_tokens": 4,
                "cache_creation_input_tokens": 30,
                "cache_creation": {
                    "ephemeral_5m_input_tokens": 18,
                    "ephemeral_1h_input_tokens": 12,
                },
            },
        ),
    ]
    await _run(provider, messages)

    llm = _llm_spans(exporter)
    assert len(llm) == 1
    attrs = llm[0].attributes
    assert attrs.get("llm.token_count.prompt_details.cache_creation") == 30
    assert attrs.get("llm.token_count.prompt_details.cache_write_1h") == 12


@pytest.mark.asyncio
async def test_no_1h_attribute_when_usage_has_no_breakdown():
    """A result usage without the per-TTL breakdown must not grow a 1h count."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [TextBlock("a")],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 5, "output_tokens": 2},
        ),
        ResultMessage(
            result="ok",
            usage={"input_tokens": 10, "output_tokens": 4, "cache_creation_input_tokens": 7},
        ),
    ]
    await _run(provider, messages)

    llm = _llm_spans(exporter)
    assert len(llm) == 1
    attrs = llm[0].attributes
    assert attrs.get("llm.token_count.prompt_details.cache_creation") == 7
    assert attrs.get("llm.token_count.prompt_details.cache_write_1h") is None


# --- Step 2: timeline / subagent spans --------------------------------------


@pytest.mark.asyncio
async def test_tool_spans_nest_under_their_llm_span():
    """A tool call must be a CHILD of the LLM span that requested it, not a flat
    sibling under the query root: query -> LLM turn -> tool."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [ToolUseBlock(id="c1", name="mcp__calc__multiply", input={"a": 17, "b": 23})],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 5, "output_tokens": 2},
        ),
        UserMessage([ToolResultBlock(tool_use_id="c1", content="391")]),
        ResultMessage(
            result="391", usage={"input_tokens": 5, "output_tokens": 20}, total_cost_usd=0.01
        ),
    ]
    await _run(provider, messages)

    spans = exporter.get_finished_spans()
    tool = next(s for s in spans if s.name == "mcp__calc__multiply")
    llm = next(s for s in spans if s.name == "anthropic.messages.create")
    query = next(s for s in spans if s.name == "ClaudeAgent.query")

    assert tool.parent is not None
    assert tool.parent.span_id == llm.context.span_id, (
        "tool span must nest under its LLM span, not be a flat sibling under the query"
    )
    assert llm.parent is not None and llm.parent.span_id == query.context.span_id


@pytest.mark.asyncio
async def test_turns_split_on_tool_result_when_message_id_absent():
    """message_id is an OPTIONAL SDK field. When it is missing, turns separated by a
    tool_result must still become DISTINCT LLM spans (the tool_result is the boundary),
    not collapse into one merged span with every tool nested under it."""
    provider, exporter = _provider()
    messages = [
        # turn 1: no message_id, requests a tool
        AssistantMessage(
            [ToolUseBlock(id="c1", name="first_tool", input={"x": 1})],
            model="claude-sonnet-4",
        ),
        UserMessage([ToolResultBlock(tool_use_id="c1", content="r1")]),
        # turn 2: still no message_id, requests another tool
        AssistantMessage(
            [ToolUseBlock(id="c2", name="second_tool", input={"x": 2})],
            model="claude-sonnet-4",
        ),
        UserMessage([ToolResultBlock(tool_use_id="c2", content="r2")]),
        ResultMessage(
            result="done", usage={"input_tokens": 5, "output_tokens": 20}, total_cost_usd=0.01
        ),
    ]
    await _run(provider, messages)

    spans = exporter.get_finished_spans()
    llm = _llm_spans(exporter)
    assert len(llm) == 2, f"expected 2 LLM turns split by the tool_result, got {len(llm)}"

    # Each tool nests under its OWN turn, not both under one merged span.
    first = next(s for s in spans if s.name == "first_tool")
    second = next(s for s in spans if s.name == "second_tool")
    assert first.parent is not None and second.parent is not None
    assert first.parent.span_id != second.parent.span_id, (
        "tools from different turns must not share one merged LLM span"
    )
    llm_ids = {s.context.span_id for s in llm}
    assert first.parent.span_id in llm_ids and second.parent.span_id in llm_ids


@pytest.mark.asyncio
async def test_task_progress_only_creates_subagent_span():
    """A subagent that streams only TaskProgress (no TaskStarted) must still get a
    task span (named by role), parented under the Agent tool span."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [ToolUseBlock(id="agent1", name="Agent", input={"subagent_type": "researcher"})],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 5, "output_tokens": 2},
        ),
        TaskProgressMessage(
            task_id="t1", description="Research OTel", tool_use_id="agent1", usage={}
        ),
        TaskProgressMessage(
            task_id="t1", description="Research OTel", tool_use_id="agent1", usage={}
        ),
        UserMessage([ToolResultBlock(tool_use_id="agent1", content="result")]),
        ResultMessage(
            result="done", usage={"input_tokens": 5, "output_tokens": 50}, total_cost_usd=0.2
        ),
    ]
    await _run(provider, messages)

    spans = exporter.get_finished_spans()
    names = [s.name for s in spans]
    task_spans = [s for s in spans if s.name == "researcher"]
    assert task_spans, f"expected a 'researcher' subagent span; got {names}"

    agent_tool = [s for s in spans if s.name == "Agent"]
    assert agent_tool, f"expected an 'Agent' tool span; got {names}"
    assert task_spans[0].parent is not None
    assert task_spans[0].parent.span_id == agent_tool[0].context.span_id, (
        "subagent span must be parented under the Agent tool span, not the root query"
    )


@pytest.mark.asyncio
async def test_result_usage_attaches_to_orchestrator_not_subagent():
    """ResultMessage usage must land on the ORCHESTRATOR LLM span (parent_tool_use_id
    is None), never a subagent turn span — even when a subagent turn is the most
    recent LLM call."""
    provider, exporter = _provider()
    messages = [
        # orchestrator turn (carries a unique marker in its output)
        AssistantMessage(
            "ORCHESTRATOR_MARKER",
            model="m",
            message_id="o1",
            usage={"input_tokens": 5, "output_tokens": 2},
            parent_tool_use_id=None,
        ),
        # a subagent turn — this is the MOST RECENT LLM span before the result
        AssistantMessage(
            "SUBAGENT_MARKER",
            model="m",
            message_id="s1",
            usage={"input_tokens": 5, "output_tokens": 2},
            parent_tool_use_id="sub1",
        ),
        ResultMessage(
            result="done", usage={"input_tokens": 5, "output_tokens": 500}, total_cost_usd=0.1
        ),
    ]
    await _run(provider, messages)

    llm = [s for s in exporter.get_finished_spans() if s.name == "anthropic.messages.create"]
    tagged = [s for s in llm if (s.attributes.get(LLM_COMPLETION) or 0) > 0]
    assert len(tagged) == 1, f"expected exactly 1 tagged LLM span, got {len(tagged)}"
    assert tagged[0].attributes.get(LLM_COMPLETION) == 500
    out = tagged[0].attributes.get("output.value") or ""
    assert "ORCHESTRATOR_MARKER" in out and "SUBAGENT_MARKER" not in out, (
        "result usage must land on the orchestrator turn's LLM span, not the subagent turn's"
    )


@pytest.mark.asyncio
async def test_sequential_subagents_are_siblings_not_nested():
    """Two sequential Agent subagents must be SIBLINGS under the query, not nested
    inside each other. The orchestrator turn that spawns the 2nd subagent must parent
    under the query, not under the still-active 1st subagent task."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [ToolUseBlock(id="r1", name="Agent", input={"subagent_type": "researcher"})],
            model="m",
            message_id="o1",
            parent_tool_use_id=None,
        ),
        TaskStartedMessage(task_id="t1", description="Research", tool_use_id="r1"),
        AssistantMessage("researching", model="m", message_id="rt", parent_tool_use_id="r1"),
        UserMessage([ToolResultBlock(tool_use_id="r1", content="done")]),
        AssistantMessage(
            [ToolUseBlock(id="a1", name="Agent", input={"subagent_type": "analyst"})],
            model="m",
            message_id="o2",
            parent_tool_use_id=None,
        ),
        TaskStartedMessage(task_id="t2", description="Analyze", tool_use_id="a1"),
        AssistantMessage("analyzing", model="m", message_id="at", parent_tool_use_id="a1"),
        UserMessage([ToolResultBlock(tool_use_id="a1", content="done")]),
        ResultMessage(
            result="ok", usage={"input_tokens": 5, "output_tokens": 10}, total_cost_usd=0.1
        ),
    ]
    await _run(provider, messages)

    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}

    def ancestors(span):
        names, cur = [], span
        while cur is not None and cur.parent is not None:
            p = by_id.get(cur.parent.span_id)
            if p is None:
                break
            names.append(p.name)
            cur = p
        return names

    researcher = next(s for s in spans if s.name == "researcher")
    analyst = next(s for s in spans if s.name == "analyst")
    assert "researcher" not in ancestors(analyst), (
        f"analyst nested under researcher; ancestors={ancestors(analyst)}"
    )
    assert "analyst" not in ancestors(researcher)


@pytest.mark.asyncio
async def test_subagent_span_named_by_role_not_description():
    """Subagent task spans use the low-cardinality agent role (researcher/...) as the
    span NAME, not the dynamic LLM-generated task description (which is kept as an attr)."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [ToolUseBlock(id="r1", name="Agent", input={"subagent_type": "researcher"})],
            model="m",
            message_id="o1",
            parent_tool_use_id=None,
        ),
        TaskStartedMessage(
            task_id="t1", description="Get key OpenTelemetry AI facts", tool_use_id="r1"
        ),
        AssistantMessage("work", model="m", message_id="rt", parent_tool_use_id="r1"),
        ResultMessage(
            result="ok", usage={"input_tokens": 5, "output_tokens": 10}, total_cost_usd=0.1
        ),
    ]
    await _run(provider, messages)

    spans = exporter.get_finished_spans()
    sub = next(s for s in spans if s.attributes.get("claude_agent_sdk.tool_use_id") == "r1")
    assert sub.name == "researcher", f"expected role name, got {sub.name!r}"
    # description preserved as an attribute, not lost
    assert "Get key OpenTelemetry AI facts" in (sub.attributes.get("input.value") or "")


@pytest.mark.asyncio
async def test_all_spans_share_single_root():
    """No orphans: every span chains back to exactly one root (the query span)."""
    provider, exporter = _provider()
    messages = [
        AssistantMessage(
            [ToolUseBlock(id="agent1", name="Agent")],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 5, "output_tokens": 2},
        ),
        TaskProgressMessage(task_id="t1", description="Analyze", tool_use_id="agent1", usage={}),
        UserMessage([ToolResultBlock(tool_use_id="agent1", content="r")]),
        ResultMessage(
            result="ok", usage={"input_tokens": 5, "output_tokens": 40}, total_cost_usd=0.1
        ),
    ]
    await _run(provider, messages)

    spans = exporter.get_finished_spans()
    ids = {s.context.span_id for s in spans}
    roots = [s for s in spans if s.parent is None or s.parent.span_id not in ids]
    assert len(roots) == 1, f"expected exactly 1 root, got {len(roots)}: {[s.name for s in roots]}"
    assert roots[0].name == "ClaudeAgent.query"


# --- ClaudeSDKClient (persistent streaming client) path ---------------------


def _make_fake_client_cls(receive=None, query=None):
    """A stand-in for ClaudeSDKClient: query() stashes the prompt, and
    receive_response() replays the instance's preset message list.

    ``receive`` / ``query`` override those methods to simulate errors,
    cancellation, or non-string prompts.
    """

    class FakeClient:
        def __init__(self, options=None, msgs=None):
            self.options = options
            self._msgs = msgs or []
            self.disconnected = False

        async def disconnect(self):
            self.disconnected = True

    if query is None:

        async def _default_query(self, prompt, session_id="default"):
            self._last_prompt = prompt

        FakeClient.query = _default_query
    else:
        FakeClient.query = query

    if receive is None:

        async def _default_receive(self):
            for m in self._msgs:
                yield m

        FakeClient.receive_response = _default_receive
    else:
        FakeClient.receive_response = receive

    return FakeClient


def _wrap_fake_client(provider, receive=None, query=None):
    cls = _make_fake_client_cls(receive=receive, query=query)
    wq, wr, wd = wrap_client(cls, provider)
    cls.query, cls.receive_response, cls.disconnect = wq, wr, wd
    return cls


@pytest.mark.asyncio
async def test_client_turn_creates_root_and_llm_spans():
    """A query()/receive_response() turn produces the same span shape as query():
    one ClaudeAgent.query root with the LLM span nested under it."""
    provider, exporter = _provider()
    cls = _wrap_fake_client(provider)
    messages = [
        AssistantMessage(
            [TextBlock("answer")],
            model="claude-sonnet-4",
            message_id="m1",
            usage={"input_tokens": 10, "output_tokens": 5},
        ),
        ResultMessage(
            result="done", usage={"input_tokens": 10, "output_tokens": 200}, total_cost_usd=0.5
        ),
    ]
    client = cls(options=None, msgs=messages)
    await client.query("hello")
    async for _ in client.receive_response():
        pass

    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == 1 and roots[0].name == "ClaudeAgent.query"
    llm = _llm_spans(exporter)
    assert llm, "expected an anthropic.messages.create LLM span on the client path"
    total = sum((s.attributes.get(LLM_COMPLETION) or 0) for s in llm)
    assert total == 200, f"expected 200 (ResultMessage total), got {total}"
    # LLM span nests under the query root.
    assert llm[0].parent is not None and llm[0].parent.span_id == roots[0].context.span_id


@pytest.mark.asyncio
async def test_client_multi_turn_creates_separate_roots():
    """Each turn on a persistent client is its own root span."""
    provider, exporter = _provider()
    cls = _wrap_fake_client(provider)
    messages = [
        AssistantMessage([TextBlock("a")], model="m", message_id="m1", usage={"input_tokens": 1}),
        ResultMessage(result="r", usage={"input_tokens": 1, "output_tokens": 2}),
    ]
    client = cls(options=None, msgs=messages)
    for _ in range(2):
        await client.query("hi")
        async for _ in client.receive_response():
            pass

    roots = [s for s in exporter.get_finished_spans() if s.name == "ClaudeAgent.query"]
    assert len(roots) == 2, f"expected 2 turn roots, got {len(roots)}"


@pytest.mark.asyncio
async def test_client_receive_without_query_is_passthrough():
    """receive_response with no active turn yields messages untraced, no crash."""
    provider, exporter = _provider()
    cls = _wrap_fake_client(provider)
    client = cls(options=None, msgs=[ResultMessage(result="r")])
    got = [m async for m in client.receive_response()]
    assert len(got) == 1
    assert exporter.get_finished_spans() == ()


@pytest.mark.asyncio
async def test_client_disconnect_ends_dangling_turn():
    """query() with no consumed response still has its span closed on disconnect()."""
    provider, exporter = _provider()
    cls = _wrap_fake_client(provider)
    client = cls(options=None, msgs=[])
    await client.query("hello")
    assert exporter.get_finished_spans() == (), "span should still be open before disconnect"
    await client.disconnect()
    roots = [s for s in exporter.get_finished_spans() if s.name == "ClaudeAgent.query"]
    assert len(roots) == 1, "disconnect must close the dangling turn span"
    assert client.disconnected is True, "original disconnect must still run"


@pytest.mark.asyncio
async def test_client_cancelled_error_in_stream_is_suppressed():
    """A CancelledError leaked by the subprocess transport (e.g. anyio cleanup
    when subagents finish) ends the turn cleanly and is NOT propagated."""
    provider, exporter = _provider()

    async def receive_then_cancel(self):
        for m in self._msgs:
            yield m
        raise asyncio.CancelledError()

    cls = _wrap_fake_client(provider, receive=receive_then_cancel)
    msgs = [
        AssistantMessage([TextBlock("a")], model="m", message_id="m1", usage={"input_tokens": 1}),
        ResultMessage(result="r", usage={"input_tokens": 1, "output_tokens": 2}),
    ]
    client = cls(options=None, msgs=msgs)
    await client.query("hello")
    got = [m async for m in client.receive_response()]  # must not raise
    assert len(got) == 2
    roots = [s for s in exporter.get_finished_spans() if s.name == "ClaudeAgent.query"]
    assert len(roots) == 1, "turn span must still be closed after a suppressed cancel"
    assert roots[0].status.status_code.name != "ERROR"


@pytest.mark.asyncio
async def test_client_async_iterable_prompt_is_handled():
    """A non-string (AsyncIterable) prompt is accepted; the turn still traces and
    no input.value is recorded for it."""
    provider, exporter = _provider()
    cls = _wrap_fake_client(provider)
    msgs = [
        AssistantMessage([TextBlock("a")], model="m", message_id="m1", usage={"input_tokens": 1}),
        ResultMessage(result="r", usage={"input_tokens": 1, "output_tokens": 2}),
    ]

    async def prompt_stream():
        yield {"type": "user", "message": {"role": "user", "content": "hi"}}

    client = cls(options=None, msgs=msgs)
    await client.query(prompt_stream())  # async iterable, not str
    async for _ in client.receive_response():
        pass

    roots = [s for s in exporter.get_finished_spans() if s.name == "ClaudeAgent.query"]
    assert len(roots) == 1
    assert "input.value" not in (roots[0].attributes or {})
    assert _llm_spans(exporter), "LLM span should still be produced for a streamed prompt"


@pytest.mark.asyncio
async def test_client_error_in_stream_marks_span_error_and_propagates():
    """A genuine exception mid-stream is re-raised and the turn span is ERROR."""
    provider, exporter = _provider()

    async def receive_then_raise(self):
        for m in self._msgs:
            yield m
        raise ValueError("boom")

    cls = _wrap_fake_client(provider, receive=receive_then_raise)
    msgs = [
        AssistantMessage([TextBlock("a")], model="m", message_id="m1", usage={"input_tokens": 1})
    ]
    client = cls(options=None, msgs=msgs)
    await client.query("hello")
    with pytest.raises(ValueError, match="boom"):
        async for _ in client.receive_response():
            pass

    roots = [s for s in exporter.get_finished_spans() if s.name == "ClaudeAgent.query"]
    assert len(roots) == 1
    assert roots[0].status.status_code.name == "ERROR"


@pytest.mark.asyncio
async def test_client_query_send_error_marks_span_error_and_propagates():
    """If query() (send) raises, the turn span is closed as ERROR and the error propagates."""
    provider, exporter = _provider()

    async def query_raises(self, prompt, session_id="default"):
        raise RuntimeError("send failed")

    cls = _wrap_fake_client(provider, query=query_raises)
    client = cls(options=None, msgs=[])
    with pytest.raises(RuntimeError, match="send failed"):
        await client.query("hello")

    roots = [s for s in exporter.get_finished_spans() if s.name == "ClaudeAgent.query"]
    assert len(roots) == 1
    assert roots[0].status.status_code.name == "ERROR"
