"""In-house Claude Agent SDK instrumentation for TraceRoot.

Wraps claude_agent_sdk.query() to create OTel spans for LLM calls, tools,
and subagents with accurate token/cost tracking. Replaces the OpenInference
ClaudeAgentSDKInstrumentor which produces inaccurate token counts and
missing LLM-level spans.

Architecture mirrors traceroot-ts/packages/traceroot/src/claude-agent-sdk.ts
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

TRACER_NAME = "traceroot.claude-agent-sdk"
QUERY_SPAN_NAME = "ClaudeAgent.query"
LLM_SPAN_NAME = "anthropic.messages.create"

# OpenInference attribute keys
OI_SPAN_KIND = "openinference.span.kind"
OI_INPUT_VALUE = "input.value"
OI_OUTPUT_VALUE = "output.value"
OI_LLM_MODEL_NAME = "llm.model_name"
OI_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
OI_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
OI_TRACE_SESSION_ID = "session.id"
TOOL_NAME_ATTR = "tool.name"

LLM_TOKEN_TOTAL = "llm.token_count.total"
LLM_TOKEN_CACHE_READ = "llm.token_count.prompt_details.cache_read"
LLM_TOKEN_CACHE_CREATION = "llm.token_count.prompt_details.cache_creation"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

CLAUDE_AGENT_MODEL = "claude_agent_sdk.model"
CLAUDE_AGENT_NUM_TURNS = "claude_agent_sdk.num_turns"
CLAUDE_AGENT_TOTAL_COST = "claude_agent_sdk.total_cost_usd"
CLAUDE_AGENT_AGENT_ID = "claude_agent_sdk.agent_id"
CLAUDE_AGENT_AGENT_TYPE = "claude_agent_sdk.agent_type"
CLAUDE_AGENT_TOOL_USE_ID = "claude_agent_sdk.tool_use_id"


def extract_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    """Extract normalized token metrics from Anthropic usage dict."""
    if not usage:
        return {}

    input_tokens = usage.get("input_tokens", 0) or 0
    output_tokens = usage.get("output_tokens", 0) or 0
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0

    prompt_tokens = input_tokens + cache_read + cache_creation

    result: dict[str, int] = {}
    if prompt_tokens > 0:
        result["prompt_tokens"] = prompt_tokens
    if output_tokens > 0:
        result["completion_tokens"] = output_tokens
    if prompt_tokens + output_tokens > 0:
        result["total_tokens"] = prompt_tokens + output_tokens
    if cache_read > 0:
        result["prompt_cached_tokens"] = cache_read
    if cache_creation > 0:
        result["prompt_cache_creation_tokens"] = cache_creation
    return result


def _set_usage_attrs(span: trace.Span, usage: dict[str, Any] | None) -> None:
    metrics = extract_usage(usage)
    if not metrics:
        return
    if "prompt_tokens" in metrics:
        span.set_attribute(OI_LLM_TOKEN_COUNT_PROMPT, metrics["prompt_tokens"])
    if "completion_tokens" in metrics:
        span.set_attribute(OI_LLM_TOKEN_COUNT_COMPLETION, metrics["completion_tokens"])
    if "total_tokens" in metrics:
        span.set_attribute(LLM_TOKEN_TOTAL, metrics["total_tokens"])
    if "prompt_cached_tokens" in metrics:
        span.set_attribute(LLM_TOKEN_CACHE_READ, metrics["prompt_cached_tokens"])
    if "prompt_cache_creation_tokens" in metrics:
        span.set_attribute(LLM_TOKEN_CACHE_CREATION, metrics["prompt_cache_creation_tokens"])


def _set_model(span: trace.Span, model: str | None) -> None:
    if not model:
        return
    span.set_attribute(OI_LLM_MODEL_NAME, model)
    span.set_attribute(GEN_AI_RESPONSE_MODEL, model)


def _try_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _msg_type(message: Any) -> str:
    """Get message type from class name. Python Claude Agent SDK uses dataclasses
    without a 'type' field, unlike the TS SDK."""
    name = type(message).__name__
    if "Assistant" in name:
        return "assistant"
    if "Result" in name:
        return "result"
    if "User" in name:
        return "user"
    if "System" in name or "Task" in name:
        return "system"
    return name.lower()


# ---------------------------------------------------------------------------
# Tool & subagent state
# ---------------------------------------------------------------------------

class _ToolState:
    __slots__ = ("ctx", "has_subagent", "name", "span", "tool_use_id")

    def __init__(self, span: trace.Span, ctx: otel_context.Context, name: str, tool_use_id: str):
        self.span = span
        self.ctx = ctx
        self.name = name
        self.tool_use_id = tool_use_id
        self.has_subagent = False


class _SubagentState:
    __slots__ = ("agent_id", "ctx", "ended", "span", "tool_use_id")

    def __init__(self, span: trace.Span, ctx: otel_context.Context, tool_use_id: str):
        self.span = span
        self.ctx = ctx
        self.tool_use_id = tool_use_id
        self.agent_id: str | None = None
        self.ended = False


# ---------------------------------------------------------------------------
# Query state — tracks everything for a single query() call
# ---------------------------------------------------------------------------

class _QueryState:
    def __init__(
        self,
        tracer: trace.Tracer,
        query_span: trace.Span,
        query_ctx: otel_context.Context,
        prompt: str | None,
    ):
        self.tracer = tracer
        self.query_span = query_span
        self.query_ctx = query_ctx
        self.prompt = prompt

        self.current_message_id: str | None = None
        self.pending_messages: list[Any] = []
        self.pending_start_time: float | None = None
        self.accumulated_output_tokens = 0

        self.tools: dict[str, _ToolState] = {}
        self.subagents: dict[str, _SubagentState] = {}
        self.agent_id_to_tool_use_id: dict[str, str] = {}
        self.tool_use_to_parent: dict[str, str | None] = {}
        # Maps Agent tool_use_id → subagent tool_use_id (they differ)
        self.agent_tool_to_subagent: dict[str, str] = {}

    # -- LLM span management --

    def _resolve_subagent_ctx(self, parent_tool_use_id: str | None) -> otel_context.Context | None:
        """Find the subagent context for a given parent_tool_use_id."""
        if not parent_tool_use_id:
            return None
        # Direct match
        if parent_tool_use_id in self.subagents:
            return self.subagents[parent_tool_use_id].ctx
        # Agent tool_use_id → subagent mapping
        sa_key = self.agent_tool_to_subagent.get(parent_tool_use_id)
        if sa_key and sa_key in self.subagents:
            return self.subagents[sa_key].ctx
        return None

    def _get_llm_parent_ctx(self, parent_tool_use_id: str | None) -> otel_context.Context:
        ctx = self._resolve_subagent_ctx(parent_tool_use_id)
        return ctx or self.query_ctx

    def flush_pending_llm(self) -> None:
        if not self.pending_messages:
            return

        first = self.pending_messages[0]
        last = self.pending_messages[-1]

        parent_tool_use_id = getattr(first, "parent_tool_use_id", None)
        parent_ctx = self._get_llm_parent_ctx(parent_tool_use_id)

        import time as _time
        start = self.pending_start_time or _time.time()
        span = self.tracer.start_span(
            LLM_SPAN_NAME,
            attributes={OI_SPAN_KIND: "LLM"},
            context=parent_ctx,
            start_time=int(start * 1e9),
        )

        model = getattr(last, "model", None)
        if not model:
            for msg in reversed(self.pending_messages):
                model = getattr(msg, "model", None)
                if model:
                    break
        _set_model(span, model)
        _set_usage_attrs(span, getattr(last, "usage", None))

        if not parent_tool_use_id and self.prompt:
            val = _try_json([{"role": "user", "content": self.prompt}])
            if val:
                span.set_attribute(OI_INPUT_VALUE, val)

        output_parts = []
        for msg in self.pending_messages:
            content = getattr(msg, "content", None)
            if content is not None:
                output_parts.append({"role": "assistant", "content": content})
        if output_parts:
            val = _try_json(output_parts)
            if val:
                span.set_attribute(OI_OUTPUT_VALUE, val)

        span.set_status(StatusCode.OK)
        span.end()

        usage = getattr(last, "usage", None) or {}
        self.accumulated_output_tokens += usage.get("output_tokens", 0)
        self.pending_messages = []
        self.pending_start_time = None

    # -- Tool use tracking from assistant messages --

    def _track_tool_use_context(self, message: Any) -> None:
        if _msg_type(message) != "assistant":
            return
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        parent_tool_use_id = getattr(message, "parent_tool_use_id", None)
        for block in content:
            block_type = getattr(block, "type", None)
            block_id = getattr(block, "id", None)
            if block_type == "tool_use" and isinstance(block_id, str):
                self.tool_use_to_parent[block_id] = parent_tool_use_id

    # -- Message processing --

    def process_message(self, message: Any) -> None:
        self._track_tool_use_context(message)
        msg_type = _msg_type(message)

        if msg_type == "assistant":
            import time as _time
            message_id = getattr(message, "message_id", None)
            if message_id and message_id != self.current_message_id:
                self.flush_pending_llm()
                self.current_message_id = message_id
            if not self.pending_messages:
                self.pending_start_time = _time.time()
            self.pending_messages.append(message)

        elif msg_type == "result":
            self.flush_pending_llm()
            result = getattr(message, "result", None)
            if isinstance(result, str):
                self.query_span.set_attribute(OI_OUTPUT_VALUE, result)
            total_cost = getattr(message, "total_cost_usd", None)
            if total_cost is not None:
                self.query_span.set_attribute(CLAUDE_AGENT_TOTAL_COST, total_cost)
            num_turns = getattr(message, "num_turns", None)
            if isinstance(num_turns, int):
                self.query_span.set_attribute(CLAUDE_AGENT_NUM_TURNS, num_turns)
            session_id = getattr(message, "session_id", None)
            if isinstance(session_id, str):
                self.query_span.set_attribute(OI_TRACE_SESSION_ID, session_id)

    # -- Cleanup --

    def end_all(self, status: StatusCode = StatusCode.OK, error: str | None = None) -> None:
        self.flush_pending_llm()
        for tool in self.tools.values():
            tool.span.set_status(status, error)
            tool.span.end()
        self.tools.clear()
        for sub in self.subagents.values():
            if not sub.ended:
                sub.span.set_status(status, error)
                sub.span.end()
                sub.ended = True
        self.subagents.clear()


# ---------------------------------------------------------------------------
# Hook creation — injected into query options
# ---------------------------------------------------------------------------

def _create_hooks(state: _QueryState) -> dict[str, list[Any]]:
    from claude_agent_sdk.types import HookMatcher

    async def pre_tool_use(input_data: dict[str, Any], tool_use_id: str | None, ctx: Any) -> dict:
        name = input_data.get("tool_name", "tool")
        tuid = tool_use_id or input_data.get("tool_use_id", "")
        if not tuid:
            return {}

        parent_tool_use_id = state.tool_use_to_parent.get(tuid)
        parent_ctx = state._get_llm_parent_ctx(parent_tool_use_id)

        # If this tool runs inside a subagent, parent under the subagent span
        agent_id = input_data.get("agent_id")
        if agent_id:
            sa_tuid = state.agent_id_to_tool_use_id.get(agent_id)
            if sa_tuid and sa_tuid in state.subagents:
                parent_ctx = state.subagents[sa_tuid].ctx

        span = state.tracer.start_span(
            name,
            attributes={
                OI_SPAN_KIND: "TOOL",
                TOOL_NAME_ATTR: name,
                GEN_AI_TOOL_NAME: name,
                GEN_AI_TOOL_CALL_ID: tuid,
            },
            context=parent_ctx,
        )
        tool_input = input_data.get("tool_input")
        if tool_input:
            val = _try_json(tool_input)
            if val:
                span.set_attribute(OI_INPUT_VALUE, val)

        tool_ctx = trace.set_span_in_context(span, parent_ctx)
        state.tools[tuid] = _ToolState(span, tool_ctx, name, tuid)
        return {}

    async def post_tool_use(input_data: dict[str, Any], tool_use_id: str | None, ctx: Any) -> dict:
        tuid = tool_use_id or input_data.get("tool_use_id", "")
        tool = state.tools.pop(tuid, None)
        if tool:
            response = input_data.get("tool_response")
            if response:
                val = _try_json(response)
                if val:
                    tool.span.set_attribute(OI_OUTPUT_VALUE, val)
            tool.span.set_status(StatusCode.OK)
            tool.span.end()

        # End subagent if this was an Agent tool
        sub = state.subagents.get(tuid)
        if sub and not sub.ended:
            response = input_data.get("tool_response")
            if response and isinstance(response, dict):
                agent_type = response.get("agentType")
                if agent_type:
                    sub.span.set_attribute(CLAUDE_AGENT_AGENT_TYPE, str(agent_type))
                content = response.get("content")
                if content:
                    val = _try_json(content)
                    if val:
                        sub.span.set_attribute(OI_OUTPUT_VALUE, val)
            sub.span.set_status(StatusCode.OK)
            sub.span.end()
            sub.ended = True
        return {}

    async def post_tool_use_failure(input_data: dict[str, Any], tool_use_id: str | None, ctx: Any) -> dict:
        tuid = tool_use_id or input_data.get("tool_use_id", "")
        error_msg = input_data.get("error", "Tool failed")
        tool = state.tools.pop(tuid, None)
        if tool:
            tool.span.set_status(StatusCode.ERROR, str(error_msg))
            tool.span.record_exception(Exception(str(error_msg)))
            tool.span.end()

        sub = state.subagents.get(tuid)
        if sub and not sub.ended:
            sub.span.set_status(StatusCode.ERROR, str(error_msg))
            sub.span.end()
            sub.ended = True
        return {}

    async def subagent_start(input_data: dict[str, Any], tool_use_id: str | None, ctx: Any) -> dict:
        agent_id = input_data.get("agent_id", "")
        agent_type = input_data.get("agent_type", "")
        tuid = tool_use_id or ""
        if not tuid:
            return {}

        # Find the Agent tool span to parent under.
        # SubagentStart's tool_use_id differs from the Agent tool's tool_use_id,
        # so find the most recent unlinked Agent tool and map them.
        tool = state.tools.get(tuid)
        if not tool:
            for t in reversed(list(state.tools.values())):
                if t.name == "Agent" and not t.has_subagent:
                    tool = t
                    state.agent_tool_to_subagent[t.tool_use_id] = tuid
                    break
        parent_ctx = tool.ctx if tool else state.query_ctx
        if tool:
            tool.has_subagent = True

        span = state.tracer.start_span(
            agent_type or "Subagent",
            attributes={
                OI_SPAN_KIND: "AGENT",
                CLAUDE_AGENT_TOOL_USE_ID: tuid,
            },
            context=parent_ctx,
        )
        if agent_id:
            span.set_attribute(CLAUDE_AGENT_AGENT_ID, agent_id)
        if agent_type:
            span.set_attribute(CLAUDE_AGENT_AGENT_TYPE, agent_type)

        sub_ctx = trace.set_span_in_context(span, parent_ctx)
        sub = _SubagentState(span, sub_ctx, tuid)
        sub.agent_id = agent_id
        state.subagents[tuid] = sub
        if agent_id:
            state.agent_id_to_tool_use_id[agent_id] = tuid
        return {}

    async def subagent_stop(input_data: dict[str, Any], tool_use_id: str | None, ctx: Any) -> dict:
        tuid = tool_use_id or ""
        sub = state.subagents.get(tuid)
        if not sub or sub.ended:
            return {}
        last_msg = input_data.get("last_assistant_message")
        if last_msg:
            val = _try_json(last_msg)
            if val:
                sub.span.set_attribute(OI_OUTPUT_VALUE, val)
        sub.span.set_status(StatusCode.OK)
        sub.span.end()
        sub.ended = True
        return {}

    return {
        "PreToolUse": [HookMatcher(hooks=[pre_tool_use])],
        "PostToolUse": [HookMatcher(hooks=[post_tool_use])],
        "PostToolUseFailure": [HookMatcher(hooks=[post_tool_use_failure])],
        "SubagentStart": [HookMatcher(hooks=[subagent_start])],
        "SubagentStop": [HookMatcher(hooks=[subagent_stop])],
    }


# ---------------------------------------------------------------------------
# Hook merging
# ---------------------------------------------------------------------------

def _merge_hooks(
    options: Any | None,
    tracing_hooks: dict[str, list[dict[str, Any]]],
) -> dict[str, list[Any]]:
    existing = getattr(options, "hooks", None) or {}
    merged = dict(existing)
    for event, matchers in tracing_hooks.items():
        merged[event] = list(merged.get(event, [])) + matchers
    return merged


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

def wrap_query(original: Any, tracer_provider: Any = None) -> Any:
    """Wrap claude_agent_sdk.query() with OTel tracing."""
    if tracer_provider is not None:
        tracer = tracer_provider.get_tracer(TRACER_NAME)
    else:
        tracer = trace.get_tracer(TRACER_NAME)

    async def wrapped_query(
        *,
        prompt: Any,
        options: Any = None,
        transport: Any = None,
    ) -> AsyncIterator[Any]:
        parent_ctx = otel_context.get_current()
        query_span = tracer.start_span(
            QUERY_SPAN_NAME,
            attributes={OI_SPAN_KIND: "AGENT"},
            context=parent_ctx,
        )
        if isinstance(prompt, str):
            query_span.set_attribute(OI_INPUT_VALUE, prompt)

        model = getattr(options, "model", None) if options else None
        if model:
            query_span.set_attribute(CLAUDE_AGENT_MODEL, model)

        query_ctx = trace.set_span_in_context(query_span, parent_ctx)
        state = _QueryState(tracer, query_span, query_ctx, prompt if isinstance(prompt, str) else None)

        # Inject tracing hooks into options
        hooks = _create_hooks(state)
        merged_hooks = _merge_hooks(options, hooks)

        import dataclasses  # noqa: I001

        from claude_agent_sdk.types import ClaudeAgentOptions as _Opts
        if options is not None and dataclasses.is_dataclass(options):
            modified_options = dataclasses.replace(options, hooks=merged_hooks)
        else:
            modified_options = _Opts(hooks=merged_hooks)

        error: Exception | None = None
        try:
            token = otel_context.attach(query_ctx)
            try:
                async for message in original(prompt=prompt, options=modified_options, transport=transport):
                    state.process_message(message)
                    yield message
            finally:
                otel_context.detach(token)
        except Exception as exc:
            error = exc
            raise
        finally:
            if error is not None:
                state.end_all(StatusCode.ERROR, str(error))
                query_span.set_status(StatusCode.ERROR, str(error))
                query_span.record_exception(error)
            else:
                state.end_all(StatusCode.OK)
                query_span.set_status(StatusCode.OK)
            query_span.end()

    return wrapped_query


# ---------------------------------------------------------------------------
# Public entry point — called by registry
# ---------------------------------------------------------------------------

_WRAPPED = False


def instrument_claude_agent_sdk(tracer_provider: Any = None) -> None:
    """Monkey-patch claude_agent_sdk.query with tracing wrapper."""
    global _WRAPPED
    if _WRAPPED:
        return
    import claude_agent_sdk

    claude_agent_sdk.query = wrap_query(claude_agent_sdk.query, tracer_provider)
    _WRAPPED = True
