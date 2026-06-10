"""In-house Claude Agent SDK instrumentation for TraceRoot.

Wraps claude_agent_sdk.query() to create OTel spans for LLM calls, tools,
and task/subagent spans with accurate token/cost tracking.

Uses a message-driven approach (like Braintrust) rather than hooks:
- AssistantMessage → LLM spans (anthropic.messages.create)
- AssistantMessage tool_use blocks → tool span start
- UserMessage tool_result blocks → tool span end
- SystemMessage TaskStarted/TaskNotification → task spans
- ResultMessage → cost/turns metadata
"""

from __future__ import annotations

import json
import logging
import time
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
    """Get message type from class name."""
    name = type(message).__name__
    if "Assistant" in name:
        return "assistant"
    if "Result" in name:
        return "result"
    if "User" in name:
        return "user"
    if "Task" in name:
        return "system"
    if "System" in name:
        return "system"
    return name.lower()


def _msg_subtype(message: Any) -> str | None:
    return getattr(message, "subtype", None)


# ---------------------------------------------------------------------------
# Active span tracking
# ---------------------------------------------------------------------------


class _ToolSpan:
    __slots__ = ("ctx", "name", "span", "tool_use_id")

    def __init__(self, span: trace.Span, ctx: otel_context.Context, name: str, tool_use_id: str):
        self.span = span
        self.ctx = ctx
        self.name = name
        self.tool_use_id = tool_use_id


class _TaskSpan:
    __slots__ = ("ctx", "ended", "span", "tool_use_id")

    def __init__(self, span: trace.Span, ctx: otel_context.Context, tool_use_id: str | None):
        self.span = span
        self.ctx = ctx
        self.tool_use_id = tool_use_id
        self.ended = False


# ---------------------------------------------------------------------------
# Query state — message-driven span tracking
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

        # LLM span state
        self.current_message_id: str | None = None
        self.pending_messages: list[Any] = []
        self.pending_start_time: float | None = None
        self.accumulated_output_tokens = 0

        # Tool spans: tool_use_id → _ToolSpan
        self.active_tools: dict[str, _ToolSpan] = {}

        # Task spans: tool_use_id → _TaskSpan (from SystemMessage TaskStarted)
        self.tasks: dict[str | None, _TaskSpan] = {}
        self.task_order: list[str | None] = []

    # -- LLM parent resolution --

    def _get_task_parent_ctx(self, parent_tool_use_id: str | None) -> otel_context.Context:
        """Find the task span context for a given parent_tool_use_id."""
        if parent_tool_use_id is not None and parent_tool_use_id in self.tasks:
            return self.tasks[parent_tool_use_id].ctx
        # Walk task_order in reverse to find the most recent active task
        for key in reversed(self.task_order):
            task = self.tasks.get(key)
            if task and not task.ended:
                return task.ctx
        return self.query_ctx

    # -- LLM span management --

    def flush_pending_llm(self) -> None:
        if not self.pending_messages:
            return

        first = self.pending_messages[0]
        last = self.pending_messages[-1]

        parent_tool_use_id = getattr(first, "parent_tool_use_id", None)
        parent_ctx = self._get_task_parent_ctx(parent_tool_use_id)

        start = self.pending_start_time or time.time()
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

        # Start tool spans from tool_use blocks in assistant messages
        llm_ctx = trace.set_span_in_context(span, parent_ctx)
        for msg in self.pending_messages:
            self._start_tool_spans_from_message(msg, llm_ctx)

        span.set_status(StatusCode.OK)
        span.end()

        usage = getattr(last, "usage", None) or {}
        self.accumulated_output_tokens += usage.get("output_tokens", 0)
        self.pending_messages = []
        self.pending_start_time = None

    def _start_tool_spans_from_message(self, message: Any, llm_ctx: otel_context.Context) -> None:
        """Create tool spans from tool_use blocks in an AssistantMessage."""
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            if type(block).__name__ != "ToolUseBlock":
                continue
            tool_use_id = getattr(block, "id", None)
            if not tool_use_id:
                continue
            tool_use_id = str(tool_use_id)
            tool_name = str(getattr(block, "name", "tool"))
            tool_input = getattr(block, "input", None)

            tool_span = self.tracer.start_span(
                tool_name,
                attributes={
                    OI_SPAN_KIND: "TOOL",
                    TOOL_NAME_ATTR: tool_name,
                    GEN_AI_TOOL_NAME: tool_name,
                    GEN_AI_TOOL_CALL_ID: tool_use_id,
                },
                context=llm_ctx,
            )
            if tool_input:
                val = _try_json(tool_input)
                if val:
                    tool_span.set_attribute(OI_INPUT_VALUE, val)

            tool_ctx = trace.set_span_in_context(tool_span, llm_ctx)
            self.active_tools[tool_use_id] = _ToolSpan(tool_span, tool_ctx, tool_name, tool_use_id)

    def _finish_tool_spans_from_message(self, message: Any) -> None:
        """End tool spans from tool_result blocks in a UserMessage."""
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            return
        for block in content:
            if type(block).__name__ != "ToolResultBlock":
                continue
            tool_use_id = getattr(block, "tool_use_id", None)
            if not tool_use_id:
                continue
            tool_use_id = str(tool_use_id)
            tool = self.active_tools.pop(tool_use_id, None)
            if not tool:
                continue

            tool_content = getattr(block, "content", None)
            if tool_content:
                val = _try_json(tool_content)
                if val:
                    tool.span.set_attribute(OI_OUTPUT_VALUE, val)

            is_error = getattr(block, "is_error", False)
            if is_error:
                tool.span.set_status(StatusCode.ERROR, "tool error")
            else:
                tool.span.set_status(StatusCode.OK)
            tool.span.end()

    # -- Task span management (from SystemMessage) --

    def _handle_task_event(self, message: Any) -> None:
        """Create/update/end task spans from TaskStarted/TaskNotification."""
        msg_cls = type(message).__name__
        tool_use_id = getattr(message, "tool_use_id", None)
        tool_use_id_str = str(tool_use_id) if tool_use_id is not None else None

        if msg_cls == "TaskStartedMessage":
            if tool_use_id_str in self.tasks:
                return
            # Find parent: the Agent tool span if it exists
            parent_ctx = self.query_ctx
            if tool_use_id_str and tool_use_id_str in self.active_tools:
                parent_ctx = self.active_tools[tool_use_id_str].ctx

            description = getattr(message, "description", None)
            task_type = getattr(message, "task_type", None)
            span_name = (
                str(description) if description else (str(task_type) if task_type else "Task")
            )

            task_span = self.tracer.start_span(
                span_name,
                attributes={OI_SPAN_KIND: "AGENT", CLAUDE_AGENT_TOOL_USE_ID: tool_use_id_str or ""},
                context=parent_ctx,
            )
            task_ctx = trace.set_span_in_context(task_span, parent_ctx)
            self.tasks[tool_use_id_str] = _TaskSpan(task_span, task_ctx, tool_use_id_str)
            self.task_order.append(tool_use_id_str)

        elif msg_cls == "TaskNotificationMessage":
            task = self.tasks.get(tool_use_id_str)
            if task and not task.ended:
                task.span.set_status(StatusCode.OK)
                task.span.end()
                task.ended = True
                self.task_order = [k for k in self.task_order if k != tool_use_id_str]

    # -- Message processing --

    def process_message(self, message: Any) -> None:
        msg_type = _msg_type(message)

        if msg_type == "assistant":
            message_id = getattr(message, "message_id", None)
            if message_id and message_id != self.current_message_id:
                self.flush_pending_llm()
                self.current_message_id = message_id
            if not self.pending_messages:
                self.pending_start_time = time.time()
            self.pending_messages.append(message)

        elif msg_type == "user":
            self.flush_pending_llm()
            self._finish_tool_spans_from_message(message)

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

        elif msg_type == "system":
            self._handle_task_event(message)

    # -- Cleanup --

    def end_all(self, status: StatusCode = StatusCode.OK, error: str | None = None) -> None:
        self.flush_pending_llm()
        for tool in self.active_tools.values():
            tool.span.set_status(status, error)
            tool.span.end()
        self.active_tools.clear()
        for task in self.tasks.values():
            if not task.ended:
                task.span.set_status(status, error)
                task.span.end()
                task.ended = True
        self.tasks.clear()
        self.task_order.clear()


# ---------------------------------------------------------------------------
# Main wrapper — no hooks, pure message-driven
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
        state = _QueryState(
            tracer, query_span, query_ctx, prompt if isinstance(prompt, str) else None
        )

        error: Exception | None = None
        try:
            token = otel_context.attach(query_ctx)
            try:
                async for message in original(prompt=prompt, options=options, transport=transport):
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
