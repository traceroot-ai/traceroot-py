"""In-house Claude Agent SDK instrumentation for TraceRoot.

Wraps claude_agent_sdk.query() to create OTel spans for LLM calls, tools,
and task/subagent spans with accurate token/cost tracking.

Uses a message-driven approach rather than hooks:
- AssistantMessage → LLM spans (anthropic.messages.create)
- AssistantMessage tool_use blocks → tool span start
- UserMessage tool_result blocks → tool span end
- SystemMessage TaskStarted/TaskNotification → task spans
- ResultMessage → cost/turns metadata
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from weakref import WeakKeyDictionary

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
# Anthropic 1h-retention subset of the cache_creation total (2.0x input pricing
# versus 1.25x for the default 5m writes); additive alongside the total, and the
# 5m portion is the remainder so it needs no attribute of its own.
LLM_TOKEN_CACHE_WRITE_1H = "llm.token_count.prompt_details.cache_write_1h"
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
    cache_creation_detail = usage.get("cache_creation")
    cache_write_1h = (
        cache_creation_detail.get("ephemeral_1h_input_tokens", 0) or 0
        if isinstance(cache_creation_detail, dict)
        else 0
    )

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
    if cache_write_1h > 0:
        result["prompt_cache_write_1h_tokens"] = cache_write_1h
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
    if "prompt_cache_write_1h_tokens" in metrics:
        span.set_attribute(LLM_TOKEN_CACHE_WRITE_1H, metrics["prompt_cache_write_1h_tokens"])


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


def _has_tool_result(message: Any) -> bool:
    """True if a UserMessage carries any tool_result block (a turn boundary)."""
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return False
    return any(type(block).__name__ == "ToolResultBlock" for block in content)


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

        # LLM span state. One open LLM span per agent context, keyed by the
        # spawning tool's id (the root/top-level turn uses the None key). Each
        # turn's span is created eagerly on its first AssistantMessage and held
        # open so (a) the tools it requests nest under it and (b) the final
        # ResultMessage usage can be recorded on the root turn's span.
        self.open_llm_spans: dict[str | None, trace.Span] = {}
        self.open_llm_ctx: dict[str | None, otel_context.Context] = {}
        # Per-context id of the in-progress turn, to detect turn boundaries.
        self.current_message_ids: dict[str | None, str | None] = {}
        # Contexts whose next AssistantMessage must begin a fresh turn because a
        # tool_result just closed the prior one. This makes message_id (an
        # optional SDK field) an optimization, not the sole boundary signal.
        self.pending_turn_boundary: set[str | None] = set()
        # Accumulated assistant output blocks for the in-progress turn per context.
        self.llm_output_parts: dict[str | None, list[Any]] = {}

        # Tool spans: tool_use_id → _ToolSpan
        self.active_tools: dict[str, _ToolSpan] = {}
        # Preserve Agent tool contexts after they close, for parenting task spans
        self.agent_tool_contexts: dict[str, otel_context.Context] = {}
        # tool_use_id → subagent role (e.g. "researcher"), used as the task span name.
        self.subagent_names: dict[str, str] = {}

        # Task spans: tool_use_id → _TaskSpan (from SystemMessage TaskStarted)
        self.tasks: dict[str | None, _TaskSpan] = {}
        self.task_order: list[str | None] = []

    # -- LLM parent resolution --

    def _get_task_parent_ctx(self, parent_tool_use_id: str | None) -> otel_context.Context:
        """Parent context for a message's parent_tool_use_id."""
        # Orchestrator turns (None) parent under the query, not the active subagent
        # task — otherwise sibling subagents nest inside each other.
        if parent_tool_use_id is None:
            return self.query_ctx
        if parent_tool_use_id in self.tasks:
            return self.tasks[parent_tool_use_id].ctx
        # Subagent message before its task span exists: most recent active task, else query.
        for key in reversed(self.task_order):
            task = self.tasks.get(key)
            if task and not task.ended:
                return task.ctx
        return self.query_ctx

    # -- LLM span management --

    def _open_or_continue_llm_span(self, message: Any) -> tuple[trace.Span, otel_context.Context]:
        """Return the LLM span for this message's turn, creating it eagerly on the
        first AssistantMessage of the turn so the tools it requests can nest under
        it. Consecutive messages sharing a message_id extend the same span.
        """
        key = getattr(message, "parent_tool_use_id", None)
        message_id = getattr(message, "message_id", None)

        forced = key in self.pending_turn_boundary
        self.pending_turn_boundary.discard(key)

        is_new_turn = (
            key not in self.open_llm_spans
            or forced
            or (message_id is not None and self.current_message_ids.get(key) != message_id)
        )

        if is_new_turn:
            start_ns = int(time.time() * 1e9)
            # Tile: close this context's previous open span at the new turn's start.
            prev = self.open_llm_spans.pop(key, None)
            if prev is not None:
                prev.end(end_time=start_ns)

            parent_ctx = self._get_task_parent_ctx(key)
            span = self.tracer.start_span(
                LLM_SPAN_NAME,
                attributes={OI_SPAN_KIND: "LLM"},
                context=parent_ctx,
                start_time=start_ns,
            )
            span.set_status(StatusCode.OK)
            # No per-turn usage: streamed counts are unreliable. The ResultMessage
            # total is attached to the orchestrator span in process_message().
            if key is None and self.prompt:
                val = _try_json([{"role": "user", "content": self.prompt}])
                if val:
                    span.set_attribute(OI_INPUT_VALUE, val)

            self.open_llm_spans[key] = span
            self.open_llm_ctx[key] = trace.set_span_in_context(span, parent_ctx)
            self.current_message_ids[key] = message_id
            self.llm_output_parts[key] = []

        span = self.open_llm_spans[key]

        model = getattr(message, "model", None)
        if model:
            _set_model(span, model)

        content = getattr(message, "content", None)
        if content is not None:
            self.llm_output_parts[key].append({"role": "assistant", "content": content})
            val = _try_json(self.llm_output_parts[key])
            if val:
                span.set_attribute(OI_OUTPUT_VALUE, val)

        return span, self.open_llm_ctx[key]

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

            # For Agent tools, check if a task span was already created (TaskStartedMessage
            # arrives before AssistantMessage). If so, reparent won't work in OTel,
            # so we just save the context for future task spans.
            if tool_name == "Agent":
                self.agent_tool_contexts[tool_use_id] = tool_ctx
                if isinstance(tool_input, dict):
                    role = tool_input.get("subagent_type") or tool_input.get("name")
                    if role:
                        self.subagent_names[tool_use_id] = str(role)

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
        """Create/update/end subagent task spans from task system messages.

        Created on the first task event of any type (incl. TaskProgress, so a subagent
        that never emits TaskStarted still gets a span), keyed by tool_use_id and
        parented under the spawning tool span; ended on TaskNotification.
        """
        msg_cls = type(message).__name__
        if msg_cls not in (
            "TaskStartedMessage",
            "TaskProgressMessage",
            "TaskNotificationMessage",
        ):
            return

        tool_use_id = getattr(message, "tool_use_id", None)
        tool_use_id_str = str(tool_use_id) if tool_use_id is not None else None
        task_id = getattr(message, "task_id", None)
        # Key by tool_use_id (links to the spawning tool span); fall back to task_id.
        key = (
            tool_use_id_str
            if tool_use_id_str is not None
            else (str(task_id) if task_id is not None else None)
        )
        if key is None:
            return

        description = getattr(message, "description", None)
        task_type = getattr(message, "task_type", None)
        # Low-cardinality span name: the subagent role, else the task type, else generic.
        # The free-text description is kept as an attribute, not as the span name.
        span_name = self.subagent_names.get(key) or (str(task_type) if task_type else "subagent")

        task = self.tasks.get(key)
        if task is None:
            # Parent under the spawning tool span (Agent/Skill/Workflow) if known.
            parent_ctx = self.query_ctx
            if tool_use_id_str:
                if tool_use_id_str in self.active_tools:
                    parent_ctx = self.active_tools[tool_use_id_str].ctx
                elif tool_use_id_str in self.agent_tool_contexts:
                    parent_ctx = self.agent_tool_contexts[tool_use_id_str]

            task_span = self.tracer.start_span(
                span_name,
                attributes={OI_SPAN_KIND: "AGENT", CLAUDE_AGENT_TOOL_USE_ID: tool_use_id_str or ""},
                context=parent_ctx,
            )
            if description:
                task_span.set_attribute(OI_INPUT_VALUE, str(description))
            task_ctx = trace.set_span_in_context(task_span, parent_ctx)
            task = _TaskSpan(task_span, task_ctx, tool_use_id_str)
            self.tasks[key] = task
            self.task_order.append(key)
        elif not task.ended:
            # Update to the role once the spawning Agent tool reveals it (rare ordering).
            role = self.subagent_names.get(key)
            if role:
                task.span.update_name(role)

        if msg_cls == "TaskNotificationMessage" and not task.ended:
            task.span.set_status(StatusCode.OK)
            task.span.end()
            task.ended = True
            self.task_order = [k for k in self.task_order if k != key]

    # -- Message processing --

    def process_message(self, message: Any) -> None:
        msg_type = _msg_type(message)

        if msg_type == "assistant":
            # Open (or extend) this turn's LLM span first, then nest the tools it
            # requests under it: query -> LLM turn -> tool. The Agent tool span is
            # created here so a later TaskStartedMessage can find it as parent.
            _span, llm_ctx = self._open_or_continue_llm_span(message)
            self._start_tool_spans_from_message(message, llm_ctx)

        elif msg_type == "user":
            self._finish_tool_spans_from_message(message)
            # A tool_result closes the current turn for its context: the next
            # AssistantMessage there starts a fresh LLM turn even if message_id
            # is absent. Key by the user message's parent_tool_use_id (the same
            # context the answering AssistantMessage will carry).
            if _has_tool_result(message):
                self.pending_turn_boundary.add(getattr(message, "parent_tool_use_id", None))

        elif msg_type == "result":
            # Attach the ResultMessage usage to the orchestrator (None) span only;
            # subagent turns stay empty (sparse).
            result_usage = getattr(message, "usage", None)
            orchestrator_span = self.open_llm_spans.get(None)
            if orchestrator_span is not None and result_usage:
                _set_usage_attrs(orchestrator_span, result_usage)
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
        for span in self.open_llm_spans.values():
            if error is not None:
                span.set_status(status, error)
            span.end()
        self.open_llm_spans.clear()
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
# Main wrapper — pure message-driven, no hooks
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
                    # Never let instrumentation break the user's query: a bad
                    # message shape must not propagate out of their async-for.
                    try:
                        state.process_message(message)
                    except Exception:
                        logger.warning(
                            "traceroot: failed to process claude_agent_sdk message",
                            exc_info=True,
                        )
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
# Persistent streaming client (ClaudeSDKClient)
# ---------------------------------------------------------------------------
#
# Unlike query(), which is a single async generator, the client splits one turn
# across separate calls: query() sends the prompt, receive_response() consumes
# the message stream. The _QueryState message engine is reused unchanged; only
# the orchestration differs. State must live on the client *instance* (not a
# closure) because send and receive are distinct calls, so we key it by instance
# in a WeakKeyDictionary. Each query()/receive_response() turn gets its own root
# span, mirroring wrap_query's per-request structure.

# client instance -> active turn state (one in-flight turn per client)
_CLIENT_STATES: WeakKeyDictionary[Any, _QueryState] = WeakKeyDictionary()


def _end_client_turn(client: Any, error: Exception | None = None) -> None:
    state = _CLIENT_STATES.pop(client, None)
    if state is None:
        return
    if error is not None:
        state.end_all(StatusCode.ERROR, str(error))
        state.query_span.set_status(StatusCode.ERROR, str(error))
        state.query_span.record_exception(error)
    else:
        state.end_all(StatusCode.OK)
        state.query_span.set_status(StatusCode.OK)
    state.query_span.end()


def _safe_end_client_turn(client: Any, error: Exception | None = None) -> None:
    # Teardown must never raise into the user's code path.
    try:
        _end_client_turn(client, error)
    except Exception:
        logger.warning("traceroot: failed to end claude_agent_sdk client turn", exc_info=True)


def wrap_client(client_cls: Any, tracer_provider: Any = None) -> tuple[Any, Any, Any]:
    """Build tracing wrappers for ClaudeSDKClient.query/receive_response/disconnect.

    Returns ``(wrapped_query, wrapped_receive_response, wrapped_disconnect)``.
    """
    if tracer_provider is not None:
        tracer = tracer_provider.get_tracer(TRACER_NAME)
    else:
        tracer = trace.get_tracer(TRACER_NAME)

    orig_query = client_cls.query
    orig_receive = client_cls.receive_response
    orig_disconnect = client_cls.disconnect

    async def wrapped_query(self: Any, prompt: Any, session_id: str = "default") -> None:
        # Start a fresh turn. Defensively close any prior turn that never had its
        # response consumed so its span is not leaked.
        try:
            _end_client_turn(self)
            parent_ctx = otel_context.get_current()
            query_span = tracer.start_span(
                QUERY_SPAN_NAME,
                attributes={OI_SPAN_KIND: "AGENT"},
                context=parent_ctx,
            )
            if isinstance(prompt, str):
                query_span.set_attribute(OI_INPUT_VALUE, prompt)
            model = getattr(getattr(self, "options", None), "model", None)
            if model:
                query_span.set_attribute(CLAUDE_AGENT_MODEL, model)
            query_ctx = trace.set_span_in_context(query_span, parent_ctx)
            _CLIENT_STATES[self] = _QueryState(
                tracer, query_span, query_ctx, prompt if isinstance(prompt, str) else None
            )
        except Exception:
            logger.warning("traceroot: failed to start claude_agent_sdk client turn", exc_info=True)

        try:
            return await orig_query(self, prompt, session_id)
        except Exception as exc:
            _safe_end_client_turn(self, exc)
            raise

    async def wrapped_receive_response(self: Any) -> AsyncIterator[Any]:
        state = _CLIENT_STATES.get(self)
        if state is None:
            # No active turn (e.g. receive_response without a preceding query) —
            # pass through untraced rather than guess at a span.
            async for message in orig_receive(self):
                yield message
            return

        error: Exception | None = None
        token = otel_context.attach(state.query_ctx)
        try:
            async for message in orig_receive(self):
                try:
                    state.process_message(message)
                except Exception:
                    logger.warning(
                        "traceroot: failed to process claude_agent_sdk message",
                        exc_info=True,
                    )
                yield message
        except asyncio.CancelledError:
            # The claude_agent_sdk subprocess transport (anyio) can raise
            # CancelledError during normal cleanup — e.g. when subagents
            # finish — rather than as a genuine external cancellation. End the
            # turn cleanly and stop without propagating; a real caller
            # cancellation still re-fires at the caller's next await point.
            pass
        except Exception as exc:
            error = exc
            raise
        finally:
            otel_context.detach(token)
            _safe_end_client_turn(self, error)

    async def wrapped_disconnect(self: Any) -> None:
        # Covers explicit disconnect() and `async with` exit (__aexit__ calls
        # disconnect). Close any turn whose stream was never fully consumed.
        _safe_end_client_turn(self)
        return await orig_disconnect(self)

    return wrapped_query, wrapped_receive_response, wrapped_disconnect


# ---------------------------------------------------------------------------
# Public entry point — called by registry
# ---------------------------------------------------------------------------

_WRAPPED = False


def instrument_claude_agent_sdk(tracer_provider: Any = None) -> None:
    """Monkey-patch claude_agent_sdk.query and the ClaudeSDKClient path."""
    global _WRAPPED
    if _WRAPPED:
        return
    import claude_agent_sdk

    claude_agent_sdk.query = wrap_query(claude_agent_sdk.query, tracer_provider)

    # Persistent streaming client: wrap the send / receive / teardown methods so
    # ClaudeSDKClient sessions are traced, not only the one-shot query() helper.
    client_cls = getattr(claude_agent_sdk, "ClaudeSDKClient", None)
    if client_cls is not None:
        wrapped_query, wrapped_receive_response, wrapped_disconnect = wrap_client(
            client_cls, tracer_provider
        )
        client_cls.query = wrapped_query
        client_cls.receive_response = wrapped_receive_response
        client_cls.disconnect = wrapped_disconnect

    _WRAPPED = True
