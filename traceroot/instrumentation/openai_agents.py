"""In-house OpenAI Agent SDK instrumentation for TraceRoot.

Registers a TracingProcessor with the OpenAI Agent SDK to convert its
built-in trace/span events into OTel spans with OpenInference attributes.

Span data type mapping:
- Agent/Handoff/Custom/SpeechGroup/Task/Turn → AGENT spans
- Function/Guardrail/MCPListTools → TOOL spans
- Generation/Response/Transcription/Speech → LLM spans
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import StatusCode

logger = logging.getLogger(__name__)

TRACER_NAME = "traceroot.openai-agents"

# OpenInference attribute keys
OI_SPAN_KIND = "openinference.span.kind"
OI_INPUT_VALUE = "input.value"
OI_INPUT_MIME_TYPE = "input.mime_type"
OI_OUTPUT_VALUE = "output.value"
OI_OUTPUT_MIME_TYPE = "output.mime_type"
OI_LLM_MODEL_NAME = "llm.model_name"
OI_LLM_INVOCATION_PARAMETERS = "llm.invocation_parameters"
OI_LLM_SYSTEM = "llm.system"
OI_LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
OI_LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
OI_LLM_TOKEN_COUNT_TOTAL = "llm.token_count.total"
OI_LLM_TOKEN_COUNT_PROMPT_CACHE_READ = "llm.token_count.prompt_details.cache_read"
OI_LLM_TOKEN_COUNT_COMPLETION_REASONING = "llm.token_count.completion_details.reasoning"
OI_TOOL_NAME = "tool.name"

GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"

GRAPH_NODE_ID = "graph.node.id"
GRAPH_NODE_PARENT_ID = "graph.node.parent_id"

MIME_JSON = "application/json"


def _try_json(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _iso_to_ns(iso: str | None) -> int | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso)
        return int(dt.astimezone(timezone.utc).timestamp() * 1_000_000_000)  # noqa: UP017
    except (ValueError, OSError):
        return None


# ---------------------------------------------------------------------------
# Span kind / name resolution
# ---------------------------------------------------------------------------

_AGENT_TYPES = frozenset({"agent", "handoff", "custom", "speech_group", "task", "turn"})
_TOOL_TYPES = frozenset({"function", "guardrail", "mcp_tools"})
_LLM_TYPES = frozenset({"generation", "response", "transcription", "speech"})


def _span_kind(span_data: Any) -> str:
    stype = getattr(span_data, "type", None)
    if stype in _AGENT_TYPES:
        return "AGENT"
    if stype in _TOOL_TYPES:
        return "TOOL"
    if stype in _LLM_TYPES:
        return "LLM"
    return "AGENT"


def _span_name(span: Any) -> str:
    data = span.span_data
    stype = getattr(data, "type", None)

    if stype in ("agent", "function", "guardrail", "custom"):
        name = getattr(data, "name", None)
        if name:
            return str(name)

    if stype == "generation":
        return "Generation"
    if stype == "response":
        return "Response"
    if stype == "handoff":
        to_agent = getattr(data, "to_agent", None)
        return f"handoff to {to_agent}" if to_agent else "Handoff"
    if stype == "mcp_tools":
        server = getattr(data, "server", None)
        return f"List Tools ({server})" if server else "MCP List Tools"
    if stype == "transcription":
        return "Transcription"
    if stype == "speech":
        return "Speech"
    if stype == "speech_group":
        return "Speech Group"
    if stype == "task":
        name = getattr(data, "name", None)
        return str(name) if name else "Task"
    if stype == "turn":
        turn = getattr(data, "turn", None)
        agent_name = getattr(data, "agent_name", None)
        return f"Turn {turn} ({agent_name})"

    return getattr(data, "type", "Unknown")


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


def _extract_usage(usage: Any) -> dict[str, int]:
    if not usage:
        return {}

    if isinstance(usage, dict):
        raw = usage
    elif hasattr(usage, "model_dump"):
        raw = usage.model_dump()
    else:
        raw = {}

    metrics: dict[str, int] = {}

    input_tokens = raw.get("input_tokens", 0) or 0
    output_tokens = raw.get("output_tokens", 0) or 0

    if input_tokens:
        metrics["prompt_tokens"] = input_tokens
    if output_tokens:
        metrics["completion_tokens"] = output_tokens
    if input_tokens + output_tokens:
        metrics["total_tokens"] = input_tokens + output_tokens

    details = raw.get("input_tokens_details") or {}
    if isinstance(details, dict):
        cached = details.get("cached_tokens", 0) or 0
        if cached:
            metrics["prompt_cached_tokens"] = cached

    out_details = raw.get("output_tokens_details") or {}
    if isinstance(out_details, dict):
        reasoning = out_details.get("reasoning_tokens", 0) or 0
        if reasoning:
            metrics["completion_reasoning_tokens"] = reasoning

    return metrics


def _set_usage_attrs(span: trace.Span, usage: Any) -> None:
    metrics = _extract_usage(usage)
    if not metrics:
        return
    if "prompt_tokens" in metrics:
        span.set_attribute(OI_LLM_TOKEN_COUNT_PROMPT, metrics["prompt_tokens"])
    if "completion_tokens" in metrics:
        span.set_attribute(OI_LLM_TOKEN_COUNT_COMPLETION, metrics["completion_tokens"])
    if "total_tokens" in metrics:
        span.set_attribute(OI_LLM_TOKEN_COUNT_TOTAL, metrics["total_tokens"])
    if "prompt_cached_tokens" in metrics:
        span.set_attribute(OI_LLM_TOKEN_COUNT_PROMPT_CACHE_READ, metrics["prompt_cached_tokens"])
    if "completion_reasoning_tokens" in metrics:
        span.set_attribute(
            OI_LLM_TOKEN_COUNT_COMPLETION_REASONING, metrics["completion_reasoning_tokens"]
        )


# ---------------------------------------------------------------------------
# Span data → OTel attributes
# ---------------------------------------------------------------------------


def _set_response_attrs(otel_span: trace.Span, span: Any) -> None:
    data = span.span_data
    input_val = getattr(data, "input", None)
    if input_val is not None:
        val = _try_json(input_val)
        if val:
            otel_span.set_attribute(OI_INPUT_VALUE, val)
            if not isinstance(input_val, str):
                otel_span.set_attribute(OI_INPUT_MIME_TYPE, MIME_JSON)

    response = getattr(data, "response", None)
    if response is not None:
        output = getattr(response, "output", None)
        if output is not None:
            val = _try_json(output)
            if val:
                otel_span.set_attribute(OI_OUTPUT_VALUE, val)
                otel_span.set_attribute(OI_OUTPUT_MIME_TYPE, MIME_JSON)

        model = getattr(response, "model", None)
        if model:
            otel_span.set_attribute(OI_LLM_MODEL_NAME, model)
            otel_span.set_attribute(GEN_AI_RESPONSE_MODEL, model)

        usage = getattr(response, "usage", None)
        _set_usage_attrs(otel_span, usage)


def _set_generation_attrs(otel_span: trace.Span, span: Any) -> None:
    data = span.span_data
    input_val = getattr(data, "input", None)
    if input_val is not None:
        val = _try_json(input_val)
        if val:
            otel_span.set_attribute(OI_INPUT_VALUE, val)
            otel_span.set_attribute(OI_INPUT_MIME_TYPE, MIME_JSON)

    output = getattr(data, "output", None)
    if output is not None:
        val = _try_json(output)
        if val:
            otel_span.set_attribute(OI_OUTPUT_VALUE, val)
            otel_span.set_attribute(OI_OUTPUT_MIME_TYPE, MIME_JSON)

    model = getattr(data, "model", None)
    if model:
        otel_span.set_attribute(OI_LLM_MODEL_NAME, model)
        otel_span.set_attribute(GEN_AI_RESPONSE_MODEL, model)

    model_config = getattr(data, "model_config", None)
    if isinstance(model_config, dict):
        filtered = {k: v for k, v in model_config.items() if v is not None}
        if filtered:
            otel_span.set_attribute(OI_LLM_INVOCATION_PARAMETERS, json.dumps(filtered))

    _set_usage_attrs(otel_span, getattr(data, "usage", None))


def _set_function_attrs(otel_span: trace.Span, span: Any) -> None:
    data = span.span_data
    name = getattr(data, "name", None)
    if name:
        otel_span.set_attribute(OI_TOOL_NAME, name)

    input_val = getattr(data, "input", None)
    if input_val is not None:
        val = _try_json(input_val)
        if val:
            otel_span.set_attribute(OI_INPUT_VALUE, val)
            otel_span.set_attribute(OI_INPUT_MIME_TYPE, MIME_JSON)

    output = getattr(data, "output", None)
    if output is not None:
        val = _try_json(output)
        if val:
            otel_span.set_attribute(OI_OUTPUT_VALUE, val)


def _set_handoff_attrs(otel_span: trace.Span, span: Any) -> None:
    data = span.span_data
    from_agent = getattr(data, "from_agent", None)
    to_agent = getattr(data, "to_agent", None)
    if from_agent:
        otel_span.set_attribute(GRAPH_NODE_PARENT_ID, from_agent)
    if to_agent:
        otel_span.set_attribute(GRAPH_NODE_ID, to_agent)


def _set_agent_attrs(otel_span: trace.Span, span: Any) -> None:
    data = span.span_data
    name = getattr(data, "name", None)
    if name:
        otel_span.set_attribute(GRAPH_NODE_ID, name)


def _set_mcp_list_tools_attrs(otel_span: trace.Span, span: Any) -> None:
    data = span.span_data
    result = getattr(data, "result", None)
    if result is not None:
        val = _try_json(result)
        if val:
            otel_span.set_attribute(OI_OUTPUT_VALUE, val)
            otel_span.set_attribute(OI_OUTPUT_MIME_TYPE, MIME_JSON)


def _set_span_data_attrs(otel_span: trace.Span, span: Any) -> None:
    stype = getattr(span.span_data, "type", None)
    if stype == "response":
        _set_response_attrs(otel_span, span)
    elif stype == "generation":
        _set_generation_attrs(otel_span, span)
    elif stype == "function":
        _set_function_attrs(otel_span, span)
    elif stype == "handoff":
        _set_handoff_attrs(otel_span, span)
    elif stype == "agent":
        _set_agent_attrs(otel_span, span)
    elif stype == "mcp_tools":
        _set_mcp_list_tools_attrs(otel_span, span)
    elif stype == "guardrail":
        pass
    elif stype in ("transcription", "speech"):
        _set_generation_attrs(otel_span, span)
    elif stype in ("task", "turn"):
        _set_usage_attrs(otel_span, getattr(span.span_data, "usage", None))


# ---------------------------------------------------------------------------
# OTel span status from SDK span error
# ---------------------------------------------------------------------------


def _get_span_status(span: Any) -> tuple[StatusCode, str | None]:
    error = getattr(span, "error", None)
    if error:
        msg = error.get("message", "") if isinstance(error, dict) else str(error)
        data = error.get("data", "") if isinstance(error, dict) else ""
        desc = f"{msg}: {data}" if data else msg
        return StatusCode.ERROR, desc
    return StatusCode.OK, None


# ---------------------------------------------------------------------------
# TracingProcessor implementation
# ---------------------------------------------------------------------------


class _TracerootTracingProcessor:
    """Converts OpenAI Agent SDK trace events into OTel spans."""

    def __init__(self, tracer: trace.Tracer) -> None:
        self._tracer = tracer
        self._root_spans: dict[str, trace.Span] = {}
        self._root_tokens: dict[str, object] = {}
        self._otel_spans: dict[str, trace.Span] = {}
        self._span_tokens: dict[str, object] = {}
        self._first_input: dict[str, Any] = {}
        self._last_output: dict[str, Any] = {}

    def on_trace_start(self, sdk_trace: Any) -> None:
        otel_span = self._tracer.start_span(
            name=sdk_trace.name,
            attributes={
                OI_SPAN_KIND: "AGENT",
                OI_LLM_SYSTEM: "openai",
            },
        )
        ctx = trace.set_span_in_context(otel_span)
        token = otel_context.attach(ctx)
        self._root_spans[sdk_trace.trace_id] = otel_span
        self._root_tokens[sdk_trace.trace_id] = token

    def on_trace_end(self, sdk_trace: Any) -> None:
        otel_span = self._root_spans.pop(sdk_trace.trace_id, None)
        if not otel_span:
            return

        first_input = self._first_input.pop(sdk_trace.trace_id, None)
        if first_input is not None:
            val = _try_json(first_input)
            if val:
                otel_span.set_attribute(OI_INPUT_VALUE, val)

        last_output = self._last_output.pop(sdk_trace.trace_id, None)
        if last_output is not None:
            val = _try_json(last_output)
            if val:
                otel_span.set_attribute(OI_OUTPUT_VALUE, val)

        otel_span.set_status(StatusCode.OK)
        otel_span.end()

        token = self._root_tokens.pop(sdk_trace.trace_id, None)
        if token is not None:
            otel_context.detach(token)

    def on_span_start(self, span: Any) -> None:
        start_ns = _iso_to_ns(getattr(span, "started_at", None))

        if span.parent_id is not None:
            parent_otel = self._otel_spans.get(span.parent_id)
        else:
            parent_otel = self._root_spans.get(span.trace_id)

        parent_ctx = trace.set_span_in_context(parent_otel) if parent_otel else None

        kind = _span_kind(span.span_data)
        name = _span_name(span)

        attrs: dict[str, Any] = {
            OI_SPAN_KIND: kind,
            OI_LLM_SYSTEM: "openai",
        }

        otel_span = self._tracer.start_span(
            name=name,
            context=parent_ctx,
            start_time=start_ns,
            attributes=attrs,
        )
        self._otel_spans[span.span_id] = otel_span

        ctx = trace.set_span_in_context(otel_span, parent_ctx)
        token = otel_context.attach(ctx)
        self._span_tokens[span.span_id] = token

    def on_span_end(self, span: Any) -> None:
        token = self._span_tokens.pop(span.span_id, None)
        if token is not None:
            otel_context.detach(token)

        otel_span = self._otel_spans.pop(span.span_id, None)
        if not otel_span:
            return

        otel_span.update_name(_span_name(span))

        try:
            _set_span_data_attrs(otel_span, span)
        except Exception:
            logger.debug("Failed to set span data attributes", exc_info=True)

        stype = getattr(span.span_data, "type", None)
        start_ns = _iso_to_ns(getattr(span, "started_at", None))
        end_ns = _iso_to_ns(getattr(span, "ended_at", None))

        # For response spans, create a nested openai.responses.create LLM span
        if stype == "response":
            self._create_responses_create_span(otel_span, span, start_ns, end_ns)

        # Track first input / last output for the root trace span
        if stype == "response":
            input_val = getattr(span.span_data, "input", None)
            if span.trace_id not in self._first_input and input_val is not None:
                self._first_input[span.trace_id] = input_val
            response = getattr(span.span_data, "response", None)
            output = getattr(response, "output", None) if response else None
            if output is not None:
                self._last_output[span.trace_id] = output
        elif stype == "function":
            input_val = getattr(span.span_data, "input", None)
            if span.trace_id not in self._first_input and input_val is not None:
                self._first_input[span.trace_id] = input_val
            output = getattr(span.span_data, "output", None)
            if output is not None:
                self._last_output[span.trace_id] = output

        status_code, desc = _get_span_status(span)
        otel_span.set_status(status_code, desc)
        otel_span.end(end_ns)

    def _create_responses_create_span(
        self,
        parent_otel_span: trace.Span,
        sdk_span: Any,
        start_ns: int | None,
        end_ns: int | None,
    ) -> None:
        """Create a nested openai.responses.create LLM span under the Response span."""
        data = sdk_span.span_data
        response = getattr(data, "response", None)
        if response is None:
            return

        parent_ctx = trace.set_span_in_context(parent_otel_span)
        child = self._tracer.start_span(
            name="openai.responses.create",
            context=parent_ctx,
            start_time=start_ns,
            attributes={
                OI_SPAN_KIND: "LLM",
                OI_LLM_SYSTEM: "openai",
            },
        )

        model = getattr(response, "model", None)
        if model:
            child.set_attribute(OI_LLM_MODEL_NAME, model)
            child.set_attribute(GEN_AI_RESPONSE_MODEL, model)

        usage = getattr(response, "usage", None)
        _set_usage_attrs(child, usage)

        input_val = getattr(data, "input", None)
        if input_val is not None:
            val = _try_json(input_val)
            if val:
                child.set_attribute(OI_INPUT_VALUE, val)
                if not isinstance(input_val, str):
                    child.set_attribute(OI_INPUT_MIME_TYPE, MIME_JSON)

        output = getattr(response, "output", None)
        if output is not None:
            val = _try_json(output)
            if val:
                child.set_attribute(OI_OUTPUT_VALUE, val)
                child.set_attribute(OI_OUTPUT_MIME_TYPE, MIME_JSON)

        self._create_tool_call_spans(child, response, start_ns, end_ns)

        status_code, desc = _get_span_status(sdk_span)
        child.set_status(status_code, desc)
        child.end(end_ns)

    def _create_tool_call_spans(
        self,
        parent_span: trace.Span,
        response: Any,
        start_ns: int | None,
        end_ns: int | None,
    ) -> None:
        """Create TOOL child spans for each function_call in the response output."""
        output = getattr(response, "output", None)
        if not output or not isinstance(output, (list, tuple)):
            return

        parent_ctx = trace.set_span_in_context(parent_span)
        for item in output:
            item_type = getattr(item, "type", None)
            if item_type != "function_call":
                continue

            name = getattr(item, "name", None) or "tool_call"
            tool_span = self._tracer.start_span(
                name=name,
                context=parent_ctx,
                start_time=start_ns,
                attributes={
                    OI_SPAN_KIND: "TOOL",
                    OI_LLM_SYSTEM: "openai",
                    OI_TOOL_NAME: name,
                },
            )

            arguments = getattr(item, "arguments", None)
            if arguments is not None:
                tool_span.set_attribute(OI_INPUT_VALUE, str(arguments))

            call_id = getattr(item, "call_id", None)
            if call_id:
                tool_span.set_attribute("tool.call_id", call_id)

            tool_span.set_status(StatusCode.OK)
            tool_span.end(end_ns)

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Public entry point — called by registry
# ---------------------------------------------------------------------------

_INSTRUMENTED = False


def instrument_openai_agents(tracer_provider: Any = None) -> None:
    """Register a TracingProcessor with the OpenAI Agent SDK."""
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return

    import agents as agents_mod
    from agents import tracing as agents_tracing

    if tracer_provider is not None:
        tracer = tracer_provider.get_tracer(TRACER_NAME)
    else:
        tracer = trace.get_tracer(TRACER_NAME)

    processor = _TracerootTracingProcessor(tracer)

    existing = _get_processors(agents_tracing)
    if not any(isinstance(p, _TracerootTracingProcessor) for p in existing):
        agents_mod.add_trace_processor(processor)

    agents_mod.set_tracing_disabled(False)
    _INSTRUMENTED = True


def _get_processors(tracing_mod: Any) -> tuple[Any, ...]:
    provider = tracing_mod.get_trace_provider()
    multi = getattr(provider, "_multi_processor", None)
    return getattr(multi, "_processors", ()) if multi else ()
