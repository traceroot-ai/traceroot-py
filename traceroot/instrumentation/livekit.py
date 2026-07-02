"""LiveKit Agents instrumentation support for TraceRoot."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace.export import SpanProcessor

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

OI_SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

_AGENT_SPANS = {"agent_session", "start_agent_activity", "agent_turn", "resume_agent_activity"}
_LLM_SPANS = {"llm_request", "llm_request_run", "llm_node"}
_TOOL_SPANS = {"function_tool"}


def _attrs(span: Any) -> dict[str, Any]:
    return dict(getattr(span, "attributes", {}) or {})


def _set_attr(span: Any, key: str, value: Any) -> None:
    if value is None:
        return
    if hasattr(span, "set_attribute"):
        span.set_attribute(key, value)
        return
    raw_attrs = getattr(span, "_attributes", None)
    if raw_attrs is not None:
        raw_attrs[key] = value


def _first(attrs: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value is not None:
            return value
    return None


def _span_kind(span_name: str, attrs: dict[str, Any]) -> str:
    if span_name in _AGENT_SPANS:
        return "AGENT"
    if span_name in _TOOL_SPANS:
        return "TOOL"
    if span_name in _LLM_SPANS or attrs.get("gen_ai.request.model") is not None:
        return "LLM"
    return "CHAIN"


def _normalize_livekit_span(span: Any) -> None:
    attrs = _attrs(span)
    span_name = getattr(span, "name", "") or ""

    _set_attr(span, OI_SPAN_KIND, _span_kind(span_name, attrs))

    input_value = _first(
        attrs,
        (
            "lk.input_text",
            "lk.user_transcript",
            "lk.chat_ctx",
            "lk.user_input",
            "lk.function_tool.arguments",
        ),
    )
    output_value = _first(
        attrs,
        (
            "lk.function_tool.output",
            "lk.response.text",
        ),
    )
    _set_attr(span, INPUT_VALUE, input_value)
    _set_attr(span, OUTPUT_VALUE, output_value)
    _set_attr(span, LLM_MODEL_NAME, attrs.get("gen_ai.request.model"))
    _set_attr(span, LLM_TOKEN_COUNT_PROMPT, attrs.get("gen_ai.usage.input_tokens"))
    _set_attr(span, LLM_TOKEN_COUNT_COMPLETION, attrs.get("gen_ai.usage.output_tokens"))
    _set_attr(span, GEN_AI_TOOL_NAME, attrs.get("lk.function_tool.name"))


class LiveKitToOpenInferenceProcessor(SpanProcessor):
    """Reshape LiveKit native OTel spans into OpenInference attributes."""

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        _normalize_livekit_span(span)

    def on_end(self, span: Any) -> None:
        _normalize_livekit_span(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
