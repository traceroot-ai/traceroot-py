"""LiveKit Agents instrumentation support for TraceRoot."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opentelemetry.sdk.trace.export import SpanProcessor

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

logger = logging.getLogger(__name__)

OI_SPAN_KIND = "openinference.span.kind"
TRACEROOT_SPAN_TYPE = "traceroot.span.type"
INPUT_VALUE = "input.value"
OUTPUT_VALUE = "output.value"
LLM_MODEL_NAME = "llm.model_name"
LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"

_AGENT_SPANS = {"agent_turn"}
_LLM_SPANS = {"llm_request"}
_TOOL_SPANS = {"function_tool"}
_INPUT_KEYS = (
    "lk.input_text",
    "lk.user_transcript",
    "lk.chat_ctx",
    "lk.user_input",
    "lk.function_tool.arguments",
)
_OUTPUT_KEYS = (
    "lk.function_tool.output",
    "lk.response.text",
)
_MIRROR_INSTALLED_ATTR = "_traceroot_livekit_attribute_mirror_installed"


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
    if span_name in _LLM_SPANS:
        return "LLM"
    return "SPAN"


def _normalize_livekit_span(span: Any) -> None:
    attrs = _attrs(span)
    span_name = getattr(span, "name", "") or ""

    span_kind = _span_kind(span_name, attrs)
    _set_attr(span, OI_SPAN_KIND, span_kind)
    _set_attr(span, TRACEROOT_SPAN_TYPE, span_kind.lower())
    _set_attr(span, INPUT_VALUE, _first(attrs, _INPUT_KEYS))
    _set_attr(span, OUTPUT_VALUE, _first(attrs, _OUTPUT_KEYS))
    if span_name in _LLM_SPANS:
        _set_attr(span, LLM_MODEL_NAME, attrs.get("gen_ai.request.model"))
        _set_attr(span, LLM_TOKEN_COUNT_PROMPT, attrs.get("gen_ai.usage.input_tokens"))
        _set_attr(span, LLM_TOKEN_COUNT_COMPLETION, attrs.get("gen_ai.usage.output_tokens"))
    _set_attr(span, GEN_AI_TOOL_NAME, attrs.get("lk.function_tool.name"))


def _mirror_livekit_attribute(key: str, value: Any, span_name: str, set_attribute: Any) -> None:
    if key in _INPUT_KEYS:
        set_attribute(INPUT_VALUE, value)
    elif key in _OUTPUT_KEYS:
        set_attribute(OUTPUT_VALUE, value)
    elif key == "gen_ai.request.model" and span_name == "llm_request":
        set_attribute(LLM_MODEL_NAME, value)
    elif key == "gen_ai.usage.input_tokens" and span_name == "llm_request":
        set_attribute(LLM_TOKEN_COUNT_PROMPT, value)
    elif key == "gen_ai.usage.output_tokens" and span_name == "llm_request":
        set_attribute(LLM_TOKEN_COUNT_COMPLETION, value)
    elif key == "lk.function_tool.name":
        set_attribute(GEN_AI_TOOL_NAME, value)


def _install_attribute_mirror(span: Any) -> None:
    if getattr(span, _MIRROR_INSTALLED_ATTR, False):
        return
    original_set_attribute = getattr(span, "set_attribute", None)
    if original_set_attribute is None:
        return

    def set_attribute(key: str, value: Any) -> Any:
        result = original_set_attribute(key, value)
        span_name = getattr(span, "name", "") or ""
        _mirror_livekit_attribute(key, value, span_name, original_set_attribute)
        return result

    span.set_attribute = set_attribute
    setattr(span, _MIRROR_INSTALLED_ATTR, True)


class LiveKitSpanProcessor(SpanProcessor):
    """Lightly annotate LiveKit native OTel spans for TraceRoot display."""

    def on_start(self, span: Any, parent_context: Any | None = None) -> None:
        _normalize_livekit_span(span)
        _install_attribute_mirror(span)

    def on_end(self, span: Any) -> None:
        try:
            _normalize_livekit_span(span)
        except TypeError:
            logger.debug("LiveKit span attributes are immutable on end; skipping remap")

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _add_livekit_processor(tracer_provider: TracerProvider) -> None:
    processor = LiveKitSpanProcessor()
    active_processor = getattr(tracer_provider, "_active_span_processor", None)
    span_processors = getattr(active_processor, "_span_processors", None)
    lock = getattr(active_processor, "_lock", None)

    if isinstance(span_processors, tuple) and lock is not None:
        with lock:
            active_processor._span_processors = (processor,) + active_processor._span_processors
        return

    tracer_provider.add_span_processor(processor)


class LiveKitInstrumentor:
    """Routes LiveKit Agents telemetry spans through TraceRoot's TracerProvider."""

    _instrumented: bool = False

    def instrument(self, tracer_provider: TracerProvider | None = None, **_kwargs: object) -> None:
        if LiveKitInstrumentor._instrumented:
            return
        if tracer_provider is None:
            return

        from livekit.agents.telemetry import set_tracer_provider

        _add_livekit_processor(tracer_provider)
        set_tracer_provider(tracer_provider)
        LiveKitInstrumentor._instrumented = True

    def uninstrument(self, **_kwargs: object) -> None:
        LiveKitInstrumentor._instrumented = False
