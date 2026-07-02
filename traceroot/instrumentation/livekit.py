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


_LIVEKIT_PROVIDER: object | None = None
_HIJACK_WARNING_EMITTED = False


def _remember_livekit_provider(provider: object) -> None:
    global _LIVEKIT_PROVIDER
    _LIVEKIT_PROVIDER = provider


def warn_if_livekit_provider_hijacked(expected_provider: object | None = None) -> bool:
    """Warn once if LiveKit Cloud re-bound tracing away from TraceRoot."""
    global _HIJACK_WARNING_EMITTED

    expected = expected_provider or _LIVEKIT_PROVIDER
    if expected is None:
        return False

    try:
        from livekit.agents import telemetry
    except Exception:
        return False

    tracer = getattr(telemetry, "tracer", None)
    current = getattr(tracer, "_tracer_provider", None)
    if current is None or current is expected:
        return False

    if not _HIJACK_WARNING_EMITTED:
        logger.warning(
            'TraceRoot: LiveKit appears to have re-bound the tracer provider. '
            'No LiveKit spans may reach TraceRoot. If running on LiveKit Cloud, '
            'pass record={"traces": False} to session.start().'
        )
        _HIJACK_WARNING_EMITTED = True
    return True


class LiveKitInstrumentor:
    """Routes LiveKit Agents telemetry spans through TraceRoot's TracerProvider."""

    _instrumented: bool = False

    def instrument(self, tracer_provider: TracerProvider | None = None, **_kwargs: object) -> None:
        if LiveKitInstrumentor._instrumented:
            return
        if tracer_provider is None:
            return

        from livekit.agents.telemetry import set_tracer_provider

        tracer_provider.add_span_processor(LiveKitToOpenInferenceProcessor())
        set_tracer_provider(tracer_provider)
        _remember_livekit_provider(tracer_provider)
        LiveKitInstrumentor._instrumented = True

    def uninstrument(self, **_kwargs: object) -> None:
        LiveKitInstrumentor._instrumented = False
