"""Adapter instrumentors for integrations that don't fit the standard OpenInference pattern.

Each class exposes a uniform .instrument(tracer_provider=...) interface so the
registry can treat every integration identically via InstrumentorEntry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider


class AutogenInstrumentor:
    """Wraps openinference AutogenInstrumentor, which does not accept tracer_provider."""

    _instrumented: bool = False

    def instrument(self, tracer_provider: TracerProvider | None = None, **_kwargs: object) -> None:
        if AutogenInstrumentor._instrumented:
            return
        from openinference.instrumentation.autogen import AutogenInstrumentor as _Inner

        _Inner().instrument()
        AutogenInstrumentor._instrumented = True

    def uninstrument(self, **_kwargs: object) -> None:
        AutogenInstrumentor._instrumented = False


class PydanticAIInstrumentor:
    """Installs OpenInferenceSpanProcessor and calls Agent.instrument_all().

    pydantic-ai emits native OTel GenAI semconv spans via its logfire integration.
    OpenInferenceSpanProcessor reshapes those into the OpenInference attribute
    schema that the traceroot backend understands.
    """

    _instrumented: bool = False

    def instrument(self, tracer_provider: TracerProvider | None = None, **_kwargs: object) -> None:
        if PydanticAIInstrumentor._instrumented:
            return
        from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
        from pydantic_ai import Agent

        if tracer_provider is not None:
            tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
        Agent.instrument_all()
        PydanticAIInstrumentor._instrumented = True

    def uninstrument(self, **_kwargs: object) -> None:
        PydanticAIInstrumentor._instrumented = False
