"""Google ADK instrumentation: OpenInference's, plus the ADK 2.x workflow/node spans it drops.

ADK 2.x opens ``invoke_workflow`` / ``invoke_node`` spans -- and nothing else -- via
``node_tracing.tracer``, bound at import time from ``telemetry.tracing.tracer``. OpenInference
(<= 0.1.28 at least) swaps the latter for a proxy that swallows those names, so if ``node_tracing``
is imported after instrumentation the spans are dropped; if before, they lack
``openinference.span.kind`` and session context. Rebinding ``node_tracing.tracer`` to an
OpenInference tracer emits them once, as CHAIN spans, whatever the import order; uninstrument()
rebinds it to ADK's own tracer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.google_adk import GoogleADKInstrumentor as _OIInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues
from opentelemetry import trace as trace_api

from traceroot.constants import SDK_VERSION

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider


class _NodeSpanTracer:
    """Stands in for ``node_tracing.tracer``: every span opened through it is a CHAIN span."""

    def __init__(self, oi_tracer: OITracer) -> None:
        self._oi_tracer = oi_tracer

    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Any:
        return self._oi_tracer.start_as_current_span(
            *args, openinference_span_kind=OpenInferenceSpanKindValues.CHAIN, **kwargs
        )


class GoogleADKInstrumentor:
    """OpenInference's GoogleADKInstrumentor, plus ADK 2.x workflow/node spans."""

    def instrument(self, tracer_provider: TracerProvider | None = None, **_kwargs: object) -> None:
        _OIInstrumentor().instrument(tracer_provider=tracer_provider)
        try:
            from google.adk.telemetry import node_tracing
        except ImportError:  # google-adk 1.x: no workflow runtime
            return
        if not isinstance(node_tracing.tracer, _NodeSpanTracer):
            oi_tracer = OITracer(
                trace_api.get_tracer(__name__, SDK_VERSION, tracer_provider), config=TraceConfig()
            )
            node_tracing.tracer = _NodeSpanTracer(oi_tracer)

    def uninstrument(self, **_kwargs: object) -> None:
        _OIInstrumentor().uninstrument()
        try:
            from google.adk.telemetry import node_tracing, tracing
        except ImportError:
            return
        if isinstance(node_tracing.tracer, _NodeSpanTracer):
            # Not the binding we replaced: that may be OpenInference's proxy, which uninstrument()
            # just retired. ADK's own tracer is what it restored to telemetry.tracing.
            node_tracing.tracer = tracing.tracer
