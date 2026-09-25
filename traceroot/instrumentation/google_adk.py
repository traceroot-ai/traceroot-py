"""Google ADK 2.x workflow/node spans, which OpenInference's ADK instrumentor drops.

ADK 2.x opens ``invoke_workflow`` / ``invoke_node`` spans via ``node_tracing.tracer``, bound at
import time from ``telemetry.tracing.tracer``. OpenInference (<= 0.1.28 at least) swaps the latter
for a proxy that swallows those names, so if ``node_tracing`` is imported after instrumentation
the spans are dropped; if before, they lack ``openinference.span.kind`` and session context.
Rebinding ``node_tracing.tracer`` to :class:`_NodeSpanTracer` emits them once, as CHAIN spans,
whatever the import order.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

from openinference.instrumentation import OITracer, TraceConfig
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as sdk_trace
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from traceroot.constants import SDK_VERSION

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.util.types import Attributes

logger = logging.getLogger(__name__)

_NODE_TRACING_MODULE = "google.adk.telemetry.node_tracing"


def _is_node_span(name: object) -> bool:
    # A bare "invoke_workflow" is an unnamed workflow.
    return isinstance(name, str) and (
        name == "invoke_workflow" or name.startswith(("invoke_workflow ", "invoke_node "))
    )


class _NodeSpanTracer(trace_api.Tracer):
    """Emits ADK workflow/node spans as OpenInference CHAIN spans; delegates all other names."""

    def __init__(self, wrapped: trace_api.Tracer, oi_tracer: trace_api.Tracer) -> None:
        self._wrapped = wrapped
        self._oi_tracer = oi_tracer

    @property
    def wrapped(self) -> trace_api.Tracer:
        return self._wrapped

    def start_span(self, name: str, *args: Any, **kwargs: Any) -> trace_api.Span:
        return self._wrapped.start_span(name, *args, **kwargs)

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: context_api.Context | None = None,
        kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
        attributes: Attributes = None,
        links: Sequence[trace_api.Link] | None = (),
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[trace_api.Span]:
        tracer = self._wrapped
        if _is_node_span(name):
            tracer = self._oi_tracer
            attributes = {
                **(attributes or {}),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            }
        with tracer.start_as_current_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
            end_on_exit=end_on_exit,
        ) as span:
            yield span


def _selective_tracer_emits_node_spans(proxy_cls: type) -> bool:
    """Whether OpenInference's tracer proxy forwards ``invoke_node`` spans (probed in isolation)."""
    provider = sdk_trace.TracerProvider(sampler=ALWAYS_ON, shutdown_on_exit=False)
    probe = proxy_cls(trace_api.NoOpTracer(), provider.get_tracer(__name__))
    token = context_api.attach(context_api.Context())
    try:
        with probe.start_as_current_span("invoke_node traceroot_probe") as span:
            return span.is_recording()
    finally:
        context_api.detach(token)


def _should_replace(binding: object) -> bool:
    """Whether ``node_tracing.tracer`` currently loses or mislabels ADK node spans."""
    # type(), not isinstance(): wrapt proxies report the wrapped object's class to isinstance.
    if type(binding) in (trace_api.ProxyTracer, sdk_trace.Tracer):
        return True
    try:
        from openinference.instrumentation.google_adk import _SelectiveExecuteToolTracer
    except ImportError:
        return False
    if type(binding) is _SelectiveExecuteToolTracer:
        try:
            return not _selective_tracer_emits_node_spans(_SelectiveExecuteToolTracer)
        except Exception:
            logger.debug("Could not probe OpenInference's ADK tracer proxy", exc_info=True)
            return True
    # Something else already owns this binding (e.g. an upstream fix): leave it alone.
    return False


def instrument_adk_node_spans(tracer_provider: TracerProvider | None = None) -> bool:
    """Route ADK 2.x workflow/node spans through an OpenInference tracer on ``tracer_provider``.

    Call after ``GoogleADKInstrumentor().instrument()``. Idempotent. Returns False on ADK 1.x or
    when another instrumentation already emits these spans.
    """
    try:
        from google.adk.telemetry import node_tracing
    except ImportError:
        logger.debug(
            "%s not found (google-adk < 2.0); no workflow spans to route", _NODE_TRACING_MODULE
        )
        return False

    binding = getattr(node_tracing, "tracer", None)
    if isinstance(binding, _NodeSpanTracer):
        return True
    if binding is None or not _should_replace(binding):
        logger.debug(
            "Leaving %s.tracer (%s) in place: already emits node spans, or not a tracer we know",
            _NODE_TRACING_MODULE,
            type(binding).__name__,
        )
        return False

    oi_tracer = OITracer(
        trace_api.get_tracer(__name__, SDK_VERSION, tracer_provider),
        config=TraceConfig(),
    )
    node_tracing.tracer = _NodeSpanTracer(binding, oi_tracer)
    logger.debug("Routing ADK invoke_workflow/invoke_node spans as CHAIN spans")
    return True


def uninstrument_adk_node_spans() -> None:
    """Restore the ``node_tracing`` tracer :func:`instrument_adk_node_spans` replaced."""
    try:
        from google.adk.telemetry import node_tracing
    except ImportError:
        return
    binding = getattr(node_tracing, "tracer", None)
    if isinstance(binding, _NodeSpanTracer):
        node_tracing.tracer = binding.wrapped
