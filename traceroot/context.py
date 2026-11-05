"""Context utilities for accessing current trace/span IDs.

These utility functions allow you to access the current OpenTelemetry
trace and span IDs from anywhere in your code.
"""

import sys

from opentelemetry import trace
from opentelemetry.instrumentation.utils import is_instrumentation_enabled


def _langchain_run_chain() -> list[trace.Span]:
    """Spans of the LangChain/LangGraph run executing the caller: innermost run first, root last.

    OpenInference's LangChain instrumentor creates a span per run from callbacks but never makes it
    the current OpenTelemetry span (attaching context from callbacks could leak it), so code running
    inside a LangGraph node sees no current span. OpenInference exposes the run's spans through its
    own ``get_current_span()`` and ``get_ancestor_spans()`` instead.
    """
    if "langchain_core" not in sys.modules:  # no LangChain run can be active
        return []
    try:
        from openinference.instrumentation.langchain import (
            LangChainInstrumentor,
            get_ancestor_spans,
            get_current_span,
        )
    except ImportError:
        return []
    # Not instrumented: there is no run span to find, and the lookups would print a warning.
    if not LangChainInstrumentor().is_instrumented_by_opentelemetry:
        return []
    try:
        span = get_current_span()
        if span is None or not span.is_recording() or not span.get_span_context().is_valid:
            return []
        return [span, *get_ancestor_spans()]
    except Exception:
        return []


def _run_started_under(chain: list[trace.Span], ambient: trace.Span) -> bool:
    """Whether ``ambient`` is still the span the run tree was started under.

    OpenInference parents a run tree's root span on the span that was current when the run began
    (none, an ``@observe`` entrypoint, or a remote parent). If that is still the current span,
    nothing has been made current inside the run since, so the run is the innermost scope. Any
    other current span was made current inside the run (an ``@observe`` helper in a node, a span
    re-attached with ``use_span``, a dropped or suppressed span) and is the inner one.
    """
    root_parent = getattr(chain[-1], "parent", None)
    ambient_ctx = ambient.get_span_context()
    if not ambient_ctx.is_valid:
        return root_parent is None or not root_parent.is_valid
    return (
        root_parent is not None
        and root_parent.span_id == ambient_ctx.span_id
        and root_parent.trace_id == ambient_ctx.trace_id
    )


def get_active_span() -> trace.Span:
    """The innermost span active for the calling code.

    This is OpenTelemetry's current span, except inside a LangChain or LangGraph run traced by
    OpenInference when nothing has been made current since the run began: then it is the span of
    the innermost run (for example the LangGraph node). Under tracing suppression (such as a
    ``local=True`` evaluation) it is always the current span, so suppressed work never writes to
    an exported run span.
    """
    ambient = trace.get_current_span()
    if not is_instrumentation_enabled():
        return ambient
    chain = _langchain_run_chain()
    if chain and _run_started_under(chain, ambient):
        return chain[0]
    return ambient


def get_trace_update_span() -> trace.Span:
    """Where ``update_current_trace`` writes trace-level attributes.

    The current span when it is recording, as before. Inside a LangChain or LangGraph run started
    with no current span, the run tree's root span, which is the trace's root: TraceRoot reads
    trace metadata and tags from the root span.
    """
    ambient = trace.get_current_span()
    if ambient.is_recording() or not is_instrumentation_enabled():
        return ambient
    chain = _langchain_run_chain()
    if chain and _run_started_under(chain, ambient):
        return chain[-1]
    return ambient


def get_current_trace_id() -> str | None:
    """Get the current trace ID from OpenTelemetry context.

    Inside a traced LangChain or LangGraph run, this is the run's trace.

    Returns:
        The trace ID as a hex string, or None if no active span.

    Example:
        @observe()
        def my_function():
            trace_id = get_current_trace_id()
            print(f"Current trace: {trace_id}")
    """
    span = get_active_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().trace_id, "032x")
    return None


def get_current_span_id() -> str | None:
    """Get the current span ID from OpenTelemetry context.

    Inside a traced LangChain or LangGraph run, this is the innermost run's span (for example the
    LangGraph node).

    Returns:
        The span ID as a hex string, or None if no active span.

    Example:
        @observe()
        def my_function():
            span_id = get_current_span_id()
            print(f"Current span: {span_id}")
    """
    span = get_active_span()
    if span and span.get_span_context().is_valid:
        return format(span.get_span_context().span_id, "016x")
    return None
