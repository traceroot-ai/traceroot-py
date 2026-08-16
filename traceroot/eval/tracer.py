"""The evaluation span tracer — the OpenTelemetry tracer used to export per-case eval spans.

Split into its own module so the span-export layer owns a file. A reported run exports its
per-case spans, which are then linked to the reported results; a ``local=True`` run gets a
non-exporting tracer instead, because the spans carry the case input/output and local promises
that nothing leaves the process.
"""

from __future__ import annotations

from opentelemetry import trace

from traceroot.constants import SDK_VERSION, TRACEROOT_TRACER_NAME

# Built once: a no-op tracer records nothing and exports nothing, whatever provider the process
# has configured. Its spans accept attributes silently and their span context is INVALID, which
# is exactly what a local run needs (there is no exported trace for an item to link to).
_NOOP_TRACER = trace.NoOpTracerProvider().get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)


def eval_tracer(local: bool = False):
    """The tracer for evaluation spans: non-exporting for a local run, else the global one."""
    if local:
        return _NOOP_TRACER
    return trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
