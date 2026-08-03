"""The evaluation span tracer — the OpenTelemetry tracer used to export per-case eval spans.

Split into its own module so the span-export layer owns a file. Evaluation is cloud-only, so a
run always exports its per-case spans, which are then linked to the reported results.
"""

from __future__ import annotations

from opentelemetry import trace

from traceroot.constants import SDK_VERSION, TRACEROOT_TRACER_NAME


def eval_tracer():
    """The global production tracer for evaluation spans."""
    return trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
