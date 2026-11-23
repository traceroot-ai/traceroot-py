"""LLM usage per evaluation case, read from the LLM spans the SDK's own instrumentation creates.

The engine opens a scope around the task call and around each scorer call. ``UsageCollector`` (a
span processor) reads that scope when a span starts and, when an LLM span ends, adds its token
counts and cost to the case's ``task`` or ``scorers`` bucket. Usage is never inferred from what a
task returns.

Capture is opt-in per process: the evaluation runner arms it around a suite (``armed()``). An
armed client adds the collector to its provider, and a client that does not export (a local run)
builds a provider whose only job is to feed the collector. ``capture_active()`` says whether a run
can see its LLM calls at all; when it can't, every usage field is ``None`` ("not captured"), never
zeros, since 0 would read as "made no LLM calls".
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor

# Attribute names, first valid one wins per field (never summed across conventions).
_INPUT_KEYS = ("llm.token_count.prompt", "gen_ai.usage.input_tokens")
_OUTPUT_KEYS = ("llm.token_count.completion", "gen_ai.usage.output_tokens")
_TOTAL_KEYS = ("llm.token_count.total",)
_CACHE_READ_KEYS = (
    "llm.token_count.prompt_details.cache_read",
    "gen_ai.usage.cache_read_input_tokens",
)
_CACHE_WRITE_KEYS = (
    "llm.token_count.prompt_details.cache_write",
    "llm.token_count.prompt_details.cache_creation",
    "gen_ai.usage.cache_creation_input_tokens",
)
_REASONING_KEYS = ("llm.token_count.completion_details.reasoning",)
_COST_KEY = "llm.cost.total"

_TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
)
_COUNT_FIELDS = ("llm_calls", "llm_calls_with_usage", "llm_calls_with_cost")

TASK = "task"
SCORER = "scorer"
_BUCKET_OF_ROLE = {TASK: "task", SCORER: "scorers"}


# =============================================================================
# Reading one call and summing buckets
# =============================================================================
def _token(value: Any) -> int | None:
    """A token count, or None when the value is not a finite non-negative number. A float is
    rounded half up, the same as JavaScript's Math.round for non-negative values."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0:
        return math.floor(value + 0.5)
    return None


def _cost(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        amount = float(value)
    except OverflowError:
        return None
    return amount if math.isfinite(amount) and amount >= 0 else None


def _first_token(attrs: Mapping[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = _token(attrs.get(key))
        if value is not None:
            return value
    return None


def read_call(attrs: Mapping[str, Any]) -> dict[str, Any]:
    """The usage one LLM span reports: each token field and the cost, None where absent."""
    input_tokens = _first_token(attrs, _INPUT_KEYS)
    output_tokens = _first_token(attrs, _OUTPUT_KEYS)
    total_tokens = _first_token(attrs, _TOTAL_KEYS)
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cache_read_tokens": _first_token(attrs, _CACHE_READ_KEYS),
        "cache_write_tokens": _first_token(attrs, _CACHE_WRITE_KEYS),
        "reasoning_tokens": _first_token(attrs, _REASONING_KEYS),
        "cost": _cost(attrs.get(_COST_KEY)),
    }


def empty_bucket() -> dict[str, Any]:
    """A bucket for a role that made no instrumented LLM call."""
    return {
        "llm_calls": 0,
        "llm_calls_with_usage": 0,
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "cache_read_tokens": None,
        "cache_write_tokens": None,
        "reasoning_tokens": None,
        "cost": None,
        "llm_calls_with_cost": 0,
        "currency": None,
    }


def _plus(a: Any, b: Any) -> Any:
    """Sum of two optional amounts: None only when both are None."""
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _add_call(bucket: dict[str, Any], call: Mapping[str, Any]) -> None:
    bucket["llm_calls"] += 1
    if call["input_tokens"] is not None or call["output_tokens"] is not None:
        bucket["llm_calls_with_usage"] += 1
    for field in _TOKEN_FIELDS:
        bucket[field] = _plus(bucket[field], call[field])
    if call["cost"] is not None:
        bucket["cost"] = _plus(bucket["cost"], call["cost"])
        bucket["llm_calls_with_cost"] += 1
    bucket["currency"] = "USD" if bucket["cost"] is not None else None


def bucket_of(calls: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The bucket for a sequence of calls (each as returned by ``read_call``)."""
    bucket = empty_bucket()
    for call in calls:
        _add_call(bucket, call)
    return bucket


def merge_buckets(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    merged = empty_bucket()
    for field in _COUNT_FIELDS:
        merged[field] = a[field] + b[field]
    for field in (*_TOKEN_FIELDS, "cost"):
        merged[field] = _plus(a[field], b[field])
    merged["currency"] = "USD" if merged["cost"] is not None else None
    return merged


def sum_usage(case_usages: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """A run's usage: each role's buckets summed over its cases (cases without usage skipped)."""
    task, scorers = empty_bucket(), empty_bucket()
    for usage in case_usages:
        if usage is None:
            continue
        task = merge_buckets(task, usage["task"])
        scorers = merge_buckets(scorers, usage["scorers"])
    return {"task": task, "scorers": scorers}


# =============================================================================
# Per-case scope
# =============================================================================
class CaseUsage:
    """Accumulates one case's LLM calls, split by role. ``take()`` returns the usage and closes
    the case, so a call that ends after its case completed is not counted."""

    def __init__(self, case_id: str, *, local: bool = False) -> None:
        self.case_id = case_id
        self.local = local
        self._lock = threading.Lock()
        self._buckets = {"task": empty_bucket(), "scorers": empty_bucket()}
        self._closed = False

    def add(self, role: str, call: Mapping[str, Any]) -> None:
        with self._lock:
            if not self._closed:
                _add_call(self._buckets[_BUCKET_OF_ROLE[role]], call)

    def take(self) -> dict[str, Any]:
        with self._lock:
            self._closed = True
            return {"task": dict(self._buckets["task"]), "scorers": dict(self._buckets["scorers"])}


_SCOPE: contextvars.ContextVar[tuple[CaseUsage, str] | None] = contextvars.ContextVar(
    "traceroot_eval_usage_scope", default=None
)


@contextlib.contextmanager
def scope(case: CaseUsage | None, role: str) -> Iterator[None]:
    """Attribute the LLM spans started in this block (and in tasks or threads that copy its
    context) to ``case`` under ``role`` ("task" or "scorer"). A no-op when ``case`` is None.

    In a local run the block also lets instrumentation build spans again: the run suppresses
    tracing, so without this the integrations would never create the LLM spans counted here. The
    local-run marker stays set, so nothing started in the block can reach a traceroot exporter.
    """
    if case is None:
        yield
        return
    token = _SCOPE.set((case, role))
    try:
        if case.local:
            from traceroot.transport.span_processor import allow_local_capture

            with allow_local_capture():
                yield
        else:
            yield
    finally:
        _SCOPE.reset(token)


# =============================================================================
# Span processor
# =============================================================================
def _is_llm_span(attrs: Mapping[str, Any]) -> bool:
    return (
        attrs.get("openinference.span.kind") == "LLM" or attrs.get("traceroot.span.type") == "llm"
    )


class _OpenSpan:
    __slots__ = ("case", "has_llm_descendant", "parent_id", "role")

    def __init__(self, case: CaseUsage, role: str, parent_id: int | None) -> None:
        self.case = case
        self.role = role
        self.parent_id = parent_id
        self.has_llm_descendant = False


class UsageCollector(SpanProcessor):
    """Counts the leaf LLM spans started inside a usage scope.

    Every span started in a scope is remembered with its parent until it ends, so an LLM span that
    ends can mark its open ancestors as having an LLM descendant. A marked LLM span is a wrapper
    around the provider call and is not counted again. AGENT, CHAIN and TOOL spans are never
    counted, whatever token totals they carry. Exports nothing.
    """

    # Spans that start in a scope but never end are forgotten oldest first past this many.
    _MAX_OPEN = 65536

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: OrderedDict[int, _OpenSpan] = OrderedDict()

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        current = _SCOPE.get()
        if current is None:
            return
        parent = getattr(span, "parent", None)
        parent_id = parent.span_id if parent is not None and parent.is_valid else None
        entry = _OpenSpan(current[0], current[1], parent_id)
        with self._lock:
            if len(self._open) >= self._MAX_OPEN:
                self._open.popitem(last=False)
            self._open[span.context.span_id] = entry

    def on_end(self, span: Any) -> None:
        with self._lock:
            entry = self._open.pop(span.context.span_id, None)
            if entry is None:
                return
            attrs = span.attributes or {}
            if not _is_llm_span(attrs):
                return
            ancestor_id = entry.parent_id
            while ancestor_id is not None:
                ancestor = self._open.get(ancestor_id)
                if ancestor is None:
                    break
                ancestor.has_llm_descendant = True
                ancestor_id = ancestor.parent_id
            if entry.has_llm_descendant:
                return
        entry.case.add(entry.role, read_call(attrs))

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# =============================================================================
# Arming and attaching
# =============================================================================
_armed_depth = 0
_collector: UsageCollector | None = None
_collector_lock = threading.Lock()
_attached: weakref.WeakSet[Any] = weakref.WeakSet()


def collector() -> UsageCollector:
    """The process-wide collector (one per process, shared by every provider it is added to)."""
    global _collector
    with _collector_lock:
        if _collector is None:
            _collector = UsageCollector()
        return _collector


def is_armed() -> bool:
    return _armed_depth > 0


def attach(provider: Any) -> None:
    """Add the collector to ``provider`` once. It is added after the processors already there,
    so it reads a span after any processor that rewrites spans into OpenInference attributes."""
    with _collector_lock:
        if provider in _attached:
            return
        _attached.add(provider)
    provider.add_span_processor(collector())


@contextlib.contextmanager
def armed() -> Iterator[None]:
    """Capture usage for the evaluations run in this block. A client initialized inside it adds
    the collector to its provider; one already running gets it now."""
    global _armed_depth
    _armed_depth += 1
    try:
        import traceroot

        client = getattr(traceroot, "_client", None)
        provider = getattr(client, "_provider", None)
        if provider is not None:
            attach(provider)
        yield
    finally:
        _armed_depth -= 1


def capture_active(local: bool) -> bool:
    """Whether a run can see its LLM calls: capture is armed, the client instrumented at least one
    integration, and the collector is on that client's provider.

    A local run suppresses tracing, which stops the integrations from building spans. The
    suppression is lifted inside a usage scope only when the client exports nothing (it built a
    provider just for the collector) and that provider is the global one, so no other provider
    receives the spans. Any other local setup reports usage as not captured.
    """
    if not is_armed():
        return False
    import traceroot

    client = getattr(traceroot, "_client", None)
    if client is None or not getattr(client, "_instrumented", None):
        return False
    provider = getattr(client, "_provider", None)
    if provider is None or provider not in _attached:
        return False
    if local:
        return bool(getattr(client, "_capture_only", False)) and (
            trace.get_tracer_provider() is provider
        )
    return True
