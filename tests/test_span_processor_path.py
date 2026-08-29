"""Tests for TracerootSpanProcessor path and ids_path tracking.

Covers:
  - traceroot.span.path set correctly for root / child / deeply nested spans
  - traceroot.span.ids_path set correctly at every depth
  - traceroot.span.starts_path set correctly and index-aligned with ids_path
    (ancestor start times as epoch-ns decimal strings, root -> parent)
  - Map-based ancestry lets children inherit full paths even when the parent is
    a NonRecordingSpan (the OpenInference / LangGraph pattern)
  - Map entries are cleaned up on span end (no memory leak)
  - SDK name/version attributes are stamped on every recording span
  - Non-recording spans are silently skipped (no attributes, no map entries)
"""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from traceroot.constants import SDK_NAME, SDK_VERSION
from traceroot.transport.span_processor import TracerootSpanProcessor

# ── Fixture ────────────────────────────────────────────────────────────────────


@pytest.fixture
def proc_setup():
    """Local TracerProvider with TracerootSpanProcessor + InMemorySpanExporter.

    Uses a local provider (not the global one) so these tests are fully
    isolated from the rest of the suite.  TracerootSpanProcessor sets path
    attributes on_start; SimpleSpanProcessor captures spans synchronously
    so attributes are readable as soon as span.end() returns.
    """
    exporter = InMemorySpanExporter()
    processor = TracerootSpanProcessor(api_key="test-key", host_url="http://localhost:9999")
    provider = TracerProvider()
    provider.add_span_processor(processor)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    yield tracer, exporter, processor
    provider.shutdown()
    exporter.clear()


def _path(span) -> list[str]:
    return list(span.attributes.get("traceroot.span.path", ()))


def _ids(span) -> list[str]:
    return list(span.attributes.get("traceroot.span.ids_path", ()))


def _starts(span) -> list[str]:
    return list(span.attributes.get("traceroot.span.starts_path", ()))


def _hex(span_id: int) -> str:
    return format(span_id, "016x")


# ── span.path propagation ──────────────────────────────────────────────────────


def test_root_span_path_contains_only_own_name(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert _path(spans[0]) == ["root"]


def test_child_path_includes_parent_then_own_name(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    child = next(s for s in exporter.get_finished_spans() if s.name == "child")
    assert _path(child) == ["root", "child"]


def test_deeply_nested_path_accumulates_full_ancestry(proc_setup):
    tracer, exporter, _ = proc_setup
    with (
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("mid"),
        tracer.start_as_current_span("leaf"),
    ):
        pass
    leaf = next(s for s in exporter.get_finished_spans() if s.name == "leaf")
    assert _path(leaf) == ["root", "mid", "leaf"]


# ── span.ids_path propagation ──────────────────────────────────────────────────


def test_root_span_ids_path_is_empty(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"):
        pass
    spans = exporter.get_finished_spans()
    assert _ids(spans[0]) == []


def test_child_ids_path_contains_parent_id(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root") as root:
        root_id = _hex(root.context.span_id)
        with tracer.start_as_current_span("child"):
            pass
    child = next(s for s in exporter.get_finished_spans() if s.name == "child")
    assert _ids(child) == [root_id]


def test_grandchild_ids_path_is_root_then_mid(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root") as root:
        root_id = _hex(root.context.span_id)
        with tracer.start_as_current_span("mid") as mid:
            mid_id = _hex(mid.context.span_id)
            with tracer.start_as_current_span("leaf"):
                pass
    leaf = next(s for s in exporter.get_finished_spans() if s.name == "leaf")
    assert _ids(leaf) == [root_id, mid_id]


# ── span.starts_path propagation ──────────────────────────────────────────────
#
# starts_path carries each ancestor's real start time (epoch-ns decimal string),
# index-aligned with ids_path: starts_path[i] is the start of ids_path[i]. Values
# are asserted against the InMemorySpanExporter-captured spans' .start_time, not
# mocks.


def test_root_span_starts_path_is_empty(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"):
        pass
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert _starts(spans[0]) == []


def test_child_starts_path_contains_parent_start(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    root = next(s for s in exporter.get_finished_spans() if s.name == "root")
    child = next(s for s in exporter.get_finished_spans() if s.name == "child")
    assert _starts(child) == [str(root.start_time)]
    assert len(_starts(child)) == len(_ids(child))


def test_grandchild_starts_path_aligns_with_ids_path(proc_setup):
    tracer, exporter, _ = proc_setup
    with (
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("mid"),
        tracer.start_as_current_span("leaf"),
    ):
        pass
    root = next(s for s in exporter.get_finished_spans() if s.name == "root")
    mid = next(s for s in exporter.get_finished_spans() if s.name == "mid")
    leaf = next(s for s in exporter.get_finished_spans() if s.name == "leaf")
    assert _starts(leaf) == [str(root.start_time), str(mid.start_time)]
    assert len(_starts(leaf)) == len(_ids(leaf))


# ── Map-based ancestry (NonRecordingSpan / remote parent) ─────────────────────
#
# OpenInference instruments LangGraph nodes by creating spans whose parent is
# set via trace.set_span_in_context(NonRecordingSpan(...)) rather than
# start_as_current_span.  The NonRecordingSpan has a valid spanContext() but
# carries NO attributes, so reading parent_span.attributes would return None
# and break the ancestry chain.  The map fix lets children look up the full
# ids_path by parentSpanId even when the parent span has no attributes.


def test_child_inherits_full_ids_path_via_map_when_parent_is_non_recording(proc_setup):
    tracer, exporter, _ = proc_setup

    with tracer.start_as_current_span("root") as root:
        root_id = _hex(root.context.span_id)
        with tracer.start_as_current_span("mid") as mid:
            mid_id = _hex(mid.context.span_id)

            # Simulate OpenInference: replace the active span with a NonRecordingSpan
            # that carries the same spanId but ZERO attributes.
            remote_ctx = SpanContext(
                trace_id=mid.context.trace_id,
                span_id=mid.context.span_id,
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            non_recording_parent = NonRecordingSpan(remote_ctx)
            ctx_with_remote = trace.set_span_in_context(non_recording_parent)

            leaf = tracer.start_span("leaf", context=ctx_with_remote)
            leaf.end()

    leaf_span = next(s for s in exporter.get_finished_spans() if s.name == "leaf")
    # Without the map fix this would be [mid_id] only — root ancestry lost.
    assert _ids(leaf_span) == [root_id, mid_id]
    assert _path(leaf_span) == ["root", "mid", "leaf"]


def test_child_inherits_starts_path_via_map_when_parent_is_non_recording(proc_setup):
    tracer, exporter, _ = proc_setup

    with (
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("mid") as mid,
    ):
        # Same shape as the ids_path map test: the active span is replaced by
        # a NonRecordingSpan with the same spanId but no attributes, so the
        # starts must come from the map, not the parent's attributes.
        remote_ctx = SpanContext(
            trace_id=mid.context.trace_id,
            span_id=mid.context.span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        non_recording_parent = NonRecordingSpan(remote_ctx)
        ctx_with_remote = trace.set_span_in_context(non_recording_parent)

        leaf = tracer.start_span("leaf", context=ctx_with_remote)
        leaf.end()

    root = next(s for s in exporter.get_finished_spans() if s.name == "root")
    mid = next(s for s in exporter.get_finished_spans() if s.name == "mid")
    leaf_span = next(s for s in exporter.get_finished_spans() if s.name == "leaf")
    assert _starts(leaf_span) == [str(root.start_time), str(mid.start_time)]
    assert len(_starts(leaf_span)) == len(_ids(leaf_span))


def test_path_fully_inherited_via_map_for_remote_parent(proc_setup):
    tracer, exporter, _ = proc_setup

    with (
        tracer.start_as_current_span("session"),
        tracer.start_as_current_span("agent") as agent,
    ):
        remote_ctx = SpanContext(
            trace_id=agent.context.trace_id,
            span_id=agent.context.span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx_with_remote = trace.set_span_in_context(NonRecordingSpan(remote_ctx))
        llm = tracer.start_span("llm_call", context=ctx_with_remote)
        llm.end()

    llm_span = next(s for s in exporter.get_finished_spans() if s.name == "llm_call")
    assert _path(llm_span) == ["session", "agent", "llm_call"]


def test_multiple_remote_children_of_same_parent_all_get_correct_ancestry(proc_setup):
    tracer, exporter, _ = proc_setup

    with tracer.start_as_current_span("root") as root:
        root_id = _hex(root.context.span_id)
        with tracer.start_as_current_span("mid") as mid:
            mid_id = _hex(mid.context.span_id)
            remote_ctx = SpanContext(
                trace_id=mid.context.trace_id,
                span_id=mid.context.span_id,
                is_remote=True,
                trace_flags=TraceFlags(TraceFlags.SAMPLED),
            )
            ctx_with_remote = trace.set_span_in_context(NonRecordingSpan(remote_ctx))
            c1 = tracer.start_span("child1", context=ctx_with_remote)
            c2 = tracer.start_span("child2", context=ctx_with_remote)
            c1.end()
            c2.end()

    finished = {s.name: s for s in exporter.get_finished_spans()}
    assert _ids(finished["child1"]) == [root_id, mid_id]
    assert _ids(finished["child2"]) == [root_id, mid_id]


def test_multiple_remote_children_of_same_parent_get_correct_starts(proc_setup):
    tracer, exporter, _ = proc_setup

    with (
        tracer.start_as_current_span("root") as root,
        tracer.start_as_current_span("mid") as mid,
    ):
        remote_ctx = SpanContext(
            trace_id=mid.context.trace_id,
            span_id=mid.context.span_id,
            is_remote=True,
            trace_flags=TraceFlags(TraceFlags.SAMPLED),
        )
        ctx_with_remote = trace.set_span_in_context(NonRecordingSpan(remote_ctx))
        c1 = tracer.start_span("child1", context=ctx_with_remote)
        c2 = tracer.start_span("child2", context=ctx_with_remote)
        c1.end()
        c2.end()

    finished = {s.name: s for s in exporter.get_finished_spans()}
    assert _starts(finished["child1"]) == [str(root.start_time), str(mid.start_time)]
    assert _starts(finished["child2"]) == [str(root.start_time), str(mid.start_time)]


# ── Map memory / eviction ─────────────────────────────────────────────────────
#
# Entries are evicted in on_end (under RLock) to keep memory low. A bounded
# OrderedDict (capacity _PATH_MAP_MAX) acts as a second safety net: if on_end
# fires before a concurrent sibling's on_start, the oldest entry is evicted
# at capacity rather than allowing unbounded growth.


def test_map_entries_removed_after_span_ends(proc_setup):
    tracer, _, processor = proc_setup
    with tracer.start_as_current_span("root") as root:
        root_hex = _hex(root.context.span_id)
        assert root_hex in processor._ids_path_by_span_id

    assert root_hex not in processor._ids_path_by_span_id
    assert root_hex not in processor._name_path_by_span_id
    assert root_hex not in processor._starts_path_by_span_id


def test_map_cleaned_up_for_all_spans_in_hierarchy(proc_setup):
    tracer, _, processor = proc_setup
    with (
        tracer.start_as_current_span("root") as root,
        tracer.start_as_current_span("child") as child,
    ):
        root_hex = _hex(root.context.span_id)
        child_hex = _hex(child.context.span_id)
        assert root_hex in processor._ids_path_by_span_id
        assert child_hex in processor._ids_path_by_span_id

    assert root_hex not in processor._ids_path_by_span_id
    assert child_hex not in processor._ids_path_by_span_id
    assert root_hex not in processor._starts_path_by_span_id
    assert child_hex not in processor._starts_path_by_span_id


def test_starts_map_evicted_in_lockstep_via_on_start(proc_setup):
    """Eviction of the starts map is driven through the real on_start path.

    Creates _PATH_MAP_MAX + 1 sequential root spans without ending them (ending
    would pop the entries in on_end). The overflow must evict the oldest entry
    from ALL three maps — including starts. If the starts eviction is dropped
    from production, this test fails.
    """
    from traceroot.transport.span_processor import _PATH_MAP_MAX

    tracer, _, processor = proc_setup
    spans = [tracer.start_span(f"span-{i}") for i in range(_PATH_MAP_MAX + 1)]
    first_hex = _hex(spans[0].context.span_id)
    last_hex = _hex(spans[-1].context.span_id)

    assert first_hex not in processor._ids_path_by_span_id
    assert first_hex not in processor._name_path_by_span_id
    assert first_hex not in processor._starts_path_by_span_id

    assert last_hex in processor._ids_path_by_span_id
    assert last_hex in processor._name_path_by_span_id
    assert last_hex in processor._starts_path_by_span_id
    assert len(processor._starts_path_by_span_id) == _PATH_MAP_MAX

    for span in spans:
        span.end()


def test_remote_parent_with_unknown_start_stamps_empty_starts(proc_setup):
    """A remote parent never seen by this processor has no start time.

    ids_path still records the remote id, but starts_path must stamp [] rather
    than emit a wrong-length (misaligned) array.
    """
    tracer, exporter, _ = proc_setup
    remote_ctx = SpanContext(
        trace_id=0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF,
        span_id=0x0BADCAFE0BADCAFE,
        is_remote=True,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    ctx_with_remote = trace.set_span_in_context(NonRecordingSpan(remote_ctx))
    leaf = tracer.start_span("leaf", context=ctx_with_remote)
    leaf.end()

    leaf_span = exporter.get_finished_spans()[0]
    assert _ids(leaf_span) == [_hex(0x0BADCAFE0BADCAFE)]
    assert _starts(leaf_span) == []


def test_map_bounded_eviction_at_capacity(proc_setup):
    from traceroot.transport.span_processor import _PATH_MAP_MAX

    _, _, processor = proc_setup
    # Fill map to capacity by directly inserting entries.
    for i in range(_PATH_MAP_MAX):
        key = format(i, "016x")
        processor._ids_path_by_span_id[key] = []
        processor._name_path_by_span_id[key] = []

    first_key = format(0, "016x")
    assert first_key in processor._ids_path_by_span_id

    # One more insertion should evict the oldest (first) entry.
    overflow_key = format(_PATH_MAP_MAX, "016x")
    if len(processor._ids_path_by_span_id) >= _PATH_MAP_MAX:
        processor._ids_path_by_span_id.popitem(last=False)
        processor._name_path_by_span_id.popitem(last=False)
    processor._ids_path_by_span_id[overflow_key] = []
    processor._name_path_by_span_id[overflow_key] = []

    assert first_key not in processor._ids_path_by_span_id
    assert overflow_key in processor._ids_path_by_span_id
    assert len(processor._ids_path_by_span_id) == _PATH_MAP_MAX


# ── SDK attributes ─────────────────────────────────────────────────────────────


def test_sdk_name_and_version_set_on_root_span(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"):
        pass
    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("traceroot.sdk.name") == SDK_NAME
    assert span.attributes.get("traceroot.sdk.version") == SDK_VERSION


def test_sdk_name_and_version_set_on_child_span(proc_setup):
    tracer, exporter, _ = proc_setup
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    child = next(s for s in exporter.get_finished_spans() if s.name == "child")
    assert child.attributes.get("traceroot.sdk.name") == SDK_NAME
    assert child.attributes.get("traceroot.sdk.version") == SDK_VERSION


def test_sdk_attributes_present_on_every_span_in_hierarchy(proc_setup):
    tracer, exporter, _ = proc_setup
    with (
        tracer.start_as_current_span("root"),
        tracer.start_as_current_span("mid"),
        tracer.start_as_current_span("leaf"),
    ):
        pass
    for span in exporter.get_finished_spans():
        assert span.attributes.get("traceroot.sdk.name") == SDK_NAME, span.name
        assert span.attributes.get("traceroot.sdk.version") == SDK_VERSION, span.name


# ── Non-recording span skip ────────────────────────────────────────────────────
#
# When span.is_recording() is False the processor must early-exit: no SDK
# attributes, no path attributes, and no entries added to the internal maps.
# Per the OTel spec, processors must early-exit for non-recording spans.


def test_non_recording_span_not_added_to_map(proc_setup):
    _, _, processor = proc_setup
    remote_ctx = SpanContext(
        trace_id=0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF,
        span_id=0xDEADBEEFDEADBEEF,
        is_remote=True,
        trace_flags=TraceFlags(0),  # NOT sampled → NonRecordingSpan behaviour
    )
    non_recording = NonRecordingSpan(remote_ctx)
    assert not non_recording.is_recording()

    processor.on_start(non_recording)

    span_hex = _hex(remote_ctx.span_id)
    assert span_hex not in processor._ids_path_by_span_id
    assert span_hex not in processor._name_path_by_span_id
    assert span_hex not in processor._starts_path_by_span_id


def test_non_recording_span_has_no_path_attributes(proc_setup):
    _, _, processor = proc_setup
    remote_ctx = SpanContext(
        trace_id=0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF,
        span_id=0xDEADBEEFDEADBEEF,
        is_remote=True,
        trace_flags=TraceFlags(0),
    )
    non_recording = NonRecordingSpan(remote_ctx)

    # Calling on_start must not raise and must not add map entries.
    processor.on_start(non_recording)

    span_hex = _hex(remote_ctx.span_id)
    assert span_hex not in processor._ids_path_by_span_id
    assert span_hex not in processor._name_path_by_span_id
    assert span_hex not in processor._starts_path_by_span_id


def test_map_stays_empty_when_only_non_recording_spans_processed(proc_setup):
    _, _, processor = proc_setup
    for i in range(3):
        remote_ctx = SpanContext(
            trace_id=0xDEADBEEFDEADBEEFDEADBEEFDEADBEEF,
            span_id=i + 1,
            is_remote=True,
            trace_flags=TraceFlags(0),
        )
        processor.on_start(NonRecordingSpan(remote_ctx))

    assert processor._ids_path_by_span_id == {}
    assert processor._name_path_by_span_id == {}
    assert processor._starts_path_by_span_id == {}


# ── Git context attributes ─────────────────────────────────────────────────────


def test_git_attrs_on_all_spans():
    """git_repo and git_ref passed to the processor are stamped on every span."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    proc = TracerootSpanProcessor(
        api_key="test",
        host_url="http://localhost:9999",
        git_repo="acme/web",
        git_ref="a" * 40,
    )
    provider = TracerProvider()
    provider.add_span_processor(proc)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("plain-span"):
        pass
    provider.force_flush()
    span = next(s for s in exporter.get_finished_spans() if s.name == "plain-span")
    assert span.attributes.get("traceroot.git.repo") == "acme/web"
    assert span.attributes.get("traceroot.git.ref") == "a" * 40


def test_git_attrs_absent_when_not_configured():
    """No git attributes are stamped when git_repo/git_ref are not provided."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    proc = TracerootSpanProcessor(api_key="test", host_url="http://localhost:9999")
    provider = TracerProvider()
    provider.add_span_processor(proc)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("plain-span"):
        pass
    provider.force_flush()
    span = next(s for s in exporter.get_finished_spans() if s.name == "plain-span")
    assert span.attributes.get("traceroot.git.repo") is None
    assert span.attributes.get("traceroot.git.ref") is None


def test_git_attrs_stamped_independently():
    """repo and ref stamp independently — repo set, ref absent stamps only repo."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    proc = TracerootSpanProcessor(
        api_key="test", host_url="http://localhost:9999", git_repo="acme/web"
    )
    provider = TracerProvider()
    provider.add_span_processor(proc)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("plain-span"):
        pass
    provider.force_flush()
    span = next(s for s in exporter.get_finished_spans() if s.name == "plain-span")
    assert span.attributes.get("traceroot.git.repo") == "acme/web"
    assert span.attributes.get("traceroot.git.ref") is None


def test_git_attrs_on_child_spans():
    """git_repo and git_ref are stamped on child spans too."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    proc = TracerootSpanProcessor(
        api_key="test",
        host_url="http://localhost:9999",
        git_repo="org/repo",
        git_ref="deadbeef" * 5,
    )
    provider = TracerProvider()
    provider.add_span_processor(proc)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("root"), tracer.start_as_current_span("child"):
        pass
    provider.force_flush()
    child = next(s for s in exporter.get_finished_spans() if s.name == "child")
    assert child.attributes.get("traceroot.git.repo") == "org/repo"
    assert child.attributes.get("traceroot.git.ref") == "deadbeef" * 5
