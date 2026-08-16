"""Tests for integration smoothness fixes.

Covers four integration-smoothness fixes:
  1. Warning logged when API key is missing
  2. Re-initialization guard (second initialize() call is a no-op)
  3. Missing instrumentation lib warns + skips instead of crashing
  4. session_id / user_id params on @observe
"""

import logging
import threading
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider

import traceroot
from tests.utils import reset_traceroot
from traceroot import observe
from traceroot.instrumentation.registry import Integration, initialize_integrations
from traceroot.span_attributes import SpanAttributes


@pytest.fixture
def spans(memory_exporter):
    """Alias for memory_exporter, used in smoothness tests."""
    return memory_exporter


# =============================================================================
# Issue 1: Warning when API key is missing
# =============================================================================


def test_warns_when_no_api_key(caplog):
    reset_traceroot()
    with caplog.at_level(logging.WARNING):
        client = traceroot.initialize()

    assert client.enabled is False
    assert "TRACEROOT_API_KEY" in caplog.text


def test_no_warning_when_api_key_provided(caplog):
    reset_traceroot()
    with caplog.at_level(logging.WARNING):
        traceroot.initialize(api_key="test-key", enabled=False)

    assert "TRACEROOT_API_KEY" not in caplog.text


def test_no_warning_when_explicitly_disabled(caplog):
    """No API key warning when tracing is explicitly disabled."""
    reset_traceroot()
    with caplog.at_level(logging.WARNING):
        client = traceroot.initialize(enabled=False)

    assert client.enabled is False
    assert "TRACEROOT_API_KEY" not in caplog.text


# =============================================================================
# Issue 2: Re-initialization guard
# =============================================================================


def test_reinitialize_returns_original_client():
    reset_traceroot()
    client1 = traceroot.initialize(api_key="key1", enabled=False)
    client2 = traceroot.initialize(api_key="key2", enabled=False)
    assert client2 is client1


def test_reinitialize_emits_warning(caplog):
    reset_traceroot()
    traceroot.initialize(api_key="key1", enabled=False)
    with caplog.at_level(logging.WARNING):
        traceroot.initialize(api_key="key2", enabled=False)
    assert "more than once" in caplog.text


def test_reinitialize_does_not_change_config():
    """Second initialize() with different api_key does not overwrite the first."""
    reset_traceroot()
    client1 = traceroot.initialize(api_key="original-key", enabled=False)
    traceroot.initialize(api_key="new-key", enabled=False)
    assert traceroot.get_client().api_key == "original-key"
    assert traceroot.get_client() is client1


def test_reinitialize_after_shutdown_works():
    """After shutdown + reset, initialize() creates a fresh client."""
    reset_traceroot()
    client1 = traceroot.initialize(api_key="key1", enabled=False)
    traceroot.shutdown()
    traceroot._client = None
    client2 = traceroot.initialize(api_key="key2", enabled=False)
    assert client2 is not client1
    assert traceroot.get_client().api_key == "key2"


# =============================================================================
# Issue 3: Missing lib warns + skips (no crash)
# =============================================================================


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_missing_lib_warns_and_skips(mock_installed, caplog):
    mock_installed.return_value = False
    provider = TracerProvider()
    with caplog.at_level(logging.WARNING, logger="traceroot.instrumentation.registry"):
        result = initialize_integrations(provider, [Integration.OPENAI])
    assert result == []
    assert "skipping" in caplog.text
    assert "openai" in caplog.text


@patch("traceroot.instrumentation.registry._is_package_installed")
def test_missing_lib_continues_other_integrations(mock_installed):
    """If one lib is missing, other integrations still get instrumented."""
    from unittest.mock import MagicMock

    def is_installed(pkg):
        return pkg != "openai"

    mock_installed.side_effect = is_installed

    mock_instrumentor = MagicMock()
    mock_cls = MagicMock(return_value=mock_instrumentor)
    mock_module = MagicMock()
    mock_module.AnthropicInstrumentor = mock_cls

    provider = TracerProvider()
    with patch("importlib.import_module", return_value=mock_module):
        result = initialize_integrations(provider, [Integration.OPENAI, Integration.ANTHROPIC])

    # openai skipped, anthropic instrumented
    assert Integration.OPENAI not in result
    assert Integration.ANTHROPIC in result


# =============================================================================
# Issue 4: session_id / user_id on @observe
# =============================================================================


def test_observe_sets_session_id(spans):
    @observe(name="with-session", session_id="sess-abc")
    def func():
        return "ok"

    func()
    span = spans.get_finished_spans()[0]
    assert span.attributes.get(SpanAttributes.TRACE_SESSION_ID) == "sess-abc"


def test_observe_sets_user_id(spans):
    @observe(name="with-user", user_id="user-xyz")
    def func():
        return "ok"

    func()
    span = spans.get_finished_spans()[0]
    assert span.attributes.get(SpanAttributes.TRACE_USER_ID) == "user-xyz"


def test_observe_sets_both_session_and_user_id(spans):
    @observe(name="with-both", session_id="sess-123", user_id="user-456")
    def func():
        return "ok"

    func()
    span = spans.get_finished_spans()[0]
    assert span.attributes.get(SpanAttributes.TRACE_SESSION_ID) == "sess-123"
    assert span.attributes.get(SpanAttributes.TRACE_USER_ID) == "user-456"


def test_observe_without_session_user_sets_nothing(spans):
    @observe(name="no-ids")
    def func():
        return "ok"

    func()
    span = spans.get_finished_spans()[0]
    assert span.attributes.get(SpanAttributes.TRACE_SESSION_ID) is None
    assert span.attributes.get(SpanAttributes.TRACE_USER_ID) is None


@pytest.mark.asyncio
async def test_observe_async_sets_session_and_user_id(spans):
    @observe(name="async-with-ids", session_id="async-sess", user_id="async-user")
    async def async_func():
        return "ok"

    await async_func()
    span = spans.get_finished_spans()[0]
    assert span.attributes.get(SpanAttributes.TRACE_SESSION_ID) == "async-sess"
    assert span.attributes.get(SpanAttributes.TRACE_USER_ID) == "async-user"


# =============================================================================
# Streaming: span deferred until generator exhausted (currently FAILING)
# =============================================================================


def test_sync_generator_span_closes_after_iteration(spans):
    """Span on a generator function must stay open until the generator is exhausted."""

    @observe(name="sync-stream")
    def stream():
        yield "a"
        yield "b"
        yield "c"

    gen = stream()

    # Span should NOT be finished yet — generator not started
    assert len(spans.get_finished_spans()) == 0

    result = list(gen)

    # Now span should be closed and output captured
    finished = spans.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "sync-stream"
    assert result == ["a", "b", "c"]

    output = finished[0].attributes.get(SpanAttributes.SPAN_OUTPUT)
    assert output is not None
    assert "a" in output
    assert "b" in output


@pytest.mark.asyncio
async def test_async_generator_span_closes_after_iteration(spans):
    """Span on an async generator must stay open until the generator is exhausted."""

    @observe(name="async-stream")
    async def async_stream():
        yield "x"
        yield "y"

    gen = async_stream()

    assert len(spans.get_finished_spans()) == 0

    result = [item async for item in gen]

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    assert finished[0].name == "async-stream"
    assert result == ["x", "y"]

    output = finished[0].attributes.get(SpanAttributes.SPAN_OUTPUT)
    assert output is not None
    assert "x" in output


def test_generator_span_closes_on_early_exit(spans):
    """Span must close even if generator is abandoned mid-way (GC'd)."""

    @observe(name="early-exit-stream")
    def stream():
        yield "first"
        yield "second"

    gen = stream()
    next(gen)  # consume one item, then abandon

    # Force close by deleting
    del gen

    finished = spans.get_finished_spans()
    assert len(finished) == 1


# =============================================================================
# Thread-safe init: concurrent initialize() must produce exactly one client
# =============================================================================


def test_concurrent_initialize_returns_same_client():
    """Concurrent calls to initialize() must return the same singleton client."""
    reset_traceroot()

    results = []
    errors = []
    barrier = threading.Barrier(10)

    def init():
        barrier.wait()  # all threads start simultaneously
        try:
            client = traceroot.initialize(api_key="test-key", enabled=False)
            results.append(id(client))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=init) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    # All threads must have gotten the exact same client object
    assert len(set(results)) == 1


# =============================================================================
# Generator edge cases
# =============================================================================


def test_sync_generator_error_mid_stream(spans):
    """Span is closed with ERROR status when a sync generator raises mid-stream."""
    from opentelemetry.trace import StatusCode

    @observe(name="error-sync-stream")
    def stream():
        yield "item-0"
        yield "item-1"
        raise ValueError("boom")

    gen = stream()
    collected = []
    with pytest.raises(ValueError, match="boom"):
        for item in gen:
            collected.append(item)

    # The two items before the error must have been delivered
    assert collected == ["item-0", "item-1"]

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "error-sync-stream"
    # Span must be closed and marked ERROR
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_generator_error_mid_stream(spans):
    """Span is closed with ERROR status when an async generator raises mid-stream."""
    from opentelemetry.trace import StatusCode

    @observe(name="error-async-stream")
    async def async_stream():
        yield "item-0"
        yield "item-1"
        raise ValueError("async-boom")

    gen = async_stream()
    collected = []
    with pytest.raises(ValueError, match="async-boom"):
        async for item in gen:
            collected.append(item)

    assert collected == ["item-0", "item-1"]

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "error-async-stream"
    assert span.status.status_code == StatusCode.ERROR


def test_empty_sync_generator(spans):
    """A sync generator that yields nothing still creates and closes exactly one span."""

    @observe(name="empty-sync-stream")
    def stream():
        return
        yield  # make this a generator function

    result = list(stream())

    assert result == []

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "empty-sync-stream"
    # No items were yielded, so no output should be captured
    assert span.attributes.get(SpanAttributes.SPAN_OUTPUT) is None


@pytest.mark.asyncio
async def test_empty_async_generator(spans):
    """An async generator that yields nothing still creates and closes exactly one span."""

    @observe(name="empty-async-stream")
    async def async_stream():
        return
        yield  # make this an async generator function

    result = [item async for item in async_stream()]

    assert result == []

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    assert span.name == "empty-async-stream"
    # No items were yielded, so no output should be captured
    assert span.attributes.get(SpanAttributes.SPAN_OUTPUT) is None


def test_generator_as_input_does_not_crash(spans):
    """Passing a generator object as an argument must not crash input capture."""

    @observe(name="takes-generator-arg")
    def func(gen_arg):
        # Consume the generator so the function does real work
        return list(gen_arg)

    def make_gen():
        yield 1
        yield 2

    result = func(make_gen())
    assert result == [1, 2]

    finished = spans.get_finished_spans()
    assert len(finished) == 1
    span = finished[0]
    # Span must have been created successfully
    assert span.name == "takes-generator-arg"
    # Input must be captured as some string representation — not None
    input_attr = span.attributes.get(SpanAttributes.SPAN_INPUT)
    assert input_attr is not None
    assert isinstance(input_attr, str)


def test_nested_generator_creates_parent_child_spans(spans):
    """An outer regular function that iterates an inner generator produces two spans
    where the inner span's parent is the outer span."""

    @observe(name="inner-stream")
    def inner():
        yield "a"
        yield "b"

    @observe(name="outer")
    def outer():
        return list(inner())

    result = outer()
    assert result == ["a", "b"]

    finished = spans.get_finished_spans()
    assert len(finished) == 2

    spans_by_name = {s.name: s for s in finished}
    outer_span = spans_by_name["outer"]
    inner_span = spans_by_name["inner-stream"]

    # Outer span has no parent (it is the root)
    assert outer_span.parent is None
    # Inner span's parent must be the outer span
    assert inner_span.parent is not None
    assert inner_span.parent.span_id == outer_span.context.span_id


def test_concurrent_initialize_only_one_client_created():
    """TracerootClient constructor must be called exactly once under concurrency."""
    reset_traceroot()

    creation_count = []
    original_init = traceroot.client.TracerootClient.__init__

    def counting_init(self, **kwargs):
        creation_count.append(1)
        original_init(self, **kwargs)

    barrier2 = threading.Barrier(10)

    def init():
        barrier2.wait()
        traceroot.initialize(api_key="test-key", enabled=False)

    with patch.object(traceroot.client.TracerootClient, "__init__", counting_init):
        threads = [threading.Thread(target=init) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(creation_count) == 1
