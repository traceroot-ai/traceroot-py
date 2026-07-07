"""Decorator-based instrumentation using OpenTelemetry."""

import functools
import inspect
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from openinference.instrumentation import get_attributes_from_context
from opentelemetry import context, trace
from opentelemetry.trace import Status, StatusCode

from traceroot.constants import SDK_VERSION, TRACEROOT_TRACER_NAME, SpanKind
from traceroot.git_context import capture_source_location
from traceroot.span_attributes import SpanAttributes
from traceroot.utils import serialize_value, set_span_attribute

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _ensure_initialized() -> None:
    """Ensure traceroot client is initialized (for auto-init from env vars).

    This enables lazy initialization - the @observe decorator works
    without explicit traceroot.initialize() if env vars are set.

    Note: This doesn't block tracing if client is disabled. The decorator
    always creates spans using whatever TracerProvider is configured.
    """
    # Import here to avoid circular import
    from traceroot import get_client

    get_client()  # auto-initializes if needed


def observe(
    name: str | None = None,
    type: SpanKind = SpanKind.SPAN,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    capture_input: bool = True,
    capture_output: bool = True,
    session_id: str | None = None,
    user_id: str | None = None,
) -> Callable[[F], F]:
    """Decorator to create an OpenTelemetry span for a function.

    Args:
        name: Span name. Defaults to function name.
        type: Span kind. Valid values: 'llm', 'span', 'agent', 'tool'.
            - 'llm': For LLM/generation calls
            - 'span': General span (default)
            - 'agent': For agent operations
            - 'tool': For tool/function calls
        metadata: Static metadata to attach.
        tags: Tags to attach.
        capture_input: Whether to capture function arguments.
        capture_output: Whether to capture return value.
        session_id: Session identifier attached to the trace.
        user_id: User identifier attached to the trace.

    Returns:
        Decorated function.

    Example:
        @observe(name="my_agent", type="agent")
        def my_agent(query: str) -> str:
            return process(query)

        @observe(type="tool")
        def search_web(query: str) -> list[str]:
            return results

        @observe(type="llm")
        def call_openai(messages: list) -> str:
            return response
    """
    # Validate type parameter — accept raw strings too
    try:
        validated_kind = SpanKind(type)
    except ValueError:
        valid = ", ".join(m.value for m in SpanKind)
        logger.warning(
            f"Invalid span kind '{type}'. Valid kinds are: {valid}. Defaulting to 'span'."
        )
        validated_kind = SpanKind.SPAN

    def decorator(func: F) -> F:
        span_name = name or func.__name__

        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                _ensure_initialized()
                tracer = trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
                span = tracer.start_span(span_name)
                _set_span_attributes(
                    span,
                    validated_kind,
                    metadata,
                    tags,
                    args,
                    kwargs,
                    func,
                    capture_input,
                    session_id,
                    user_id,
                )
                _set_source_context(span)
                gen = func(*args, **kwargs)
                async for item in _wrap_async_generator(gen, span, capture_output):
                    yield item

            return async_gen_wrapper  # type: ignore[return-value]

        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def sync_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                _ensure_initialized()
                tracer = trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
                span = tracer.start_span(span_name)
                _set_span_attributes(
                    span,
                    validated_kind,
                    metadata,
                    tags,
                    args,
                    kwargs,
                    func,
                    capture_input,
                    session_id,
                    user_id,
                )
                _set_source_context(span)
                gen = func(*args, **kwargs)
                yield from _wrap_sync_generator(gen, span, capture_output)

            return sync_gen_wrapper  # type: ignore[return-value]

        elif inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _ensure_initialized()
                tracer = trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(
                        span,
                        validated_kind,
                        metadata,
                        tags,
                        args,
                        kwargs,
                        func,
                        capture_input,
                        session_id,
                        user_id,
                    )
                    _set_source_context(span)

                    try:
                        result = await func(*args, **kwargs)
                        if capture_output and result is not None:
                            _set_output(span, result)
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return async_wrapper  # type: ignore[return-value]

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _ensure_initialized()
                tracer = trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
                with tracer.start_as_current_span(span_name) as span:
                    _set_span_attributes(
                        span,
                        validated_kind,
                        metadata,
                        tags,
                        args,
                        kwargs,
                        func,
                        capture_input,
                        session_id,
                        user_id,
                    )
                    _set_source_context(span)

                    try:
                        result = func(*args, **kwargs)
                        if capture_output and result is not None:
                            _set_output(span, result)
                        return result
                    except Exception as e:
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.record_exception(e)
                        raise

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def _wrap_sync_generator(
    gen: Any,
    span: trace.Span,
    capture_output: bool,
) -> Any:
    """Yield items from a sync generator, keeping the span open until exhausted.

    The span's context is attached only while the generator body is actively
    executing (between resumptions), and detached again before control
    returns to the caller at each yield. This ensures spans created inside
    the generator body (nested @observe calls, auto-instrumented LLM calls)
    correctly nest under this generator's span, while the caller's code
    between yields does not inherit this context.
    """
    span_ctx = trace.set_span_in_context(span)
    collected: list[Any] = []
    try:
        while True:
            token = context.attach(span_ctx)
            try:
                item = next(gen)
            except StopIteration:
                break
            finally:
                context.detach(token)
            collected.append(item)
            yield item
        if capture_output and collected:
            _set_output(span, collected)
    except GeneratorExit:
        raise
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        span.end()


async def _wrap_async_generator(
    gen: Any,
    span: trace.Span,
    capture_output: bool,
) -> Any:
    """Yield items from an async generator, keeping the span open until exhausted.

    The span's context is attached only while the generator body is actively
    executing (between resumptions), and detached again before control
    returns to the caller at each yield. This ensures spans created inside
    the generator body (nested @observe calls, auto-instrumented LLM calls)
    correctly nest under this generator's span, while the caller's code
    between yields does not inherit this context.
    """
    span_ctx = trace.set_span_in_context(span)
    agen = gen.__aiter__()
    collected: list[Any] = []
    try:
        while True:
            token = context.attach(span_ctx)
            try:
                item = await agen.__anext__()
            except StopAsyncIteration:
                break
            finally:
                context.detach(token)
            collected.append(item)
            yield item
        if capture_output and collected:
            _set_output(span, collected)
    except Exception as e:
        span.set_status(Status(StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
    finally:
        span.end()


def _set_source_context(span: trace.Span) -> None:
    """Set source location attributes on span.

    Git repo/ref are now stamped on every recording span by the
    TracerootSpanProcessor (constructor-injected), so this function only
    handles the per-call source file/line/function context.
    """
    source = capture_source_location()
    if source.get("git_source_file"):
        span.set_attribute(SpanAttributes.GIT_SOURCE_FILE, source["git_source_file"])
    if source.get("git_source_line"):
        span.set_attribute(SpanAttributes.GIT_SOURCE_LINE, source["git_source_line"])
    if source.get("git_source_function"):
        span.set_attribute(SpanAttributes.GIT_SOURCE_FUNCTION, source["git_source_function"])


def _set_span_attributes(
    span: trace.Span,
    span_kind: SpanKind,
    metadata: dict[str, Any] | None,
    tags: list[str] | None,
    args: tuple,
    kwargs: dict,
    func: Callable,
    capture_input: bool,
    session_id: str | None = None,
    user_id: str | None = None,
) -> None:
    """Set attributes on an OpenTelemetry span."""
    # Set span kind
    span.set_attribute(SpanAttributes.SPAN_TYPE, span_kind)

    # Set session/user ID if provided directly on the decorator
    if session_id:
        span.set_attribute(SpanAttributes.TRACE_SESSION_ID, session_id)
    if user_id:
        span.set_attribute(SpanAttributes.TRACE_USER_ID, user_id)

    # Set attributes from OpenInference context (session_id, user_id, etc.)
    try:
        for key, value in get_attributes_from_context():
            span.set_attribute(key, value)
    except Exception as e:
        logger.debug(f"Failed to get context attributes: {e}")

    # Set input if capturing
    if capture_input:
        try:
            input_data = _capture_args(args, kwargs, func)
            set_span_attribute(span, SpanAttributes.SPAN_INPUT, input_data)
        except Exception as e:
            logger.debug(f"Failed to capture input: {e}")

    # Set metadata
    if metadata:
        set_span_attribute(span, SpanAttributes.SPAN_METADATA, metadata)

    # Set tags
    if tags:
        span.set_attribute(SpanAttributes.SPAN_TAGS, tags)


def _set_output(span: trace.Span, result: Any) -> None:
    """Set output attribute on span."""
    try:
        output_data = serialize_value(result)
        set_span_attribute(span, SpanAttributes.SPAN_OUTPUT, output_data)
    except Exception as e:
        logger.debug(f"Failed to capture output: {e}")


def _capture_args(args: tuple, kwargs: dict, func: Callable) -> dict[str, Any]:
    """Capture function arguments as a dictionary."""
    sig = inspect.signature(func)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    # Filter out 'self' and 'cls' to avoid capturing instance/class references
    return {k: serialize_value(v) for k, v in bound.arguments.items() if k not in ("self", "cls")}
