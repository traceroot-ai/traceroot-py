"""Source location recorded by ``@observe`` on async functions.

When a coroutine is started by the event loop (``asyncio.run``,
``create_task``, ``gather``) the frame directly above the SDK used to be
``asyncio/events.py``, so the span pointed at Python's own installation
instead of the user's file. Standard-library frames are skipped now, and a
decorated function's definition is used when no user frame is left.
"""

import asyncio

from traceroot import observe
from traceroot.span_attributes import SpanAttributes


def _source(span):
    return (
        span.attributes.get(SpanAttributes.GIT_SOURCE_FILE),
        span.attributes.get(SpanAttributes.GIT_SOURCE_LINE),
        span.attributes.get(SpanAttributes.GIT_SOURCE_FUNCTION),
    )


def _assert_user_location(span):
    file, line, func = _source(span)
    assert file is not None
    assert "asyncio" not in file
    assert file.endswith("test_source_location_async.py")
    assert line is not None
    assert func not in ("_run", "run", "__step")


def test_asyncio_run_records_user_location(memory_exporter):
    @observe(type="tool")
    async def tool():
        return 1

    @observe(type="agent")
    async def agent():
        return await tool()

    assert asyncio.run(agent()) == 1
    spans = {s.name: s for s in memory_exporter.get_finished_spans()}
    _assert_user_location(spans["tool"])


def test_create_task_records_user_location(memory_exporter):
    @observe(type="tool")
    async def tool():
        return 1

    @observe(type="agent")
    async def agent():
        return await asyncio.create_task(tool())

    assert asyncio.run(agent()) == 1
    spans = {s.name: s for s in memory_exporter.get_finished_spans()}
    _assert_user_location(spans["tool"])


def test_gather_records_user_location(memory_exporter):
    @observe(type="tool")
    async def tool():
        return 1

    @observe(type="agent")
    async def agent():
        return (await asyncio.gather(tool()))[0]

    assert asyncio.run(agent()) == 1
    spans = {s.name: s for s in memory_exporter.get_finished_spans()}
    _assert_user_location(spans["tool"])


def test_direct_await_still_records_caller(memory_exporter):
    @observe(type="tool")
    async def tool():
        return 1

    @observe(type="agent")
    async def agent():
        return await tool()

    assert asyncio.run(agent()) == 1
    spans = {s.name: s for s in memory_exporter.get_finished_spans()}
    file, _line, func = _source(spans["tool"])
    assert file.endswith("test_source_location_async.py")
    assert func == "agent"


def test_falls_back_to_function_definition(monkeypatch, memory_exporter):
    import traceroot.decorators as decorators

    # No user frame reachable (the coroutine-resumed-without-caller case).
    monkeypatch.setattr(decorators, "capture_source_location", lambda fallback=None: {})

    @observe(type="tool")
    def tool():
        return 1

    assert tool() == 1
    span = memory_exporter.get_finished_spans()[0]
    file, line, func = _source(span)
    assert file.endswith("test_source_location_async.py")
    assert func == "tool"
    assert line == tool.__wrapped__.__code__.co_firstlineno
