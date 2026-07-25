"""Local evaluation engine (OE-3).

Runs a task over every case, scores each result, isolates failures, aggregates.
Sync and async tasks and scorers are unified; concurrency is bounded; results
are returned in dataset input order. Each case emits a trace-native span tree
(evaluation-item -> task -> scorer, OE-4). Remote transport is wired in OE-5.
See design spec sections 2.4, 3, 4, 5, 8.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from traceroot.constants import SDK_VERSION, TRACEROOT_TRACER_NAME, SpanKind
from traceroot.eval.results import EvalItemResult, EvalRunResult, aggregate_scores
from traceroot.eval.transport import EvalTransport, LocalTransport, RunHandle
from traceroot.eval.types import Dataset, EvalCase, Score, ScorerContext
from traceroot.span_attributes import SpanAttributes
from traceroot.utils import serialize_value, set_span_attribute

_INLINE_DATASET = "<inline>"

_CASE_FIELDS = {f.name for f in dataclasses.fields(EvalCase)}


def _fmt_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _coerce_case(item: Any) -> EvalCase:
    """Coerce a list item into an EvalCase. Dicts must contain 'input' and only
    known EvalCase fields; anything else is a configuration error."""
    if isinstance(item, EvalCase):
        return item
    if isinstance(item, dict):
        if "input" not in item:
            raise ValueError("dataset dict item is missing required 'input' key")
        unknown = set(item) - _CASE_FIELDS
        if unknown:
            raise ValueError(f"dataset dict item has unknown key(s): {sorted(unknown)}")
        return EvalCase(**item)
    raise TypeError(f"cannot coerce {type(item).__name__} into an EvalCase")


def _normalize_data(data: Dataset | Sequence[EvalCase | dict]) -> list[EvalCase]:
    raw = list(data) if isinstance(data, Dataset) else [_coerce_case(it) for it in data]
    cases: list[EvalCase] = []
    for i, case in enumerate(raw):
        if case.id is None:
            case = dataclasses.replace(case, id=f"case-{i}")
        cases.append(case)
    return cases


def _normalize_score_like(raw: Any, default_name: str) -> list[Score]:
    """Normalize a scorer's return value into a list of Score."""
    if raw is None:
        return []
    if isinstance(raw, Score):
        return [raw]
    if isinstance(raw, (list, tuple)):
        for s in raw:
            if not isinstance(s, Score):
                raise TypeError(
                    f"scorer returned a sequence containing a non-Score: {type(s).__name__}"
                )
        return list(raw)
    if isinstance(raw, dict):
        if "name" not in raw or "value" not in raw:
            raise ValueError("scorer dict result must contain 'name' and 'value'")
        unknown = set(raw) - {"name", "value", "comment", "metadata"}
        if unknown:
            raise ValueError(f"scorer dict result has unknown key(s): {sorted(unknown)}")
        return [Score(**raw)]
    # bool is a subclass of int; str is categorical - all valid scalars.
    if isinstance(raw, (bool, int, float, str)):
        return [Score(name=default_name, value=raw)]
    raise TypeError(f"scorer returned an unsupported type: {type(raw).__name__}")


async def _await_or_run(fn: Callable[[Any], Any], arg: Any) -> Any:
    """Call fn(arg) whether it is sync or async.

    Coroutine functions are awaited directly. Sync callables run in a worker
    thread via asyncio.to_thread, which propagates the current contextvars
    Context - so the OTel span active at the call site (added in OE-4) parents
    any spans the sync callable creates.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(arg)
    return await asyncio.to_thread(fn, arg)


def _set_root_attrs(span, run_name: str, dataset_name: str, case: EvalCase) -> None:
    span.set_attribute(SpanAttributes.SPAN_TYPE, SpanKind.EVALUATION)
    span.set_attribute(SpanAttributes.EVAL_RUN_NAME, run_name)
    span.set_attribute(SpanAttributes.EVAL_DATASET_NAME, dataset_name)
    span.set_attribute(SpanAttributes.EVAL_CASE_ID, case.id)  # type: ignore[arg-type]
    span.set_attribute(SpanAttributes.EVAL_HAS_EXPECTED, case.expected is not None)
    # set_span_attribute no-ops on None, so provenance is only stamped when present.
    set_span_attribute(span, SpanAttributes.EVAL_SOURCE_TRACE_ID, case.source_trace_id)
    set_span_attribute(span, SpanAttributes.EVAL_SOURCE_SPAN_ID, case.source_span_id)
    set_span_attribute(span, SpanAttributes.EVAL_SCORE_TARGET_SPAN_ID, case.score_target_span_id)


def _record_scorer_span(span, scores: list[Score]) -> None:
    """Stamp the produced score(s) onto the scorer span (first score for convenience)."""
    if not scores:
        return
    first = scores[0]
    span.set_attribute(SpanAttributes.EVAL_SCORE_VALUE, first.value)
    if first.comment is not None:
        span.set_attribute(SpanAttributes.EVAL_SCORE_COMMENT, first.comment)


async def _run_case(
    case: EvalCase,
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    semaphore: asyncio.Semaphore,
    run_name: str,
    dataset_name: str,
    transport: EvalTransport,
    run: RunHandle,
) -> EvalItemResult:
    tracer = trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
    async with semaphore:
        output: Any = None
        error: str | None = None
        scores: list[Score] = []
        scorer_errors: dict[str, str] = {}

        # Pre-register the item (before execution) so a future live UI can show it.
        transport.register_item(run, case)

        # Root span opened INSIDE this per-case coroutine (its own asyncio Task via
        # gather), so concurrent cases never share a current-span and never tangle.
        with tracer.start_as_current_span("evaluation-item") as root:
            _set_root_attrs(root, run_name, dataset_name, case)
            sc = root.get_span_context()
            trace_id = format(sc.trace_id, "032x") if sc.is_valid else None

            # Task span current while the user's task runs -> user @observe spans nest here.
            with tracer.start_as_current_span("task") as task_span:
                task_span.set_attribute(SpanAttributes.SPAN_TYPE, SpanKind.TASK)
                task_span.set_attribute(SpanAttributes.EVAL_RUN_NAME, run_name)
                task_span.set_attribute(
                    SpanAttributes.EVAL_TASK_NAME,
                    getattr(task, "__name__", task.__class__.__name__),
                )
                set_span_attribute(
                    task_span, SpanAttributes.SPAN_INPUT, serialize_value(case.input)
                )
                try:
                    output = await _await_or_run(task, case.input)
                    set_span_attribute(
                        task_span, SpanAttributes.SPAN_OUTPUT, serialize_value(output)
                    )
                except Exception as exc:  # per-case isolation
                    error = _fmt_error(exc)
                    task_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    task_span.record_exception(exc)
                    task_span.set_attribute(SpanAttributes.EVAL_ERROR, error)

            if error is not None:
                root.set_status(Status(StatusCode.ERROR, error))
                root.set_attribute(SpanAttributes.EVAL_ERROR, error)
            else:
                # Scorer spans opened from the root (task 'with' has exited) -> siblings of task.
                ctx = ScorerContext(
                    input=case.input, output=output, expected=case.expected, metadata=case.metadata
                )
                for scorer in scorers:
                    name = getattr(scorer, "__name__", scorer.__class__.__name__)
                    with tracer.start_as_current_span(name) as scorer_span:
                        scorer_span.set_attribute(SpanAttributes.SPAN_TYPE, SpanKind.SCORER)
                        scorer_span.set_attribute(SpanAttributes.EVAL_RUN_NAME, run_name)
                        scorer_span.set_attribute(SpanAttributes.EVAL_SCORER_NAME, name)
                        try:
                            raw = await _await_or_run(scorer, ctx)
                            produced = _normalize_score_like(raw, name)
                            scores.extend(produced)
                            _record_scorer_span(scorer_span, produced)
                        except Exception as exc:  # per-scorer isolation
                            scorer_errors[name] = _fmt_error(exc)
                            scorer_span.set_status(Status(StatusCode.ERROR, str(exc)))
                            scorer_span.record_exception(exc)
                            scorer_span.set_attribute(
                                SpanAttributes.EVAL_ERROR, scorer_errors[name]
                            )

        item_result = EvalItemResult(
            case_id=case.id,  # type: ignore[arg-type]  (id assigned in _normalize_data)
            input=case.input,
            output=output,
            expected=case.expected,
            scores=scores,
            scorer_errors=scorer_errors,
            error=error,
            trace_id=trace_id,
        )
        transport.record_item_result(run, item_result)
        transport.record_scores(run, item_result.case_id, scores)
        return item_result


def _auto_transport(
    data: Dataset | Sequence[EvalCase | dict],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    dataset_id: str | None,
    candidate_version: str | None,
    environment: str,
) -> EvalTransport | None:
    """Build a PlatformTransport when a platform dataset id is available, else None.

    The dataset id comes from the explicit ``dataset_id`` argument or from a
    Dataset produced by ``pull_dataset`` (which stamps ``dataset_id`` /
    ``dataset_version_id``).
    """
    effective_id = dataset_id
    version_id: str | None = None
    if isinstance(data, Dataset):
        effective_id = effective_id or data.dataset_id
        version_id = data.dataset_version_id
    if effective_id is None:
        return None
    # Lazy import so the engine stays light and avoids any import cycle.
    from traceroot.eval.platform import PlatformTransport

    names = [getattr(s, "__name__", s.__class__.__name__) for s in scorers]
    return PlatformTransport(
        effective_id,
        scorer_names=names,
        candidate_version=candidate_version,
        environment=environment,
        dataset_version_id=version_id,
    )


def _validate_config(name, task, scorers, max_concurrency) -> None:
    if not name or not str(name).strip():
        raise ValueError("evaluate() requires a non-empty 'name'")
    if not callable(task):
        raise TypeError("'task' must be callable")
    if not scorers:
        raise ValueError("evaluate() requires at least one scorer")
    for s in scorers:
        if not callable(s):
            raise TypeError(f"scorer {s!r} is not callable")
    if max_concurrency < 1:
        raise ValueError("'max_concurrency' must be >= 1")


async def evaluate_async(
    *,
    name: str,
    data: Dataset | Sequence[EvalCase | dict],
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    environment: str = "evaluation",
) -> EvalRunResult:
    """Run an evaluation. The async engine; ``evaluate`` wraps this for sync callers.

    When ``dataset_id`` is given (or ``data`` is a Dataset from ``pull_dataset``)
    and no explicit ``transport`` is passed, results are reported to the TraceRoot
    backend via ``PlatformTransport`` (``upload.status == "uploaded"``). Otherwise
    the run is local-only.
    """
    _validate_config(name, task, scorers, max_concurrency)
    cases = _normalize_data(data)
    if not cases:
        raise ValueError("evaluate() requires non-empty 'data'")

    dataset_name = data.name if isinstance(data, Dataset) else _INLINE_DATASET
    if transport is not None:
        active_transport: EvalTransport = transport
    else:
        active_transport = (
            _auto_transport(data, scorers, dataset_id, candidate_version, environment)
            or LocalTransport()
        )
    run = active_transport.create_run(name=name, dataset_name=dataset_name, metadata=None)

    semaphore = asyncio.Semaphore(max_concurrency)
    item_results = await asyncio.gather(
        *[
            _run_case(c, task, scorers, semaphore, name, dataset_name, active_transport, run)
            for c in cases
        ]
    )

    upload = active_transport.finish_run(run)
    return EvalRunResult(
        name=name,
        item_results=list(item_results),
        score_summary=aggregate_scores(list(item_results)),
        upload=upload,
    )


def evaluate(
    *,
    name: str,
    data: Dataset | Sequence[EvalCase | dict],
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    environment: str = "evaluation",
) -> EvalRunResult:
    """Synchronous entry point. Always returns a completed EvalRunResult.

    Prefer ``evaluate_async`` inside an existing event loop: the fresh-loop path
    below runs in a worker thread and can break tasks/scorers holding resources
    bound to the outer loop.
    """
    kwargs = dict(
        name=name,
        data=data,
        task=task,
        scorers=scorers,
        max_concurrency=max_concurrency,
        transport=transport,
        dataset_id=dataset_id,
        candidate_version=candidate_version,
        environment=environment,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(evaluate_async(**kwargs))
    # A loop is already running in this thread: run to completion in a worker thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(evaluate_async(**kwargs))).result()
