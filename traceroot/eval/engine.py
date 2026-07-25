"""Core evaluation runner.

Runs a task over every case, scores each result, isolates failures, aggregates,
and emits a trace-native span tree (evaluation-item -> task -> scorer). Sync and
async tasks/scorers are unified; concurrency is bounded; results are returned in
input order; the executed cases are snapshotted for an immutable run record.

This module is the internal engine (``_run`` / ``_run_async``); the public API is
``Evaluation`` and ``evaluate`` / ``evaluate_async`` in ``evaluation.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from traceroot.constants import SDK_VERSION, TRACEROOT_TRACER_NAME, SpanKind
from traceroot.eval.ids import new_run_id
from traceroot.eval.results import (
    EvalItemResult,
    EvalRunResult,
    RunDatasetRef,
    RunView,
    aggregate_scores,
)
from traceroot.eval.session import RunSession
from traceroot.eval.transport import EvalTransport, LocalTransport
from traceroot.eval.types import (
    Dataset,
    DatasetSnapshot,
    DeferredScore,
    EvalCase,
    Score,
    ScorerContext,
    _content_revision,
)
from traceroot.span_attributes import SpanAttributes
from traceroot.utils import serialize_value, set_span_attribute

_INLINE_DATASET = "<inline>"

_CASE_FIELDS = {f.name for f in dataclasses.fields(EvalCase)}

# A private, non-exporting provider for LOCAL evaluation spans. See _eval_tracer.
_LOCAL_EVAL_PROVIDER: Any = None


def _eval_tracer(reporting: bool):
    """Return the tracer for evaluation spans, honoring the trace-privacy boundary.

    Reported run -> the global production tracer, so per-case eval spans export and
    can be linked to reported results.

    Local run -> a private provider with an ALWAYS_OFF sampler. Its spans have a
    valid context (so nested application/LLM/@observe spans still parent to the eval
    tree) but are NOT sampled, so neither the eval spans nor any nested spans created
    via the global production tracer are exported -- the default ParentBased sampler
    drops the children of an unsampled parent. The global production TracerProvider is
    never modified, so normal tracing OUTSIDE the evaluation keeps exporting. A local
    run therefore honestly has no platform trace id (see _run_case)."""
    if reporting:
        return trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)
    global _LOCAL_EVAL_PROVIDER
    if _LOCAL_EVAL_PROVIDER is None:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import ALWAYS_OFF

        _LOCAL_EVAL_PROVIDER = TracerProvider(sampler=ALWAYS_OFF)
    return _LOCAL_EVAL_PROVIDER.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)


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
    if isinstance(raw, DeferredScore):
        # A deferred/human score is pending, never a numeric zero.
        return [
            Score(name=raw.name, value="pending", comment=raw.reason, metadata={"deferred": True})
        ]
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


@dataclasses.dataclass(frozen=True)
class _RunIdentity:
    """Immutable identity stamped on every evaluation trace (see the attribute
    contract note). Built once per run; the same values mark each per-case root."""

    name: str
    dataset_name: str
    dataset_id: str
    dataset_version_id: str | None
    candidate_version: str | None
    environment: str
    run_id: str | None
    local_run_id: str


# Version of the evaluation trace attribute contract emitted on the root span.
_EVAL_ATTR_CONTRACT_VERSION = "1"


def _set_root_attrs(span, identity: _RunIdentity, case: EvalCase) -> None:
    attr = SpanAttributes
    span.set_attribute(attr.SPAN_TYPE, SpanKind.EVALUATION)  # eval-trace marker
    span.set_attribute(attr.EVAL_CONTRACT_VERSION, _EVAL_ATTR_CONTRACT_VERSION)
    span.set_attribute(attr.ENVIRONMENT, identity.environment)
    span.set_attribute(attr.EVAL_ENVIRONMENT, identity.environment)
    span.set_attribute(attr.EVAL_NAME, identity.name)
    span.set_attribute(attr.EVAL_RUN_NAME, identity.name)  # retained alias
    span.set_attribute(attr.EVAL_DATASET_NAME, identity.dataset_name)
    span.set_attribute(attr.EVAL_CASE_ID, case.id)  # type: ignore[arg-type]
    span.set_attribute(attr.EVAL_HAS_EXPECTED, case.expected is not None)
    # set_span_attribute no-ops on None, so optional identity/provenance is only
    # stamped when present (dataset_version_id/candidate_version/run_id, source ids).
    set_span_attribute(span, attr.EVAL_DATASET_ID, identity.dataset_id)
    set_span_attribute(span, attr.EVAL_DATASET_VERSION_ID, identity.dataset_version_id)
    set_span_attribute(span, attr.EVAL_CANDIDATE_VERSION, identity.candidate_version)
    set_span_attribute(span, attr.EVAL_RUN_ID, identity.run_id)
    set_span_attribute(span, attr.EVAL_LOCAL_RUN_ID, identity.local_run_id)
    set_span_attribute(span, attr.EVAL_SOURCE_TRACE_ID, case.source_trace_id)
    set_span_attribute(span, attr.EVAL_SOURCE_SPAN_ID, case.source_span_id)
    set_span_attribute(span, attr.EVAL_SCORE_TARGET_SPAN_ID, case.score_target_span_id)


def _stamp_scorer_version(scores: list[Score], scorer: Any) -> list[Score]:
    """Apply a scorer's EXPLICITLY declared version (a ``version`` attribute on the
    scorer) to produced scores that did not set their own. Absent a declaration the
    version stays None -- V1 never invents a version. A score that already carries an
    explicit version (e.g. returned as ``Score(..., version="2")``) is left untouched.
    """
    declared = getattr(scorer, "version", None)
    if declared is None:
        return scores
    return [
        s if s.version is not None else dataclasses.replace(s, version=declared) for s in scores
    ]


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
    identity: _RunIdentity,
    session: RunSession,
    reporting: bool = False,
    timeout: float | None = None,
    on_case_start: Callable[[EvalCase], None] | None = None,
    on_case_complete: Callable[[EvalItemResult, float], None] | None = None,
) -> EvalItemResult:
    tracer = _eval_tracer(reporting)
    async with semaphore:
        started = time.perf_counter()
        output: Any = None
        error: str | None = None
        scores: list[Score] = []
        scorer_errors: dict[str, str] = {}

        if on_case_start is not None:
            on_case_start(case)
        # Pre-register the item (before execution) so a future live UI can show it.
        session.register(case)

        # Root span opened INSIDE this per-case coroutine (its own asyncio Task via
        # gather), so concurrent cases never share a current-span and never tangle.
        with tracer.start_as_current_span("evaluation-item") as root:
            _set_root_attrs(root, identity, case)
            sc = root.get_span_context()
            # Local runs export nothing, so their results honestly carry no platform
            # trace id (the ALWAYS_OFF span has a valid context but is never exported).
            trace_id = format(sc.trace_id, "032x") if (reporting and sc.is_valid) else None

            # Task span current while the user's task runs -> user @observe spans nest here.
            with tracer.start_as_current_span("task") as task_span:
                task_span.set_attribute(SpanAttributes.SPAN_TYPE, SpanKind.TASK)
                task_span.set_attribute(SpanAttributes.EVAL_RUN_NAME, identity.name)
                task_span.set_attribute(SpanAttributes.EVAL_CASE_ID, case.id)  # type: ignore[arg-type]
                task_span.set_attribute(
                    SpanAttributes.EVAL_TASK_NAME,
                    getattr(task, "__name__", task.__class__.__name__),
                )
                set_span_attribute(
                    task_span, SpanAttributes.SPAN_INPUT, serialize_value(case.input)
                )
                try:
                    if timeout is not None:
                        output = await asyncio.wait_for(_await_or_run(task, case.input), timeout)
                    else:
                        output = await _await_or_run(task, case.input)
                    set_span_attribute(
                        task_span, SpanAttributes.SPAN_OUTPUT, serialize_value(output)
                    )
                except Exception as exc:  # per-case isolation (incl. asyncio.TimeoutError)
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
                        scorer_span.set_attribute(SpanAttributes.EVAL_RUN_NAME, identity.name)
                        scorer_span.set_attribute(SpanAttributes.EVAL_SCORER_NAME, name)
                        try:
                            raw = await _await_or_run(scorer, ctx)
                            produced = _stamp_scorer_version(
                                _normalize_score_like(raw, name), scorer
                            )
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
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        session.record(item_result)
        if on_case_complete is not None:
            on_case_complete(item_result, item_result.duration_ms)
        return item_result


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


def _to_snapshot(
    data: Dataset | DatasetSnapshot | Sequence[EvalCase | dict],
    cases: list[EvalCase],
) -> DatasetSnapshot:
    """The immutable description of exactly which cases ran (post-select)."""
    if isinstance(data, DatasetSnapshot) and len(data.cases) == len(cases):
        return data
    if isinstance(data, Dataset):
        name, description, dataset_id = data.name, data.description, data.dataset_id
        base_version_id = data.dataset_version_id
    elif isinstance(data, DatasetSnapshot):
        name, description, dataset_id = data.name, data.description, data.dataset_id
        base_version_id = data.base_version_id
    else:
        name, description, dataset_id, base_version_id = _INLINE_DATASET, None, "ds_inline", None
    cases_t = tuple(cases)
    return DatasetSnapshot(
        dataset_id=dataset_id,
        name=name,
        description=description,
        revision=_content_revision(cases_t),
        cases=cases_t,
        base_version_id=base_version_id,
    )


async def _run_async(
    *,
    name: str,
    data: Dataset | DatasetSnapshot | Sequence[EvalCase | dict],
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    environment: str = "evaluation",
    select: Callable[[EvalCase], bool] | None = None,
    run_scorers: Sequence[Callable[[RunView], Any]] | None = None,
    timeout: float | None = None,
    metadata: dict[str, Any] | None = None,
    baseline: EvalRunResult | None = None,
    on_case_start: Callable[[EvalCase], None] | None = None,
    on_case_complete: Callable[[EvalItemResult, float], None] | None = None,
) -> EvalRunResult:
    """Core async runner. Public entry is ``evaluate``/``Evaluation`` (evaluation.py).

    ``timeout`` bounds each task (seconds; a timeout is an isolated per-case error).
    ``on_case_start``/``on_case_complete`` are internal hooks the CLI runner uses to
    stream per-case events; ``on_case_complete(item, duration_ms)``.
    """
    _validate_config(name, task, scorers, max_concurrency)
    cases = _normalize_data(data)
    if select is not None:
        cases = [c for c in cases if select(c)]
    if not cases:
        raise ValueError("evaluate() requires at least one case to run")

    snapshot = _to_snapshot(data, cases)
    dataset_ref = RunDatasetRef(
        # dataset_id is identity only: an explicit dataset_id= associates this run
        # with a platform dataset without authorizing any upload.
        dataset_id=dataset_id or snapshot.dataset_id,
        revision=snapshot.revision,
        dataset_version_id=data.dataset_version_id
        if isinstance(data, Dataset)
        else snapshot.base_version_id,
        case_count=len(cases),
    )

    dataset_name = snapshot.name
    # Publication boundary (product direction 10.5): local unless the caller EXPLICITLY
    # opts in via report_to/transport. A pulled/remote-pinned dataset, a bare dataset_id,
    # and ambient credentials are NOT consent -- there is no implicit platform transport.
    active_transport: EvalTransport = transport if transport is not None else LocalTransport()

    # The high-level runner drives the SAME low-level RunSession that custom
    # harnesses use -- one code path.
    session = RunSession(
        active_transport,
        name=name,
        dataset_name=dataset_name,
        dataset_ref=dataset_ref,
        candidate_version=candidate_version,
        environment=environment,
        metadata=metadata,
    ).start()

    # Trace-privacy boundary: only a reported run exports its per-case eval traces.
    reporting = getattr(active_transport, "reports_traces", False)

    # Client-side run id generated up front so it can be stamped on the per-case trace
    # identity AND carried on the result (the platform run_id is separate, when reported).
    local_run_id = new_run_id()

    # One immutable identity stamped on every per-case evaluation trace (attribute
    # contract). run_id is available now (post create_run) when reported.
    identity = _RunIdentity(
        name=name,
        dataset_name=dataset_name,
        dataset_id=dataset_ref.dataset_id,
        dataset_version_id=dataset_ref.dataset_version_id,
        candidate_version=candidate_version,
        environment=environment,
        run_id=getattr(active_transport, "run_id", None),
        local_run_id=local_run_id,
    )

    semaphore = asyncio.Semaphore(max_concurrency)
    item_results = await asyncio.gather(
        *[
            _run_case(
                c,
                task,
                scorers,
                semaphore,
                identity,
                session,
                reporting=reporting,
                timeout=timeout,
                on_case_start=on_case_start,
                on_case_complete=on_case_complete,
            )
            for c in cases
        ]
    )

    upload = session.complete()

    results_list = list(item_results)
    summary = aggregate_scores(results_list)
    run_scores, run_scorer_errors = await _run_run_scorers(run_scorers, name, results_list, summary)

    return EvalRunResult(
        name=name,
        item_results=results_list,
        score_summary=summary,
        upload_state=upload,
        local_run_id=local_run_id,
        candidate_version=candidate_version,
        dataset=dataset_ref,
        run_id=getattr(active_transport, "run_id", None),
        run_scores=run_scores,
        run_scorer_errors=run_scorer_errors,
        metadata=metadata,
        baseline=baseline,
    )


async def _run_run_scorers(
    run_scorers: Sequence[Callable[[RunView], Any]] | None,
    name: str,
    item_results: list[EvalItemResult],
    summary: Any,
) -> tuple[list[Score], dict[str, str]]:
    """Run whole-run scorers over the completed items. Errors are isolated per scorer."""
    run_scores: list[Score] = []
    run_scorer_errors: dict[str, str] = {}
    if not run_scorers:
        return run_scores, run_scorer_errors
    view = RunView(name=name, item_results=item_results, score_summary=summary)
    for rs in run_scorers:
        rname = getattr(rs, "__name__", rs.__class__.__name__)
        try:
            raw = rs(view)
            if inspect.iscoroutine(raw):
                raw = await raw
            run_scores.extend(_stamp_scorer_version(_normalize_score_like(raw, rname), rs))
        except Exception as exc:  # per-run-scorer isolation
            run_scorer_errors[rname] = _fmt_error(exc)
    return run_scores, run_scorer_errors


def _run(**kwargs: Any) -> EvalRunResult:
    """Synchronous core runner. Always returns a completed EvalRunResult."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run_async(**kwargs))
    # A loop is already running in this thread: run to completion in a worker thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_run_async(**kwargs))).result()
