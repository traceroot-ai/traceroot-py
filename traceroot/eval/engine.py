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
import contextvars
import dataclasses
import inspect
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
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
from traceroot.eval.transport import EvalTransport, RunHandle
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


def _eval_tracer():
    """The global production tracer for evaluation spans. Evaluation is cloud-only, so a
    run always exports its per-case spans, which are then linked to the reported results."""
    return trace.get_tracer(TRACEROOT_TRACER_NAME, SDK_VERSION)


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


async def _await_or_run(
    fn: Callable[[Any], Any], arg: Any, executor: ThreadPoolExecutor | None = None
) -> Any:
    """Call fn(arg) whether it is sync or async.

    Coroutine functions — and callable *instances* whose ``__call__`` is async — are awaited on
    the loop. Plain sync callables run in ``executor`` (a bounded per-run thread pool) so that a
    task orphaned by a timeout occupies a bounded slot instead of an unbounded ``to_thread``
    worker, keeping true concurrency within ``max_concurrency``. The current contextvars Context
    is copied into the worker so the OTel span active at the call site still parents any spans the
    sync callable creates.
    """
    if inspect.iscoroutinefunction(fn):
        return await fn(arg)
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    result = await loop.run_in_executor(executor, lambda: ctx.run(fn, arg))
    # A callable *instance* whose __call__ is async is NOT a coroutine function, so it ran in the
    # thread and returned an un-awaited coroutine; await it on the loop so the coroutine is never
    # surfaced as the task/scorer output.
    if inspect.isawaitable(result):
        result = await result
    return result


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
    from traceroot.eval.scorers import declared_version

    declared = declared_version(scorer)
    if declared is None:
        return scores
    return [
        s if s.version is not None else dataclasses.replace(s, version=declared) for s in scores
    ]


def _scorer_output_repr(scores: list[Score]) -> list[dict[str, Any]]:
    """The scorer span's span.output: each produced score's value + explanation."""
    return [{"name": s.name, "value": s.value, "explanation": s.comment} for s in scores]


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
    transport: EvalTransport,
    run: RunHandle,
    timeout: float | None = None,
    on_case_start: Callable[[EvalCase], None] | None = None,
    on_case_complete: Callable[[EvalItemResult, float], None] | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> EvalItemResult:
    tracer = _eval_tracer()
    async with semaphore:
        started = time.perf_counter()
        output: Any = None
        error: str | None = None
        scores: list[Score] = []
        scorer_errors: dict[str, str] = {}

        if on_case_start is not None:
            on_case_start(case)
        # Pre-register the item (before execution) so a future live UI can show it.
        try:
            transport.register_item(run, case)
        except Exception:
            pass  # reporting is best-effort; a transport blip must not drop the case

        # Root span opened INSIDE this per-case coroutine (its own asyncio Task via
        # gather), so concurrent cases never share a current-span and never tangle.
        with tracer.start_as_current_span("evaluation-item", context=Context()) as root:
            _set_root_attrs(root, identity, case)
            # Standard span I/O on the root so the trace viewer renders trace-level and
            # root-span Input/Output (output is set below, once the candidate output/error
            # is known). See offline-eval/contract-notes/eval-span-io.md.
            set_span_attribute(root, SpanAttributes.SPAN_INPUT, serialize_value(case.input))
            sc = root.get_span_context()
            # A reported run exports its per-case spans, so the result carries the platform
            # trace id for the item->trace link.
            trace_id = format(sc.trace_id, "032x") if sc.is_valid else None

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
                        output = await asyncio.wait_for(
                            _await_or_run(task, case.input, executor), timeout
                        )
                    else:
                        output = await _await_or_run(task, case.input, executor)
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
                set_span_attribute(root, SpanAttributes.SPAN_OUTPUT, error)  # task error as output
            else:
                # Root output = the candidate output (mirrors the task span's output).
                set_span_attribute(root, SpanAttributes.SPAN_OUTPUT, serialize_value(output))
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
                        # span.input = what the scorer compared (candidate + expected,
                        # plus the scored span target when present).
                        scorer_input: dict[str, Any] = {
                            "candidate": output,
                            "expected": case.expected,
                        }
                        if case.score_target_span_id is not None:
                            scorer_input["target_span_id"] = case.score_target_span_id
                        set_span_attribute(
                            scorer_span, SpanAttributes.SPAN_INPUT, serialize_value(scorer_input)
                        )
                        try:
                            raw = await _await_or_run(scorer, ctx, executor)
                            produced = _stamp_scorer_version(
                                _normalize_score_like(raw, name), scorer
                            )
                            scores.extend(produced)
                            _record_scorer_span(scorer_span, produced)
                            if produced:  # span.output = the score value(s) + explanation
                                set_span_attribute(
                                    scorer_span,
                                    SpanAttributes.SPAN_OUTPUT,
                                    serialize_value(_scorer_output_repr(produced)),
                                )
                        except Exception as exc:  # per-scorer isolation
                            scorer_errors[name] = _fmt_error(exc)
                            scorer_span.set_status(Status(StatusCode.ERROR, str(exc)))
                            scorer_span.record_exception(exc)
                            scorer_span.set_attribute(
                                SpanAttributes.EVAL_ERROR, scorer_errors[name]
                            )
                            set_span_attribute(
                                scorer_span, SpanAttributes.SPAN_OUTPUT, scorer_errors[name]
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
        try:
            transport.record_item_result(run, item_result)
            transport.record_scores(run, item_result.case_id, item_result.scores)
        except Exception:
            pass  # reporting is best-effort; the computed result is still returned
        if on_case_complete is not None:
            on_case_complete(item_result, item_result.duration_ms)
        return item_result


def _auto_transport(
    data: Dataset | DatasetSnapshot | Sequence[EvalCase | dict],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    dataset_id: str | None,
    candidate_version: str | None,
    environment: str,
) -> EvalTransport | None:
    """Build the reporting transport from credentials + a synced dataset. Returns None when
    it cannot (no synced dataset, or no credentials); the caller turns that into a clear error
    (evaluation is cloud-only).

    A synced dataset means an explicit ``dataset_id`` OR a Dataset that was pulled/pushed
    (``dataset_version_id`` stamped). A locally-created ``ds_`` id is not one.
    """
    effective_id = dataset_id
    version_id: str | None = None
    if isinstance(data, Dataset):
        version_id = data.dataset_version_id
        if effective_id is None and version_id is not None:
            effective_id = data.dataset_id
    elif isinstance(data, DatasetSnapshot):
        # A snapshot taken from a synced dataset carries base_version_id; treat it like a synced
        # Dataset so `evaluate(dataset.snapshot())` reports instead of failing "no synced dataset".
        version_id = data.base_version_id
        if effective_id is None and version_id is not None:
            effective_id = data.dataset_id
    if effective_id is None:
        return None  # inline/unsynced dataset -> nothing to report against

    from traceroot.eval.platform import PlatformTransport, _resolve_credentials

    key, _host = _resolve_credentials(None, None)
    if not key:  # no credentials
        return None
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
    progress: bool | None = None,
    on_case_start: Callable[[EvalCase], None] | None = None,
    on_case_complete: Callable[[EvalItemResult, float], None] | None = None,
    on_run_start: Callable[[str, str | None], None] | None = None,
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

    # Enrich run metadata with auto-discovered provenance (git/ci) for the run record.
    # Cheap path (no git-status subprocess); dirty is available via the public
    # collect_run_provenance helper. NOTE: not uploaded -- the backend's strict
    # run-registration schema has no metadata field.
    from traceroot.eval.provenance import collect_run_provenance

    run_metadata = collect_run_provenance(metadata, detect_dirty=False)

    dataset_name = snapshot.name
    # Cloud-only: an explicit transport wins; otherwise build a reporting transport from
    # credentials + the synced dataset. Evaluation always reports -- there is no local run.
    if transport is not None:
        active_transport: EvalTransport = transport
    else:
        active_transport = _auto_transport(
            data, scorers, dataset_id, candidate_version, environment
        )
        if active_transport is None:
            raise RuntimeError(
                "evaluate() reports to the TraceRoot platform, but no credentials or synced "
                "dataset were found. Set TRACEROOT_API_KEY and pass a pulled dataset "
                "(traceroot.pull_dataset(...)), or pass an explicit transport=."
            )

    # Forward scorer comparison metadata (value_type/direction/threshold) from the actual
    # scorer callables when the transport accepts specs and the caller did not pre-set them.
    # Must happen BEFORE create_run so the descriptors reach registration.
    if getattr(active_transport, "scorer_specs", "unset") is None:
        from traceroot.eval.scorers import describe_scorers

        active_transport.scorer_specs = describe_scorers(scorers)

    # Client-side run id: the idempotency key for run registration AND the id carried on
    # the result (the platform run_id is separate, assigned by the backend).
    local_run_id = new_run_id()

    # Register the run up front so scorer descriptors + the run_id are available before the
    # per-case traces are stamped.
    run_handle = active_transport.create_run(
        name=name,
        dataset_name=dataset_name,
        metadata=metadata,
        client_run_id=local_run_id,
    )
    # Surface the registered ids immediately so a caller that is interrupted mid-run (e.g. the CLI
    # runner on SIGINT) can finalize its partial artifact under the SAME ids the run registered
    # with, keeping it reconcilable with the cloud run.
    if on_run_start is not None:
        on_run_start(local_run_id, getattr(active_transport, "run_id", None))

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

    # Live console progress bar (local presentation only; auto-on for an
    # interactive terminal, off when piped/CI). It composes with any caller
    # hooks rather than replacing them.
    reporter = None
    case_start_hook = on_case_start
    case_complete_hook = on_case_complete
    from traceroot.eval.progress import should_show_progress

    if should_show_progress(progress):
        from traceroot.eval.progress import ConsoleProgress

        reporter = ConsoleProgress(len(cases), name)
        reporter.start()

        def case_complete_hook(item, duration_ms, _next=on_case_complete):
            reporter.on_case_complete(item, duration_ms)
            if _next is not None:
                _next(item, duration_ms)

    semaphore = asyncio.Semaphore(max_concurrency)
    # Bounded thread pool for sync tasks/scorers: a task orphaned by a timeout keeps running in a
    # pool slot but can't push real thread use past max_concurrency (an unbounded to_thread could).
    sync_executor = ThreadPoolExecutor(max_workers=max_concurrency)
    try:
        item_results = await asyncio.gather(
            *[
                _run_case(
                    c,
                    task,
                    scorers,
                    semaphore,
                    identity,
                    active_transport,
                    run_handle,
                    timeout=timeout,
                    on_case_start=case_start_hook,
                    on_case_complete=case_complete_hook,
                    executor=sync_executor,
                )
                for c in cases
            ]
        )
    finally:
        # Don't block shutdown on threads orphaned by a timeout; the process/pool reclaims them.
        sync_executor.shutdown(wait=False)
        if reporter is not None:
            reporter.finish()
        # Finish the run inside finally so a mid-run failure never leaves it open on the backend.
        upload = active_transport.finish_run(run_handle, status=None)

    results_list = list(item_results)
    summary = aggregate_scores(results_list)
    run_scores, run_scorer_errors = await _run_run_scorers(run_scorers, name, results_list, summary)

    result = EvalRunResult(
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
        metadata=run_metadata,
    )

    # When the bar was shown (interactive), surface the clickable run link if the
    # backend returned one. Off-terminal callers read result.upload_state instead.
    # (Candidate-vs-baseline comparison is the backend's job, not the SDK's.)
    if reporter is not None and getattr(upload, "dashboard_url", None):
        from traceroot.eval.progress import print_run_url

        print_run_url(upload.dashboard_url)

    return result


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
