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

from opentelemetry.context import Context
from opentelemetry.trace import Status, StatusCode

from traceroot.constants import SpanKind
from traceroot.eval.ids import new_run_id
from traceroot.eval.results import (
    EvalItemResult,
    EvalRunResult,
    RunDatasetRef,
    RunView,
    UploadState,
    aggregate_scores,
)
from traceroot.eval.scorers import scorer_name
from traceroot.eval.tracer import eval_tracer as _eval_tracer  # span-export tracer (own module)
from traceroot.eval.transport import EvalCompletionError, EvalTransport, RunHandle
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


class _CaseTimeoutError(Exception):
    """A per-case deadline expiry. A distinct type so the scorer loop can tell a budget overrun
    (which errors the whole case) apart from an ordinary per-scorer failure (which is isolated),
    and so a task/scorer raising ``TimeoutError`` itself is never mistaken for one."""


def _fmt_error(exc: BaseException) -> str:
    # A deadline expiry reads as a TimeoutError (same wording as the TS engine), not as the
    # internal marker class.
    if isinstance(exc, _CaseTimeoutError):
        return f"TimeoutError: {exc}"
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
        if "name" in raw or "value" in raw:
            # Single named Score: {"name", "value", comment?, metadata?}. Presence of EITHER reserved
            # key means the single-Score shape is intended, so both are required (a {"value": ...}
            # without a name stays an error, not a metric named "value").
            if "name" not in raw or "value" not in raw:
                raise ValueError("scorer dict result must contain 'name' and 'value'")
            unknown = set(raw) - {"name", "value", "comment", "metadata"}
            if unknown:
                raise ValueError(f"scorer dict result has unknown key(s): {sorted(unknown)}")
            return [Score(**raw)]
        # Neither 'name' nor 'value' -> a metric->value mapping; one Score per entry, keyed by the
        # metric name. (A metric literally named "name"/"value" must use the single-Score form.)
        scores: list[Score] = []
        for k, v in raw.items():
            if not isinstance(v, (bool, int, float, str)):
                raise TypeError(
                    f"metric-map value for {k!r} must be a scalar (bool/int/float/str), "
                    f"got {type(v).__name__}"
                )
            scores.append(Score(name=str(k), value=v))
        return scores
    # bool is a subclass of int; str is categorical - all valid scalars.
    if isinstance(raw, (bool, int, float, str)):
        return [Score(name=default_name, value=raw)]
    raise TypeError(f"scorer returned an unsupported type: {type(raw).__name__}")


# Process-wide bounded thread pool for sync tasks/scorers, reused across every run. A task
# orphaned by a timeout can't be cancelled (Python threads), so it keeps a slot until it finishes;
# sharing ONE bounded pool means those orphans occupy bounded, shared slots instead of a fresh
# per-run pool leaking threads on every evaluation.
_SYNC_EXECUTOR: ThreadPoolExecutor | None = None


def _sync_executor(max_workers: int) -> ThreadPoolExecutor:
    global _SYNC_EXECUTOR
    if _SYNC_EXECUTOR is None:
        _SYNC_EXECUTOR = ThreadPoolExecutor(
            max_workers=max(max_workers, 8), thread_name_prefix="traceroot-eval-sync"
        )
    return _SYNC_EXECUTOR


async def _await_or_run(
    fn: Callable[..., Any],
    args: tuple,
    kwargs: dict | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> Any:
    """Call fn(*args, **kwargs) whether it is sync or async.

    Coroutine functions — and callable *instances* whose ``__call__`` is async — are awaited on
    the loop. Plain sync callables run in ``executor`` (a bounded per-run thread pool) so that a
    task orphaned by a timeout occupies a bounded slot instead of an unbounded ``to_thread``
    worker, keeping true concurrency within ``max_concurrency``. The current contextvars Context
    is copied into the worker so the OTel span active at the call site still parents any spans the
    sync callable creates.
    """
    kwargs = kwargs or {}
    if inspect.iscoroutinefunction(fn):
        return await fn(*args, **kwargs)
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    result = await loop.run_in_executor(executor, lambda: ctx.run(fn, *args, **kwargs))
    # A callable *instance* whose __call__ is async is NOT a coroutine function, so it ran in the
    # thread and returned an un-awaited coroutine; await it on the loop so the coroutine is never
    # surfaced as the task/scorer output.
    if inspect.isawaitable(result):
        result = await result
    return result


async def _bounded(coro: Any, deadline: float | None, timeout: float | None) -> Any:
    """Await ``coro`` within the case's SHARED deadline (one budget for task + scorers, so a
    hung judge cannot outlive ``timeout=``). ``deadline`` is an absolute ``time.monotonic()``
    mark; expiry raises ``_CaseTimeoutError``. The awaitable is raced rather than wrapped in
    ``wait_for`` so a TimeoutError the user's own code raises stays their error."""
    if deadline is None:
        return await coro
    fut = asyncio.ensure_future(coro)
    done, _pending = await asyncio.wait({fut}, timeout=max(0.0, deadline - time.monotonic()))
    if not done:
        fut.cancel()  # an uncancellable sync call keeps its bounded pool slot until it returns
        raise _CaseTimeoutError(f"case exceeded {timeout}s")
    return fut.result()


_PLAIN_FIELDS = ("input", "output", "expected", "metadata")


def _scorer_target(scorer: Any) -> Any:
    if inspect.isfunction(scorer) or inspect.ismethod(scorer) or inspect.isbuiltin(scorer):
        return scorer
    return getattr(scorer, "__call__", scorer)


def _scorer_call_plan(scorer: Any) -> tuple[str, tuple[str, ...]]:
    """Decide how to invoke a scorer, by PARAMETER NAME (never by bare arity, so a closure default
    like ``def fn(ctx, _cfg=X)`` is not mistaken for a two-arg plain scorer):

      - a parameter named ``input`` or ``output`` -> PLAIN form; called by keyword with the declared
        subset of (input, output, expected, metadata);
      - otherwise -> the ScorerContext form ``scorer(ctx)`` (any extra parameters keep their defaults).

    Returns ``("plain", fields)`` or ``("ctx", ())``. Raises an actionable error for a signature that
    fits neither (e.g. a plain scorer with an undeclared required parameter, or a ctx scorer that
    requires more than the single context argument). Un-introspectable callables default to ctx."""
    try:
        params = list(inspect.signature(_scorer_target(scorer)).parameters.values())
    except (ValueError, TypeError):
        return ("ctx", ())
    pos = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    names = [p.name for p in pos]
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in params)
    has_varargs = any(p.kind == p.VAR_POSITIONAL for p in params)
    label = scorer_name(scorer)
    if "input" in names or "output" in names:
        unsupported = [p.name for p in pos if p.default is p.empty and p.name not in _PLAIN_FIELDS]
        if unsupported and not has_kwargs:
            raise TypeError(
                f"scorer {label!r} declares unsupported required parameter(s) {unsupported}; a plain "
                f"scorer may take only {list(_PLAIN_FIELDS)} (e.g. "
                f"def scorer(input, output, expected=None, metadata=None))."
            )
        return ("plain", tuple(nm for nm in _PLAIN_FIELDS if nm in names))
    required = [p for p in pos if p.default is p.empty]
    if len(required) > 1 and not has_varargs:
        raise TypeError(
            f"scorer {label!r} has an unsupported signature: {len(required)} required positional "
            f"parameters. Use `def scorer(ctx)` or "
            f"`def scorer(input, output, expected=None, metadata=None)`."
        )
    if not pos and not has_varargs:
        raise TypeError(
            f"scorer {label!r} takes no arguments; a scorer must accept a ScorerContext or "
            f"(input, output, ...)."
        )
    return ("ctx", ())


def _bind_scorer_args(scorer: Any, ctx: "ScorerContext") -> tuple[tuple, dict]:
    """(args, kwargs) to invoke a scorer for one case, per its call plan (see _scorer_call_plan)."""
    kind, fields = _scorer_call_plan(scorer)
    if kind == "plain":
        by = {"input": ctx.input, "output": ctx.output, "expected": ctx.expected,
              "metadata": ctx.metadata}
        return ((), {f: by[f] for f in fields})
    return ((ctx,), {})


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
    emitted_ownership: dict[str, set[str]] | None = None,
    dropped_results: list[str] | None = None,
) -> EvalItemResult:
    tracer = _eval_tracer()
    async with semaphore:
        started = time.perf_counter()
        # ONE deadline for the whole case (task + scorers), so a hung scorer cannot outlive it.
        deadline = time.monotonic() + timeout if timeout is not None else None
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
                    output = await _bounded(
                        _await_or_run(task, (case.input,), executor=executor), deadline, timeout
                    )
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
                case_timeout: _CaseTimeoutError | None = None
                for scorer in scorers:
                    # The DECLARED name (same resolver the manifest uses) -- the emitted Score,
                    # the ownership key and the registered definition must agree or the platform
                    # drops the metric's policy.
                    name = scorer_name(scorer)
                    # Record scorer->metric ownership AT THE POINT OF PRODUCTION (never inferred
                    # afterward by matching names). Seed the definition now so a scorer that errors
                    # before emitting still appears in the completion manifest with no metrics.
                    if emitted_ownership is not None:
                        emitted_ownership.setdefault(name, set())
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
                            s_args, s_kwargs = _bind_scorer_args(scorer, ctx)
                            raw = await _bounded(
                                _await_or_run(scorer, s_args, s_kwargs, executor), deadline, timeout
                            )
                            produced = _stamp_scorer_version(
                                _normalize_score_like(raw, name), scorer
                            )
                            scores.extend(produced)
                            if emitted_ownership is not None:
                                emitted_ownership[name].update(
                                    s.name for s in produced if s.name
                                )
                            _record_scorer_span(scorer_span, produced)
                            if produced:  # span.output = the score value(s) + explanation
                                set_span_attribute(
                                    scorer_span,
                                    SpanAttributes.SPAN_OUTPUT,
                                    serialize_value(_scorer_output_repr(produced)),
                                )
                        except _CaseTimeoutError as exc:
                            # A deadline hit is a case-level budget overrun, not an isolated
                            # scorer failure: record it on the span, stop scoring, and error the
                            # whole case below.
                            case_timeout = exc
                            scorer_span.set_status(Status(StatusCode.ERROR, str(exc)))
                            scorer_span.set_attribute(
                                SpanAttributes.EVAL_ERROR, _fmt_error(exc)
                            )
                            set_span_attribute(
                                scorer_span, SpanAttributes.SPAN_OUTPUT, _fmt_error(exc)
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
                    if case_timeout is not None:
                        break  # (outside the span 'with' so the scorer span still closes)

                if case_timeout is not None:
                    # Scoring exceeded the case deadline -> the same isolated per-case error path
                    # as a task timeout (errored case, not scored). Scores collected before the
                    # timeout are dropped so a timed-out case is truly "not scored" and cannot
                    # contaminate the summaries.
                    error = _fmt_error(case_timeout)
                    scores.clear()
                    root.set_status(Status(StatusCode.ERROR, error))
                    root.set_attribute(SpanAttributes.EVAL_ERROR, error)
                    set_span_attribute(root, SpanAttributes.SPAN_OUTPUT, error)

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
        except Exception as exc:
            # Reporting is best-effort; the computed result is still returned. But a dropped
            # result must not be invisible -- it is counted onto the run's upload state, so a
            # run that completes "uploaded" with missing results can be told from a clean one.
            if dropped_results is not None:
                dropped_results.append(_fmt_error(exc))
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
    names = [scorer_name(s) for s in scorers]
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
        _scorer_call_plan(s)  # fail fast on an unsupported scorer signature (not a per-case error)
    # A Score row's identity IS its emitted-metric name, and the platform keys that metric's
    # direction/threshold on the name. Two scorers resolving to the same name make the policy
    # ambiguous, so the platform drops the metric to non-directional. Catch the static case here,
    # before a single case runs.
    seen: dict[str, int] = {}
    for s in scorers:
        n = scorer_name(s)
        seen[n] = seen.get(n, 0) + 1
    duplicated = sorted(n for n, count in seen.items() if count > 1)
    if duplicated:
        listed = ", ".join(repr(n) for n in duplicated)
        raise ValueError(
            f"two or more scorers report the same metric name ({listed}); metric names must be "
            "unique within a run. Give each scorer a distinct name (or key)."
        )
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
    ``on_case_start``/``on_case_complete`` are internal hooks the runner uses to
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

    # A run is not identified by which SDK produced it, so no SDK-language identity is reported.
    # But git/CI provenance (commit/branch/dirty, CI build) is NON-IDENTITY reproducibility
    # metadata — it rides the wire as free-form run metadata (user keys win on conflict), so the
    # platform can tie a run to the exact commit + dataset version. Same combined view backs the
    # local artifact.
    from traceroot.eval.provenance import collect_run_provenance

    run_metadata = collect_run_provenance(metadata, detect_dirty=False)

    dataset_name = snapshot.name
    # Cloud-only: an explicit transport wins; otherwise build a reporting transport from
    # credentials + the synced dataset. Evaluation always reports -- there is no local run.
    if transport is not None:
        active_transport: EvalTransport = transport
    else:
        # Auto-provision a locally-authored, unsynced Dataset: publish it once so the run has a
        # server-side version to attach to -- the user never writes a manual "sync then run" step
        # (matches how Braintrust/Laminar provision on run). Idempotent: unchanged content reuses
        # the current version; a changed dataset under an existing name prompts to confirm the new
        # version (TTY) before publishing. Only for a local Dataset with credentials; a pulled
        # dataset (already synced), an explicit dataset_id, or an explicit transport skip this.
        if (
            isinstance(data, Dataset)
            and data.dataset_version_id is None
            and dataset_id is None
        ):
            from traceroot.eval.platform import _resolve_credentials

            api_key, _ = _resolve_credentials(None, None)
            if api_key:
                from traceroot.eval.dataset_sync import PlatformDatasetSync

                data.push(PlatformDatasetSync())  # stamps dataset_version_id (may prompt / abort)
        active_transport = _auto_transport(
            data, scorers, dataset_id, candidate_version, environment
        )
        if active_transport is None:
            # _auto_transport returns None for two unrelated reasons; saying "no credentials" for
            # both misdirects a user who HAS a key but passed inline cases or an unsynced snapshot.
            from traceroot.eval.platform import _resolve_credentials as _creds

            if _creds(None, None)[0]:
                raise RuntimeError(
                    "evaluate() reports to the TraceRoot platform, but this run has no synced "
                    "dataset to report against. Pass a Dataset that was pulled or pushed "
                    "(pull_dataset(...) / Dataset.push(...)), or pass dataset_id=..., or pass an "
                    "explicit transport= (e.g. FakeTransport() to run offline)."
                )
            raise RuntimeError(
                "evaluate() reports to the TraceRoot platform, but no credentials were found. "
                "Set TRACEROOT_API_KEY (initialize traceroot or set the env var), or pass an "
                "explicit transport= (e.g. FakeTransport() to run offline)."
            )

    # Forward scorer comparison metadata (value_type/direction/threshold) from the actual
    # scorer callables when the transport accepts specs and the caller did not pre-set them.
    # Must happen BEFORE create_run so the descriptors reach registration (each emitted score's
    # per-score ``passed`` is derived from its owning scorer's declared threshold/direction).
    from traceroot.eval.scorers import describe_scorers

    if getattr(active_transport, "scorer_specs", "unset") is None:
        active_transport.scorer_specs = describe_scorers(scorers)

    # Client-side run id: the idempotency key for run registration AND the id carried on
    # the result (the platform run_id is separate, assigned by the backend).
    local_run_id = new_run_id()

    # Register the run up front so scorer descriptors + the run_id are available before the
    # per-case traces are stamped.
    run_handle = active_transport.create_run(
        name=name,
        dataset_name=dataset_name,
        metadata=run_metadata,
        client_run_id=local_run_id,
    )
    # Surface the registered ids immediately so a caller that is interrupted mid-run (e.g. the runner
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
    # Reuse the process-wide bounded pool (not a fresh per-run one) so timed-out, uncancellable sync
    # tasks can't accumulate leaked threads across repeated evaluations.
    sync_executor = _sync_executor(max_concurrency)
    results_list: list[EvalItemResult] = []
    summary: dict[str, Any] = {}
    # definition name -> the metric names it emitted, accumulated across cases (union: a metric
    # emitted by only some cases still counts). Recorded during execution, sent at completion.
    emitted_ownership: dict[str, set[str]] = {}
    # Per-case result POSTs the transport dropped (best-effort reporting), surfaced on the
    # upload state so a green run with silently-missing results is detectable.
    dropped_results: list[str] = []
    body_error: BaseException | None = None
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
                    emitted_ownership=emitted_ownership,
                    dropped_results=dropped_results,
                )
                for c in cases
            ]
        )
        results_list = list(item_results)
        summary = aggregate_scores(results_list)
    except BaseException as exc:  # remembered, then re-raised untouched
        body_error = exc
        raise
    finally:
        # The shared executor is intentionally NOT shut down here — it's reused across runs so
        # orphaned timed-out threads stay bounded to its fixed worker count.
        if reporter is not None:
            reporter.finish()
        # Finish inside finally so a mid-run failure never leaves the run open; a registered run
        # is never left orphaned in 'running'. Completion is the LAST thing to run, so it must
        # never become the story: if the run already failed, the original error keeps propagating
        # and the completion failure rides along as a note (it used to REPLACE the real cause --
        # a /complete 400 buried the actual exception under a second traceback).
        try:
            upload = active_transport.finish_run(
                run_handle,
                status=None,
                emitted_metrics={k: sorted(v) for k, v in emitted_ownership.items()},
            )
        except BaseException as completion_exc:
            if body_error is None:
                # The run itself succeeded: one clear error naming the completion failure, with
                # the transport exception carried as data (`from None`) rather than chained into
                # a double traceback.
                raise EvalCompletionError(
                    "the evaluation ran but the run could not be finalized on the platform: "
                    f"{_fmt_error(completion_exc)}. The run stays 'running' until a retry "
                    "completes it.",
                    completion_exc,
                ) from None
            body_error.add_note(
                f"run completion also failed: {_fmt_error(completion_exc)} "
                "(the run may stay 'running' on the platform)"
            )
        else:
            if dropped_results and isinstance(upload, UploadState):
                upload = dataclasses.replace(
                    upload, failed_result_count=len(dropped_results)
                )

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
        rname = scorer_name(rs, "run_scorer")
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
