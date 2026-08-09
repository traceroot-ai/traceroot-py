"""High-level evaluation API: the reusable ``Evaluation`` definition and the
``evaluate`` / ``evaluate_async`` convenience functions.

Both go through one code path (``engine._run`` / ``engine._run_async``): the
convenience functions build an ``Evaluation`` and run it. An ``Evaluation`` is a
mutable, reusable definition (compose, reuse in CI, select a
subset); calling ``run()`` produces an immutable ``EvalRunResult``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from traceroot.eval.engine import _run, _run_async
from traceroot.eval.results import EvalRunResult
from traceroot.eval.transport import EvalTransport
from traceroot.eval.types import Dataset, DatasetSnapshot, EvalCase, ScorerContext

DataSource = Dataset | DatasetSnapshot | Sequence["EvalCase | dict"]


class Evaluation:
    """A reusable, code-level evaluation definition.

    Mutable in user code; changing it affects the next ``run()``. Evaluation is cloud-only:
    every run reports to the platform, which needs credentials and a synced dataset
    (pulled/pushed, or an explicit ``dataset_id``); pass ``report_to=`` to supply an explicit
    transport. ``timeout`` bounds each case; ``metadata`` is attached to the run record;
    ``run_scorers`` run whole-run scorers. Candidate-vs-baseline comparison is the backend's
    job (the SDK reports raw runs; it does not compare). ``retry`` is not implemented and is
    rejected rather than silently ignored.
    """

    def __init__(
        self,
        *,
        name: str,
        dataset: DataSource,
        task: Callable[[Any], Any],
        scorers: Sequence[Callable[[ScorerContext], Any]],
        run_scorers: Sequence[Callable[[Any], Any]] | None = None,
        candidate_version: str | None = None,
        main_score: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_concurrency: int = 10,
        timeout: float | None = None,
        retry: Any = None,
        select: Callable[[EvalCase], bool] | None = None,
        report_to: EvalTransport | None = None,
        dataset_id: str | None = None,
        environment: str = "evaluation",
        progress: bool | None = None,
    ) -> None:
        if retry is not None:
            # Deferred by design: automatic retry can bias nondeterministic results and
            # hide flaky apps. Reject explicitly so it is never a silent no-op.
            raise NotImplementedError(
                "retry is not implemented in V1 (its semantics are deliberately deferred). "
                "Handle retries inside the task, or omit retry."
            )
        self.name = name
        self.dataset = dataset
        self.task = task
        self.scorers = scorers
        self.run_scorers = run_scorers
        self.candidate_version = candidate_version
        self.main_score = main_score
        self.metadata = metadata
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.retry = retry
        self.select = select
        self.report_to = report_to
        self.dataset_id = dataset_id
        self.environment = environment
        self.progress = progress

    def _kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        base = dict(
            name=self.name,
            data=self.dataset,
            task=self.task,
            scorers=self.scorers,
            max_concurrency=self.max_concurrency,
            transport=self.report_to,
            dataset_id=self.dataset_id,
            candidate_version=self.candidate_version,
            main_score=self.main_score,
            environment=self.environment,
            select=self.select,
            run_scorers=self.run_scorers,
            timeout=self.timeout,
            metadata=self.metadata,
            progress=self.progress,
        )
        base.update(overrides)
        return base

    def run(self, **overrides: Any) -> EvalRunResult:
        return _run(**self._kwargs(overrides))

    async def run_async(self, **overrides: Any) -> EvalRunResult:
        return await _run_async(**self._kwargs(overrides))


def _resolve_dataset(dataset: DataSource | None, data: DataSource | None) -> DataSource:
    source = dataset if dataset is not None else data
    if source is None:
        raise ValueError("evaluate() requires 'dataset' (or the 'data' alias)")
    return source


def evaluate(
    *,
    name: str,
    dataset: DataSource | None = None,
    data: DataSource | None = None,  # alias for dataset
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    report_to: EvalTransport | None = None,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    main_score: str | None = None,
    primary_metric: str | None = None,  # alias for main_score (the headline metric)
    environment: str = "evaluation",
    select: Callable[[EvalCase], bool] | None = None,
    run_scorers: Sequence[Callable[[Any], Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    progress: bool | None = None,
    retry: Any = None,
) -> EvalRunResult:
    """Construct-and-run an :class:`Evaluation`. ``dataset=`` is primary; ``data=`` is an alias."""
    return Evaluation(
        name=name,
        dataset=_resolve_dataset(dataset, data),
        task=task,
        scorers=scorers,
        max_concurrency=max_concurrency,
        report_to=report_to or transport,
        dataset_id=dataset_id,
        candidate_version=candidate_version,
        main_score=main_score if main_score is not None else primary_metric,
        environment=environment,
        select=select,
        run_scorers=run_scorers,
        metadata=metadata,
        timeout=timeout,
        progress=progress,
        retry=retry,  # honor the documented NotImplementedError guard instead of a raw TypeError
    ).run()


async def evaluate_async(
    *,
    name: str,
    dataset: DataSource | None = None,
    data: DataSource | None = None,
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    report_to: EvalTransport | None = None,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    main_score: str | None = None,
    primary_metric: str | None = None,  # alias for main_score (the headline metric)
    environment: str = "evaluation",
    select: Callable[[EvalCase], bool] | None = None,
    run_scorers: Sequence[Callable[[Any], Any]] | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    progress: bool | None = None,
    retry: Any = None,
) -> EvalRunResult:
    """Async form of :func:`evaluate`."""
    return await Evaluation(
        name=name,
        dataset=_resolve_dataset(dataset, data),
        task=task,
        scorers=scorers,
        max_concurrency=max_concurrency,
        report_to=report_to or transport,
        dataset_id=dataset_id,
        candidate_version=candidate_version,
        main_score=main_score if main_score is not None else primary_metric,
        environment=environment,
        select=select,
        run_scorers=run_scorers,
        metadata=metadata,
        timeout=timeout,
        progress=progress,
        retry=retry,  # honor the documented NotImplementedError guard instead of a raw TypeError
    ).run_async()
