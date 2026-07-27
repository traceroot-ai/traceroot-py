"""High-level evaluation API: the reusable ``Evaluation`` definition and the
``evaluate`` / ``evaluate_async`` convenience functions.

Both go through one code path (``engine._run`` / ``engine._run_async``): the
convenience functions build an ``Evaluation`` and run it. An ``Evaluation`` is a
mutable, reusable definition (compose, reuse in CI, pin a baseline, select a
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

    Mutable in user code; changing it affects the next ``run()``. Reporting is
    upload-by-default (like Braintrust/Langfuse/Laminar): with credentials and a platform
    dataset (pulled/pushed, or an explicit ``dataset_id``) the run reports to the backend;
    pass ``local=True`` to keep it local, or ``report_to=`` an explicit transport to control
    it (e.g. baseline linking). No credentials or a purely local dataset stays local.
    ``timeout`` bounds each case; ``metadata`` is attached to the run record; ``baseline``
    is attached so ``run.compare()`` needs no argument; ``run_scorers`` run whole-run
    scorers. ``retry`` is not implemented and is rejected rather than silently ignored.
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
        metadata: dict[str, Any] | None = None,
        baseline: EvalRunResult | None = None,
        max_concurrency: int = 10,
        timeout: float | None = None,
        retry: Any = None,
        select: Callable[[EvalCase], bool] | None = None,
        report_to: EvalTransport | None = None,
        dataset_id: str | None = None,
        environment: str = "evaluation",
        local: bool = False,
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
        self.metadata = metadata
        self.baseline = baseline
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.retry = retry
        self.select = select
        self.report_to = report_to
        self.dataset_id = dataset_id
        self.environment = environment
        self.local = local
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
            environment=self.environment,
            select=self.select,
            run_scorers=self.run_scorers,
            timeout=self.timeout,
            metadata=self.metadata,
            baseline=self.baseline,
            local=self.local,
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
    environment: str = "evaluation",
    select: Callable[[EvalCase], bool] | None = None,
    run_scorers: Sequence[Callable[[Any], Any]] | None = None,
    baseline: EvalRunResult | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    local: bool = False,
    progress: bool | None = None,
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
        environment=environment,
        select=select,
        run_scorers=run_scorers,
        baseline=baseline,
        metadata=metadata,
        timeout=timeout,
        local=local,
        progress=progress,
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
    environment: str = "evaluation",
    select: Callable[[EvalCase], bool] | None = None,
    run_scorers: Sequence[Callable[[Any], Any]] | None = None,
    baseline: EvalRunResult | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    local: bool = False,
    progress: bool | None = None,
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
        environment=environment,
        select=select,
        run_scorers=run_scorers,
        baseline=baseline,
        metadata=metadata,
        timeout=timeout,
        local=local,
        progress=progress,
    ).run_async()
