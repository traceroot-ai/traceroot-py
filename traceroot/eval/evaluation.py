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

from traceroot.eval.engine import _LOCAL_AND_TRANSPORT, _run, _run_async
from traceroot.eval.results import EvalRunResult
from traceroot.eval.transport import EvalTransport
from traceroot.eval.types import Dataset, DatasetSnapshot, EvalCase, ScorerContext

DataSource = Dataset | DatasetSnapshot | Sequence["EvalCase | dict"]


class Evaluation:
    """A reusable, code-level evaluation definition.

    Mutable in user code; changing it affects the next ``run()``. Evaluation is cloud-only:
    every run reports to the platform, which needs credentials and a synced dataset
    (pulled/pushed, or an explicit ``dataset_id``); pass ``report_to=`` to supply an explicit
    transport, or ``local=True`` to run in full and report nowhere.
    ``timeout`` bounds each case; ``metadata`` is attached to the run record.
    ``evaluation_key`` is the stable identity runs are grouped by, separate from the display
    ``name`` (the same split as a scorer's ``key``): set it to keep one history across a rename,
    or to group the Python and TypeScript runs of one evaluation. It defaults to ``name``.
    Candidate-vs-baseline comparison is the backend's job (the SDK reports raw runs; it does
    not compare). ``retry`` is not implemented and is rejected rather than silently ignored.
    """

    def __init__(
        self,
        *,
        name: str,
        dataset: DataSource,
        task: Callable[[Any], Any],
        scorers: Sequence[Callable[[ScorerContext], Any]],
        candidate_version: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_concurrency: int = 10,
        timeout: float | None = None,
        retry: Any = None,
        select: Callable[[EvalCase], bool] | None = None,
        report_to: EvalTransport | None = None,
        local: bool = False,
        dataset_id: str | None = None,
        environment: str = "evaluation",
        evaluation_key: str | None = None,
        progress: bool | None = None,
    ) -> None:
        if local and report_to is not None:
            raise ValueError(_LOCAL_AND_TRANSPORT)
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
        self.candidate_version = candidate_version
        self.metadata = metadata
        self.max_concurrency = max_concurrency
        self.timeout = timeout
        self.retry = retry
        self.select = select
        self.report_to = report_to
        self.local = local
        self.dataset_id = dataset_id
        self.environment = environment
        self.evaluation_key = evaluation_key
        self.progress = progress

    def _kwargs(self, overrides: dict[str, Any]) -> dict[str, Any]:
        base = dict(
            name=self.name,
            data=self.dataset,
            task=self.task,
            scorers=self.scorers,
            max_concurrency=self.max_concurrency,
            transport=self.report_to,
            local=self.local,
            dataset_id=self.dataset_id,
            candidate_version=self.candidate_version,
            environment=self.environment,
            evaluation_key=self.evaluation_key,
            select=self.select,
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


def evaluate(
    *,
    name: str,
    dataset: DataSource,
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    report_to: EvalTransport | None = None,
    local: bool = False,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    environment: str = "evaluation",
    evaluation_key: str | None = None,
    select: Callable[[EvalCase], bool] | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    progress: bool | None = None,
    retry: Any = None,
) -> EvalRunResult:
    """Construct-and-run an :class:`Evaluation`.

    ``dataset`` is a ``Dataset`` / ``DatasetSnapshot`` or an inline ``list`` of cases. Reporting to
    the platform requires a synced dataset, so an inline list must be run with ``local=True``.
    ``local=True`` is the shortcut for ``transport=FakeTransport()``: the run executes in full and
    returns a complete result, but reports nowhere -- no credentials, no dataset publish, no run record.
    """
    return Evaluation(
        name=name,
        dataset=dataset,
        task=task,
        scorers=scorers,
        max_concurrency=max_concurrency,
        report_to=report_to or transport,
        local=local,
        dataset_id=dataset_id,
        candidate_version=candidate_version,
        environment=environment,
        evaluation_key=evaluation_key,
        select=select,
        metadata=metadata,
        timeout=timeout,
        progress=progress,
        retry=retry,  # honor the documented NotImplementedError guard instead of a raw TypeError
    ).run()


async def evaluate_async(
    *,
    name: str,
    dataset: DataSource,
    task: Callable[[Any], Any],
    scorers: Sequence[Callable[[ScorerContext], Any]],
    max_concurrency: int = 10,
    transport: EvalTransport | None = None,
    report_to: EvalTransport | None = None,
    local: bool = False,
    dataset_id: str | None = None,
    candidate_version: str | None = None,
    environment: str = "evaluation",
    evaluation_key: str | None = None,
    select: Callable[[EvalCase], bool] | None = None,
    metadata: dict[str, Any] | None = None,
    timeout: float | None = None,
    progress: bool | None = None,
    retry: Any = None,
) -> EvalRunResult:
    """Async form of :func:`evaluate`."""
    return await Evaluation(
        name=name,
        dataset=dataset,
        task=task,
        scorers=scorers,
        max_concurrency=max_concurrency,
        report_to=report_to or transport,
        local=local,
        dataset_id=dataset_id,
        candidate_version=candidate_version,
        environment=environment,
        evaluation_key=evaluation_key,
        select=select,
        metadata=metadata,
        timeout=timeout,
        progress=progress,
        retry=retry,  # honor the documented NotImplementedError guard instead of a raw TypeError
    ).run_async()
