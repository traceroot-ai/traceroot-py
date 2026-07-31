"""Result and summary types for offline evaluation.

Structured, JSON-serializable, inspectable outputs plus per-scorer aggregation.
A completed run is an immutable execution record (see architecture v2).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Literal

from traceroot.eval.types import Score
from traceroot.utils import serialize_value


@dataclasses.dataclass
class EvalItemResult:
    """The outcome of running one case: the task output and every scorer result."""

    case_id: str
    input: Any
    output: Any | None
    expected: Any | None
    scores: list[Score]
    scorer_errors: dict[str, str]
    error: str | None
    trace_id: str | None
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "input": self.input,
            "output": self.output,
            "expected": self.expected,
            "scores": [dataclasses.asdict(s) for s in self.scores],
            "scorer_errors": self.scorer_errors,
            "error": self.error,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalItemResult:
        return cls(
            case_id=d["case_id"],
            input=d.get("input"),
            output=d.get("output"),
            expected=d.get("expected"),
            scores=[Score(**s) for s in d.get("scores", [])],
            scorer_errors=d.get("scorer_errors", {}),
            error=d.get("error"),
            trace_id=d.get("trace_id"),
            duration_ms=d.get("duration_ms"),
        )


@dataclasses.dataclass
class UploadState:
    """Explicit record of the run's platform persistence. Never silent."""

    status: Literal["uploaded"] = "uploaded"
    dashboard_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "dashboard_url": self.dashboard_url}


@dataclasses.dataclass
class ScoreSummary:
    """Aggregate of one score name across a run."""

    name: str
    mean: float | None
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "mean": self.mean, "count": self.count}


@dataclasses.dataclass(frozen=True)
class RunDatasetRef:
    """Immutable description of the exact dataset version/content a run executed."""

    dataset_id: str
    revision: str
    dataset_version_id: str | None
    case_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "revision": self.revision,
            "dataset_version_id": self.dataset_version_id,
            "case_count": self.case_count,
        }


@dataclasses.dataclass
class RunView:
    """The read-only view passed to a whole-run scorer."""

    name: str
    item_results: list[EvalItemResult]
    score_summary: dict[str, ScoreSummary]


def case_status(item: EvalItemResult, pass_threshold: float = 1.0) -> str:
    """Derive passed | failed | errored | not_scored for one item.

    A task error is 'errored' (distinct from a scorer error). A case with no
    numeric/boolean score is 'not_scored' (distinct from a score of zero).
    """
    if item.error is not None:
        return "errored"
    main: float | None = None
    for s in item.scores:
        if isinstance(s.value, bool):
            main = 1.0 if s.value else 0.0
            break
        if isinstance(s.value, (int, float)):
            main = float(s.value)
            break
    if main is None:
        return "not_scored"
    return "passed" if main >= pass_threshold else "failed"


def aggregate_scores(item_results: list[EvalItemResult]) -> dict[str, ScoreSummary]:
    """Aggregate every produced score by name.

    Numeric and boolean values contribute to ``mean``; categorical (str) values
    contribute to ``count`` only. ``mean`` is ``None`` when a name has no numeric
    values. Scores never produced (scorer errored) do not appear.
    """
    numeric_sums: dict[str, float] = {}
    numeric_counts: dict[str, int] = {}
    total_counts: dict[str, int] = {}
    order: list[str] = []

    for item in item_results:
        for score in item.scores:
            if score.name not in total_counts:
                order.append(score.name)
                total_counts[score.name] = 0
            total_counts[score.name] += 1
            if isinstance(score.value, (int, float)):  # bool -> 1.0/0.0
                numeric_sums[score.name] = numeric_sums.get(score.name, 0.0) + float(score.value)
                numeric_counts[score.name] = numeric_counts.get(score.name, 0) + 1

    summary: dict[str, ScoreSummary] = {}
    for name in order:
        n = numeric_counts.get(name, 0)
        mean = numeric_sums[name] / n if n else None
        summary[name] = ScoreSummary(name=name, mean=mean, count=total_counts[name])
    return summary


@dataclasses.dataclass
class EvalRunResult:
    """The full, immutable result of an evaluation run."""

    name: str
    item_results: list[EvalItemResult]
    score_summary: dict[str, ScoreSummary]
    upload_state: UploadState
    local_run_id: str = ""
    candidate_version: str | None = None
    dataset: RunDatasetRef | None = None
    run_id: str | None = None  # server-assigned id when uploaded
    run_scores: list[Score] = dataclasses.field(default_factory=list)  # whole-run scores
    run_scorer_errors: dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] | None = None  # run context (model, prompt, branch, CI, ...)

    # --- inspection ---
    @property
    def results(self) -> list[EvalItemResult]:
        return self.item_results

    def _by_status(self, status: str) -> list[EvalItemResult]:
        return [it for it in self.item_results if case_status(it) == status]

    def failures(self) -> list[EvalItemResult]:
        return self._by_status("failed")

    def errors(self) -> list[EvalItemResult]:
        return [it for it in self.item_results if it.error is not None or it.scorer_errors]

    @property
    def case_count(self) -> int:
        return len(self.item_results)

    @property
    def passed(self) -> int:
        return len(self._by_status("passed"))

    @property
    def failed(self) -> int:
        return len(self._by_status("failed"))

    @property
    def not_scored(self) -> int:
        return len(self._by_status("not_scored"))

    @property
    def task_error_count(self) -> int:
        return sum(1 for it in self.item_results if it.error is not None)

    @property
    def scorer_error_count(self) -> int:
        return sum(len(it.scorer_errors) for it in self.item_results)

    @property
    def scored_count(self) -> int:
        return self.passed + self.failed

    # --- serialization ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "name": self.name,
            "local_run_id": self.local_run_id,
            "run_id": self.run_id,
            "candidate_version": self.candidate_version,
            "dataset": self.dataset.to_dict() if self.dataset else None,
            "counts": {
                "case_count": self.case_count,
                "scored_count": self.scored_count,
                "passed": self.passed,
                "failed": self.failed,
                "not_scored": self.not_scored,
                "task_errors": self.task_error_count,
                "scorer_errors": self.scorer_error_count,
            },
            "item_results": [it.to_dict() for it in self.item_results],
            "score_summary": {k: v.to_dict() for k, v in self.score_summary.items()},
            "run_scores": [dataclasses.asdict(s) for s in self.run_scores],
            "run_scorer_errors": self.run_scorer_errors,
            "metadata": self.metadata,
            "upload": self.upload_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalRunResult:
        """Backwards-compatible reader for BOTH V1 run-artifact shapes:

        - the ``EvalRunResult.save()`` shape (has ``item_results``), and
        - the CLI runner's ``run.json`` artifact (``kind == "eval_run"``, with an
          embedded ``cases`` array and a ``scores`` summary map).

        This lets ``--baseline`` load an artifact created by either path without a
        schema ``KeyError``.
        """
        if d.get("kind") == "eval_run" or ("cases" in d and "item_results" not in d):
            return cls._from_runner_artifact(d)
        ds = d.get("dataset")
        return cls(
            name=d["name"],
            item_results=[EvalItemResult.from_dict(it) for it in d.get("item_results", [])],
            score_summary={k: ScoreSummary(**v) for k, v in d.get("score_summary", {}).items()},
            upload_state=UploadState(**d.get("upload", {})),
            local_run_id=d.get("local_run_id", ""),
            candidate_version=d.get("candidate_version"),
            dataset=RunDatasetRef(**ds) if ds else None,
            run_id=d.get("run_id"),
            run_scores=[Score(**s) for s in d.get("run_scores", [])],
            run_scorer_errors=d.get("run_scorer_errors", {}),
            metadata=d.get("metadata"),
        )

    @classmethod
    def _from_runner_artifact(cls, d: dict[str, Any]) -> EvalRunResult:
        """Adapt the runner's ``run.json`` into an EvalRunResult sufficient for
        comparison. Per-case input/output/expected live only in the ``.cases.jsonl``
        sidecar, not run.json, so they are reconstructed as None; ``case_id`` and
        scores (what row-matched comparison needs) are preserved."""
        items: list[EvalItemResult] = []
        for c in d.get("cases", []):
            scores = [
                Score(
                    name=s["scorer_name"],
                    value=s["value"],
                    comment=s.get("explanation"),
                    version=s.get("scorer_version"),
                )
                for s in c.get("scores", [])
            ]
            scorer_errors = {
                se["scorer_name"]: se.get("message", "") for se in c.get("scorer_errors", [])
            }
            items.append(
                EvalItemResult(
                    case_id=c["case_id"],
                    input=None,
                    output=None,
                    expected=None,
                    scores=scores,
                    scorer_errors=scorer_errors,
                    error=c.get("task_error"),
                    trace_id=c.get("trace_id"),
                    duration_ms=c.get("duration_ms"),
                )
            )
        ds = d.get("dataset")
        return cls(
            name=d.get("evaluation_name") or d.get("name", ""),
            item_results=items,
            score_summary={k: ScoreSummary(**v) for k, v in d.get("scores", {}).items()},
            upload_state=UploadState(),
            local_run_id=d.get("local_run_id", ""),
            candidate_version=d.get("candidate_version"),
            dataset=RunDatasetRef(**ds) if ds else None,
            run_id=d.get("run_id"),
            metadata=d.get("metadata"),
        )

    def save(self, path: str) -> None:
        Path(path).write_text(
            json.dumps(serialize_value(self.to_dict()), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str) -> EvalRunResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def upload(self, transport: Any = None) -> EvalRunResult:
        """Explicitly upload this retained run's results/scores (idempotent).

        Replays the item results through the reporting layer, preserving the local
        ``test_case_id``s; the ``local_run_id`` is the idempotency key so a retried
        upload does not duplicate the run. If ``transport`` is omitted, a
        ``PlatformTransport`` is built from this run's dataset ref + scorer names
        (requires credentials). Documented limitation: trace SPANS are not
        re-uploadable -- only ``trace_id`` links present from the original run are sent.
        """
        if not self.local_run_id:
            raise ValueError(
                "run.upload() needs a local_run_id as the idempotency key; load a run "
                "produced by evaluate() or set local_run_id before uploading"
            )
        active = transport
        if active is None:
            from traceroot.eval.platform import PlatformTransport

            if self.dataset is None:
                raise ValueError("run.upload() needs a dataset ref or an explicit transport")
            active = PlatformTransport(
                self.dataset.dataset_id,
                scorer_names=list(self.score_summary.keys()),
                candidate_version=self.candidate_version,
                dataset_version_id=self.dataset.dataset_version_id,
                client_run_id=self.local_run_id,
            )
        dataset_name = self.dataset.dataset_id if self.dataset else "<inline>"
        run = active.create_run(
            name=self.name,
            dataset_name=dataset_name,
            metadata=None,
            client_run_id=self.local_run_id,
        )
        for item in self.item_results:
            active.record_item_result(run, item)
            active.record_scores(run, item.case_id, item.scores)
        self.upload_state = active.finish_run(run, status=None)
        self.run_id = getattr(active, "run_id", None)
        return self

    def summary(self) -> str:
        return str(self)

    def __str__(self) -> str:
        head = (
            f"EvalRunResult(name={self.name!r}, cases={self.case_count}, "
            f"passed={self.passed}, failed={self.failed}, not_scored={self.not_scored}, "
            f"task_errors={self.task_error_count}, upload={self.upload_state.status})"
        )
        lines = [head]
        for name, summ in self.score_summary.items():
            mean = "n/a" if summ.mean is None else f"{summ.mean:.4g}"
            lines.append(f"  {name}: mean={mean} count={summ.count}")
        return "\n".join(lines)
