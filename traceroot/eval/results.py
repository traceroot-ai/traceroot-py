"""Result and summary types for offline evaluation.

Structured, JSON-serializable, inspectable outputs plus per-scorer aggregation.
A completed run is an immutable execution record (see architecture v2).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
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
    # Per-case result POSTs that failed and were dropped (reporting is best-effort so the run
    # still completes). Counted so a run that reports "uploaded" with silently-missing results
    # is detectable instead of looking green.
    failed_result_count: int = 0

    @property
    def partial(self) -> bool:
        """True when the run was completed but some per-case results never reached the platform."""
        return self.failed_result_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "dashboard_url": self.dashboard_url,
            "failed_result_count": self.failed_result_count,
        }


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


def case_status(item: EvalItemResult) -> str:
    """Derive ``errored`` | ``not_scored`` for one item.

    A case with a task error OR a scorer error is ``errored``. Otherwise it is
    ``not_scored``: the SDK records every emitted score with its own per-score
    ``passed`` verdict (see the transport) and invents no case-level headline
    pass/fail.
    """
    if item.error is not None or item.scorer_errors:
        return "errored"
    return "not_scored"


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
            # A non-finite value (NaN/inf) must not fold into the mean: it would make the local
            # aggregate and run.json disagree with the wire (where a non-finite score is errored)
            # and .summary(), and a bare `NaN` token makes the artifact invalid per the JSON spec
            # (strict parsers, including JS `JSON.parse`, reject it). Exclude it from the numeric
            # aggregate; it still counts as a produced score.
            if isinstance(score.value, (int, float)) and math.isfinite(float(score.value)):
                numeric_sums[score.name] = numeric_sums.get(score.name, 0.0) + float(score.value)
                numeric_counts[score.name] = numeric_counts.get(score.name, 0) + 1

    summary: dict[str, ScoreSummary] = {}
    for name in order:
        n = numeric_counts.get(name, 0)
        mean = numeric_sums[name] / n if n else None
        summary[name] = ScoreSummary(name=name, mean=mean, count=total_counts[name])
    return summary


def _score_verdict(value: Any, owner: dict[str, Any] | None) -> bool | None:
    """Whether one score passes, given its ALREADY-RESOLVED owning scorer spec. Parity with
    ``PlatformTransport._score_passed``: a bool value IS its verdict; a numeric value passes iff it
    clears the owner's declared threshold in its declared direction (defaulting to higher_is_better).
    None (no verdict) for a non-finite value, no owner, or an owner with no threshold / 'none'
    direction — the SDK never fabricates a pass/fail."""
    if isinstance(value, bool):
        return value
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if owner is None:
        return None
    threshold = owner.get("threshold")
    direction = owner.get("direction")
    direction = direction if direction is not None else "higher_is_better"
    if threshold is None or direction == "none":
        return None
    return value <= threshold if direction == "lower_is_better" else value >= threshold


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
    metadata: dict[str, Any] | None = None  # run context (model, prompt, branch, CI, ...)
    # Declared scorer policy (name/version/value_type/direction/threshold) captured at run time.
    # Retained so an explicit upload() re-declares each metric's threshold/direction to the platform
    # instead of re-registering policy-less -- otherwise a re-upload's per-score ``passed`` verdicts
    # would silently disagree with the original run's.
    scorer_specs: list[dict[str, Any]] | None = None

    # --- inspection ---
    @property
    def results(self) -> list[EvalItemResult]:
        return self.item_results

    def _by_status(self, status: str) -> list[EvalItemResult]:
        return [it for it in self.item_results if case_status(it) == status]

    def errors(self) -> list[EvalItemResult]:
        return [it for it in self.item_results if it.error is not None or it.scorer_errors]

    @property
    def case_count(self) -> int:
        return len(self.item_results)

    @property
    def errored(self) -> int:
        return len(self._by_status("errored"))

    @property
    def not_scored(self) -> int:
        return len(self._by_status("not_scored"))

    @property
    def task_error_count(self) -> int:
        return sum(1 for it in self.item_results if it.error is not None)

    @property
    def scorer_error_count(self) -> int:
        return sum(len(it.scorer_errors) for it in self.item_results)

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
                "errored": self.errored,
                "not_scored": self.not_scored,
                "task_errors": self.task_error_count,
                "scorer_errors": self.scorer_error_count,
            },
            "item_results": [it.to_dict() for it in self.item_results],
            "score_summary": {k: v.to_dict() for k, v in self.score_summary.items()},
            "metadata": self.metadata,
            "scorer_specs": self.scorer_specs,
            "upload": self.upload_state.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EvalRunResult:
        """Backwards-compatible reader for BOTH V1 run-artifact shapes:

        - the ``EvalRunResult.save()`` shape (has ``item_results``), and
        - the runner's ``run.json`` artifact (``kind == "eval_run"``, with an
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
            metadata=d.get("metadata"),
            scorer_specs=d.get("scorer_specs"),
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
            upload_state=UploadState(**d.get("upload", {})),
            local_run_id=d.get("local_run_id", ""),
            candidate_version=d.get("candidate_version"),
            dataset=RunDatasetRef(**ds) if ds else None,
            run_id=d.get("run_id"),
            metadata=d.get("metadata"),
            # Restore the declared policy so a run loaded from a runner artifact re-uploads under
            # the thresholds it was scored with (older artifacts have none -> falls back to names).
            scorer_specs=d.get("scorer_specs"),
        )

    def save(self, path: str) -> None:
        # Atomic write: a temp file + os.replace, so an interrupted write can't destroy a
        # previously valid artifact (leaving load() unable to read either version).
        text = json.dumps(serialize_value(self.to_dict()), ensure_ascii=False)
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, p)

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
                # Include scorers that produced scores AND ones that errored on every case (absent
                # from score_summary), so an all-failing scorer still appears in run registration.
                scorer_names=list(
                    dict.fromkeys(
                        [
                            *self.score_summary,
                            *(s.name for item in self.item_results for s in item.scores),
                            *(name for item in self.item_results for name in item.scorer_errors),
                        ]
                    )
                ),
                candidate_version=self.candidate_version,
                dataset_version_id=self.dataset.dataset_version_id,
                client_run_id=self.local_run_id,
                # Re-declare each metric's threshold/direction (captured at run time) so a
                # re-upload's per-score `passed` matches the original run instead of registering
                # policy-less. None (an older/loaded run without specs) falls back to names.
                scorer_specs=self.scorer_specs,
            )
        dataset_name = self.dataset.dataset_id if self.dataset else "<inline>"
        run = active.create_run(
            name=self.name,
            dataset_name=dataset_name,
            metadata=self.metadata,  # preserve the run's metadata/provenance on re-upload
            client_run_id=self.local_run_id,
        )
        for item in self.item_results:
            active.record_item_result(run, item)
            active.record_scores(run, item.case_id, item.scores)
        self.upload_state = active.finish_run(run, status=None)
        # Keep the existing server run id if this transport doesn't expose one (don't erase it).
        self.run_id = getattr(active, "run_id", None) or self.run_id
        return self

    def summary(self) -> str:
        return str(self)

    def __str__(self) -> str:
        head = (
            f"EvalRunResult(name={self.name!r}, cases={self.case_count}, "
            f"errored={self.errored}, not_scored={self.not_scored}, "
            f"task_errors={self.task_error_count}, upload={self.upload_state.status})"
        )
        lines = [head]
        passed, judged = self._pass_tally()
        for name, summ in self.score_summary.items():
            mean = "n/a" if summ.mean is None else f"{summ.mean:.4g}"
            n = judged.get(name, 0)
            pass_seg = f" pass={passed.get(name, 0)}/{n}" if n else ""
            lines.append(f"  {name}: mean={mean}{pass_seg} count={summ.count}")
        return "\n".join(lines)

    def _pass_tally(self) -> tuple[dict[str, int], dict[str, int]]:
        """(passed, judged) counts per metric for ``summary()``. Resolves each score's owning scorer
        EXACTLY as ``PlatformTransport._score_policy`` does — a lone scorer emitting a lone metric
        owns it name-agnostically (single emission), otherwise the emitted name must match a declared
        scorer name — so the local pass-rate equals the platform's. Metrics with no resolvable policy
        contribute no pass-rate, only a count."""
        specs = self.scorer_specs or []
        by_name = {s.get("name"): s for s in specs}
        passed: dict[str, int] = {}
        judged: dict[str, int] = {}
        for item in self.item_results:
            single = len(item.scores) == 1
            for score in item.scores:
                owner = specs[0] if (len(specs) == 1 and single) else by_name.get(score.name)
                verdict = _score_verdict(score.value, owner)
                if verdict is None:
                    continue
                judged[score.name] = judged.get(score.name, 0) + 1
                if verdict:
                    passed[score.name] = passed.get(score.name, 0) + 1
        return passed, judged
