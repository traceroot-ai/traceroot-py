"""Stable programmatic runner entry point for offline evaluation.

Invoked out-of-process as::

    <interpreter> -m traceroot.eval.runner <eval paths...>

It imports each eval module, inspects module-level ``Evaluation`` instances (no
registry, no import-side-effect execution), runs them via the user's own installed
SDK, and streams NDJSON events on fd 3 (``TRACEROOT_EVAL_EVENT_FD``) or an event
file (``TRACEROOT_EVAL_EVENT_FILE``). Structured options come from the JSON env var
``TRACEROOT_EVAL_OPTIONS``; eval paths come from argv.

The runner ALWAYS exits 0 when the harness itself functioned - pass/fail is carried
in the events; the caller decides the process exit code. It emits ``fatal`` before
dying whenever it can.

Local-only by default: unless ``reporting`` is explicitly enabled, the runner
disables TraceRoot span export and never uses a reporting transport, so a run makes
no TraceRoot reporting/dataset-sync/telemetry-export/platform request - even if the
evaluated app called ``traceroot.initialize()``. (User task/scorer code may still
call LLM providers, tools, or other services; the runner neither intercepts nor
restricts that.)

The event stream is the authoritative interface.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import random
import sys
import threading
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import traceroot
import traceroot.eval as ev
from traceroot.constants import SDK_VERSION
from traceroot.eval.evaluation import Evaluation
from traceroot.eval.ids import new_run_id
from traceroot.eval.platform import PlatformTransport
from traceroot.eval.results import (
    EvalItemResult,
    EvalRunResult,
    RunDatasetRef,
    UploadState,
    aggregate_scores,
    case_status,
)
from traceroot.eval.scorers import describe_scorers
from traceroot.eval.types import Dataset, DatasetSnapshot, EvalCase
from traceroot.utils import serialize_value

_DEFAULT_RUN_DIR = ".traceroot/eval/runs"


class _CancelledError(Exception):
    """Internal: an evaluation was interrupted (SIGINT) and finalized as incomplete."""


# =============================================================================
# Event emitter (NDJSON, thread-safe)
# =============================================================================
class Emitter:
    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        """Best-effort: emitting is reporting, never control flow.

        The channel can die under us (the harness went away -> EPIPE) and the sink is caller-
        supplied. Emitting happens inside per-case hooks the engine calls UNGUARDED, so letting a
        write error out of here would abort the remaining cases over a reporting failure. Report
        the breakage on stderr and keep running.
        """
        try:
            line = json.dumps(event, default=str, ensure_ascii=False) + "\n"
            with self._lock:
                self._write(line)
        except Exception as exc:
            print(
                f"traceroot-eval: dropped a {event.get('type')} event: {_fmt(exc)}", file=sys.stderr
            )


def _isolated(fn: Callable[..., None]) -> Callable[..., None]:
    """Make a per-case hook non-fatal: the engine invokes these hooks unguarded, so a failure
    inside one (a dead event channel, a payload that resists shaping) would abort every case
    still to run. Report it on stderr and carry on. Cancellation (KeyboardInterrupt) is a
    BaseException and still propagates - stopping is exactly what it means."""

    def guarded(*args: Any, **kwargs: Any) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            print(f"traceroot-eval: {fn.__name__} hook failed: {_fmt(exc)}", file=sys.stderr)

    return guarded


def _open_channel() -> Callable[[str], None]:
    """Resolve the event sink: fd (TRACEROOT_EVAL_EVENT_FD) or file, else stdout."""
    event_file = os.environ.get("TRACEROOT_EVAL_EVENT_FILE")
    if event_file:
        # Long-lived stream for the whole run; closed at process exit.
        fh = open(event_file, "a", encoding="utf-8")  # noqa: SIM115

        def write_file(line: str) -> None:
            fh.write(line)
            fh.flush()

        return write_file

    fd_raw = os.environ.get("TRACEROOT_EVAL_EVENT_FD")
    fd = int(fd_raw) if fd_raw and fd_raw.isdigit() else None
    if fd is not None:
        stream = os.fdopen(fd, "w", encoding="utf-8")

        def write_fd(line: str) -> None:
            stream.write(line)
            stream.flush()

        return write_fd

    # Fallback: stdout (used only when no channel is configured, e.g. manual runs).
    def write_stdout(line: str) -> None:
        sys.stdout.write(line)
        sys.stdout.flush()

    return write_stdout


# =============================================================================
# Options
# =============================================================================
def _load_options() -> dict[str, Any]:
    raw = os.environ.get("TRACEROOT_EVAL_OPTIONS")
    return json.loads(raw) if raw else {}


def _enforce_local_only() -> None:
    """Guarantee no TraceRoot data leaves this process in local-only mode.

    Disables span export (so an app that called initialize() exports nothing) by
    resetting any existing client and forcing TRACEROOT_ENABLED=false BEFORE the
    eval modules import. Reporting transports are simply never constructed.
    """
    os.environ["TRACEROOT_ENABLED"] = "false"
    # Neutralize ambient credentials too: TRACEROOT_ENABLED=false disables span export, but a
    # reporting transport resolves its key straight from TRACEROOT_API_KEY (independent of
    # `enabled`), so a synced-dataset run would still upload. Drop the key so credential
    # resolution finds none and the run stays strictly local.
    os.environ.pop("TRACEROOT_API_KEY", None)
    try:
        traceroot.shutdown()
    except Exception:
        pass
    traceroot._client = None  # a later initialize() rebuilds a disabled client


# =============================================================================
# Discovery (module inspection, no registry)
# =============================================================================
def _iter_paths(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(sorted(f for f in p.rglob("*.py") if not f.name.startswith("_")))
        else:
            files.append(p)
    return files


def _import_module(path: Path) -> Any:
    mod_name = f"_traceroot_eval_{abs(hash(str(path.resolve())))}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # Put the eval file's own directory on sys.path while it executes so it can import sibling
    # modules (a multi-file eval suite invoked by path would otherwise fail on import).
    pkg_dir = str(path.resolve().parent)
    added = pkg_dir not in sys.path
    if added:
        sys.path.insert(0, pkg_dir)
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            try:
                sys.path.remove(pkg_dir)
            except ValueError:
                pass
    return module


def discover(paths: list[str]) -> list[tuple[str, Evaluation]]:
    """Import each eval file and collect module-level Evaluation instances,
    in definition order, deduped by object identity."""
    found: list[tuple[str, Evaluation]] = []
    seen: set[int] = set()
    for path in _iter_paths(paths):
        module = _import_module(path)
        for value in vars(module).values():
            if isinstance(value, Evaluation) and id(value) not in seen:
                seen.add(id(value))
                found.append((str(path), value))
    return found


# =============================================================================
# Sampling / filtering
# =============================================================================
def _cases(data: Any) -> list[EvalCase]:
    """EvalCase objects for the dataset (positional ids filled in), for count/selection checks."""
    if isinstance(data, DatasetSnapshot):
        return list(data.cases)
    if isinstance(data, Dataset):
        return list(data)  # active cases
    out: list[EvalCase] = []
    for i, item in enumerate(data):
        if isinstance(item, EvalCase):
            out.append(item if item.id else dataclasses.replace(item, id=f"case-{i}"))
        elif isinstance(item, dict):
            out.append(EvalCase(**{**item, "id": item.get("id") or f"case-{i}"}))
    return out


def _case_ids(data: Any) -> list[str]:
    return [c.id for c in _cases(data)]  # type: ignore[misc]


def _subset(
    data: Any, first: int | None, sample: int | None, seed: int
) -> tuple[set[str] | None, str, bool]:
    """Return (chosen_ids | None, run_mode, is_final).

    Seeded sampling selects by POSITION in the stable dataset order, not by case id,
    so it is reproducible across repeated imports even when freshly constructed cases
    receive new generated ULIDs each import (same seed + same dataset content order ->
    same cases selected). ``first`` is already order-based.
    """
    if first is None and sample is None:
        return None, "full", True
    ids = _case_ids(data)
    if first is not None:
        return set(ids[:first]), "first", False
    n = min(sample, len(ids)) if sample is not None else len(ids)
    positions = random.Random(seed).sample(range(len(ids)), n) if ids else []
    return {ids[i] for i in positions}, "sample", False


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _dataset_identity(data: Any) -> dict[str, Any]:
    """Best-effort dataset id / version id / name from the evaluation's data source."""
    return {
        "dataset_id": getattr(data, "dataset_id", None),
        "dataset_version_id": getattr(data, "dataset_version_id", None)
        or getattr(data, "base_version_id", None),
        "name": getattr(data, "name", None),
    }


# =============================================================================
# Event / artifact shaping
# =============================================================================
class ScorePolicy:
    """Resolves a score's pass/fail from the owning scorer's DECLARED threshold + direction.

    The resolution itself is PlatformTransport's, reused verbatim: its two methods read nothing
    but ``scorer_specs``, so rebinding them here keeps one implementation instead of a second,
    divergent one (constructing a PlatformTransport is impossible in a local run - it demands
    credentials). That is the point: the local artifact and the platform UI must never disagree
    about whether the same score passed.
    """

    _score_policy = PlatformTransport._score_policy
    _score_passed = PlatformTransport._score_passed

    def __init__(self, scorer_specs: list[dict[str, Any]] | None = None) -> None:
        self.scorer_specs = scorer_specs

    def passed(self, score: Any, *, single_emission: bool) -> bool | None:
        return self._score_passed(score, single_emission=single_emission)


def _evaluation_policy(evaluation: Evaluation) -> ScorePolicy:
    """The declared policy of the evaluation's own scorers - the SAME descriptors the engine
    hands the transport at registration, so both sides resolve verdicts from identical input."""
    try:
        return ScorePolicy(describe_scorers(evaluation.scorers))
    except Exception:
        # A scorer that resists introspection costs us verdicts, never the run.
        return ScorePolicy(None)


def _score_event(score: Any, policy: ScorePolicy, *, single_emission: bool) -> dict[str, Any]:
    return {
        "scorer_name": score.name,
        "scorer_version": score.version,
        "value": score.value,
        "passed": policy.passed(score, single_emission=single_emission),
        "explanation": score.comment,
    }


def _score_events(item: EvalItemResult, policy: ScorePolicy) -> list[dict[str, Any]]:
    single = len(item.scores) == 1
    return [_score_event(s, policy, single_emission=single) for s in item.scores]


def _scorer_error_events(item: EvalItemResult) -> list[dict[str, Any]]:
    # scorer_version is unknown at the error site (no Score was produced) -> null,
    # never an invented "1".
    return [
        {"scorer_name": name, "scorer_version": None, "message": msg}
        for name, msg in item.scorer_errors.items()
    ]


def _scorer_versions(result: EvalRunResult) -> dict[str, str | None]:
    """Map scorer name -> its declared version (from produced scores), else None."""
    versions: dict[str, str | None] = {}
    for item in result.item_results:
        for s in item.scores:
            versions.setdefault(s.name, s.version)
            if s.version is not None:
                versions[s.name] = s.version
    return versions


def _case_metadata(item: EvalItemResult, policy: ScorePolicy) -> dict[str, Any]:
    return {
        "case_id": item.case_id,
        "status": case_status(item),
        "scores": _score_events(item, policy),
        "task_error": item.error,
        "scorer_errors": _scorer_error_events(item),
        "trace_id": item.trace_id,
        "duration_ms": item.duration_ms,
    }


def _run_status(result: EvalRunResult, cancelled: bool) -> str:
    if cancelled:
        return "incomplete"
    if result.task_error_count or result.scorer_error_count:
        return "completed_with_errors"
    return "completed"


def _counts(result: EvalRunResult) -> dict[str, int]:
    return {
        "cases": result.case_count,
        "errored": result.errored,
        "task_errors": result.task_error_count,
        "scorer_errors": result.scorer_error_count,
        "not_scored": result.not_scored,
    }


def _ensure_gitignore(directory: Path) -> None:
    """Drop a '*' .gitignore in the artifact dir so local eval payloads (which may hold
    PII/secrets in case input/output) can't be accidentally committed. Never clobbers a
    user's existing file."""
    gi = directory / ".gitignore"
    if gi.exists():
        return
    try:
        gi.write_text(
            "# TraceRoot local evaluation artifacts -- may contain payloads. Do not commit.\n*\n"
        )
    except OSError:
        pass


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    _ensure_gitignore(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _safe_payload(value: Any) -> Any:
    """A case payload reduced to something JSON can hold.

    Task output is arbitrary user data: a reference cycle (or anything else the serializer
    chokes on) would otherwise raise while writing the artifact and take the WHOLE run down
    after every case had already succeeded. Degrade that one payload to an explicit marker
    instead - the case, its scores and the rest of the run survive, and nobody mistakes the
    marker for real data.
    """
    try:
        return serialize_value(value)
    except Exception as exc:
        return {"unserializable": True, "reason": _fmt(exc)}


def _truncate(value: Any, limit: int | None) -> tuple[Any, bool]:
    if limit is None:
        return value, False
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text.encode()) <= limit:
        return value, False
    # ``limit`` is a byte budget, so slice UTF-8 bytes (not characters) and decode back at a safe
    # boundary -- slicing characters would let a multi-byte (non-ASCII) preview exceed the budget.
    preview = text.encode()[:limit].decode("utf-8", errors="ignore")
    return {"truncated": True, "preview": preview}, True


def write_artifacts(
    result: EvalRunResult,
    run_path: Path,
    cases_path: Path,
    *,
    status: str,
    run_mode: str,
    is_final: bool,
    sample_count: int | None,
    sample_seed: int | None,
    candidate_version: str | None,
    max_payload_bytes: int | None = None,
    created_at: str | None = None,
    policy: ScorePolicy | None = None,
) -> dict[str, Any]:
    """Write the two-file artifact. Returns the artifact descriptor."""
    policy = policy or ScorePolicy(None)
    truncated_any = False
    case_lines: list[str] = []
    for item in result.item_results:
        inp, t1 = _truncate(_safe_payload(item.input), max_payload_bytes)
        out, t2 = _truncate(_safe_payload(item.output), max_payload_bytes)
        exp, t3 = _truncate(_safe_payload(item.expected), max_payload_bytes)
        truncated_any = truncated_any or t1 or t2 or t3
        record = {
            "schema_version": "1",
            "case_id": item.case_id,
            "status": case_status(item),
            "input": inp,
            "output": out,
            "expected": exp,
            "scores": _score_events(item, policy),
            "scorer_errors": _scorer_error_events(item),
            "task_error": item.error,
            "trace_id": item.trace_id,
            "duration_ms": item.duration_ms,
        }
        case_lines.append(json.dumps(serialize_value(record), ensure_ascii=False))
    _atomic_write(cases_path, "\n".join(case_lines) + ("\n" if case_lines else ""))

    payloads = "truncated" if truncated_any else "complete"
    artifact = {"run": str(run_path), "cases": str(cases_path), "payloads": payloads}
    run_doc = {
        "schema_version": "1",
        "kind": "eval_run",
        "local_run_id": result.local_run_id,
        "run_id": result.run_id,
        "created_at": created_at or _now_iso(),
        "evaluation_name": result.name,
        "status": status,
        "candidate_version": candidate_version,
        "run_mode": run_mode,
        "is_final": is_final,
        "sample": {"count": sample_count, "seed": sample_seed},
        "dataset": result.dataset.to_dict() if result.dataset else None,
        "scorers": [
            {"name": n, "version": _scorer_versions(result).get(n)} for n in result.score_summary
        ],
        # The full declared policy (threshold/direction/value_type), alongside the name/version
        # list above: a run loaded back from this artifact is re-uploadable, and without it the
        # re-upload registers policy-less and its per-score `passed` contradicts this run's.
        "scorer_specs": result.scorer_specs,
        "counts": _counts(result),
        "scores": {k: v.to_dict() for k, v in result.score_summary.items()},
        "upload": result.upload_state.to_dict(),
        "artifact": artifact,
        "cases": [_case_metadata(it, policy) for it in result.item_results],
    }
    _atomic_write(run_path, json.dumps(serialize_value(run_doc), ensure_ascii=False, indent=2))
    return artifact


# =============================================================================
# Suite execution
# =============================================================================
def run_suite(paths: list[str], options: dict[str, Any], emitter: Emitter) -> bool:
    """Discover + (list or run) the evaluations, emitting the full event stream."""
    reporting = bool(options.get("reporting"))
    if not reporting:
        _enforce_local_only()

    emitter.emit(
        {
            "type": "hello",
            "sdk_version": SDK_VERSION,
            "eval_api_version": ev.__api_version__,
            "capabilities": ev.capabilities(),
        }
    )

    try:
        discovered = discover(paths)
    except Exception as exc:  # import error is a harness signal
        emitter.emit(
            {
                "type": "fatal",
                "kind": "import_error",
                "message": _fmt(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return False

    filters = options.get("filter") or []
    if filters:
        wanted = set(filters)
        discovered = [(m, e) for (m, e) in discovered if e.name in wanted]

    if options.get("mode") == "list":
        emitter.emit(
            {
                "type": "definitions",
                "evaluations": [{"name": e.name, "module": m} for (m, e) in discovered],
            }
        )
        return False

    completed = 0
    cancelled = False
    for _module_path, evaluation in discovered:
        try:
            _run_one(evaluation, options, emitter)
            completed += 1
        except _CancelledError:  # SIGINT: this eval was finalized as incomplete; stop the suite
            completed += 1
            cancelled = True
            break

    emitter.emit({"type": "suite_completed", "evaluations": completed, "cancelled": cancelled})
    return cancelled


def _run_one(evaluation: Evaluation, options: dict[str, Any], emitter: Emitter) -> None:
    first = options.get("first")
    sample = options.get("sample")
    seed = int(options.get("sample_seed", 0) or 0)
    # An option is an OVERRIDE only when the caller actually supplied it; otherwise
    # the Evaluation's own value stands (never silently replaced with a runner default).
    candidate_version = options.get("candidate_version") or evaluation.candidate_version
    created_at = _now_iso()
    identity = _dataset_identity(evaluation.dataset)
    policy = _evaluation_policy(evaluation)

    chosen, run_mode, is_final = _subset(evaluation.dataset, first, sample, seed)
    base_select = evaluation.select
    if chosen is not None:
        select = (
            (lambda c: (base_select is None or base_select(c)) and c.id in chosen)
            if base_select is not None
            else (lambda c: c.id in chosen)
        )
    else:
        select = base_select

    # Report the count that will actually run: apply BOTH the sample subset and the evaluation's
    # select filter, so the start event's case_count matches the results (an Evaluation.select that
    # drops cases was previously ignored here).
    if select is not None:
        case_count = sum(1 for c in _cases(evaluation.dataset) if select(c))
    else:
        case_count = len(_case_ids(evaluation.dataset))
    emitter.emit(
        {
            "type": "evaluation_started",
            "name": evaluation.name,
            "created_at": created_at,
            "candidate_version": candidate_version,
            "run_mode": run_mode,
            "is_final": is_final,
            "dataset": {
                "case_count": case_count,
                "dataset_id": identity["dataset_id"],
                "dataset_version_id": identity["dataset_version_id"],
                "name": identity["name"],
            },
        }
    )

    # Collect completed cases so a cancelled run can still be finalized partially.
    collected: list[EvalItemResult] = []
    # Capture the registered run ids so a cancel finalizes the partial artifact under the SAME
    # ids the run registered with (reconcilable with the cloud run), not a fresh local id.
    registered: dict[str, str | None] = {}

    @_isolated
    def on_run_start(local_run_id: str, run_id: str | None) -> None:
        registered["local_run_id"] = local_run_id
        registered["run_id"] = run_id

    # The engine calls these hooks UNGUARDED, so anything they raise would abort the remaining
    # cases. Reporting a case must never cost us the run: isolate both bodies.
    @_isolated
    def on_start(case: EvalCase) -> None:
        emitter.emit({"type": "case_started", "case_id": case.id})

    @_isolated
    def on_complete(item: EvalItemResult, _dur: float) -> None:
        collected.append(item)
        emitter.emit({"type": "case_completed", **_case_metadata(item, policy)})

    # Only override the Evaluation's own concurrency/timeout when the option is present;
    # an absent option must NOT replace the definition's value with a runner default.
    run_kwargs: dict[str, Any] = {
        "candidate_version": candidate_version,
        "select": select,
        "on_case_start": on_start,
        "on_case_complete": on_complete,
        "on_run_start": on_run_start,
        # The runner speaks NDJSON on its own channel; never draw a progress bar.
        "progress": False,
    }
    if not bool(options.get("reporting")):
        # Local-only guarantee: run the engine in local mode, ignoring any transport the eval module
        # configured (report_to=). This reports nowhere AND activates the non-exporting evaluation
        # tracer, so neither lifecycle requests NOR per-case spans leave the process -- even if the
        # evaluated module re-enabled export via initialize(enabled=True), which would defeat the
        # transport-swap-only approach. (local + transport are mutually exclusive, so clear it.)
        run_kwargs["local"] = True
        run_kwargs["transport"] = None
    if options.get("max_concurrency"):
        run_kwargs["max_concurrency"] = int(options["max_concurrency"])
    if options.get("timeout") is not None:
        run_kwargs["timeout"] = options["timeout"]

    cancelled = False
    try:
        result = evaluation.run(**run_kwargs)
    except KeyboardInterrupt:
        # Finalize the in-flight run as incomplete from the cases that finished.
        cancelled = True
        is_final = False
        result = _partial_result(
            evaluation,
            collected,
            candidate_version,
            registered.get("local_run_id"),
            registered.get("run_id"),
        )

    status = _run_status(result, cancelled=cancelled)
    artifact = None
    if not options.get("no_artifact"):
        run_path, cases_path = _artifact_paths(options, result.local_run_id)
        artifact = write_artifacts(
            result,
            run_path,
            cases_path,
            status=status,
            run_mode=run_mode,
            is_final=is_final,
            sample_count=len(chosen) if chosen is not None else None,
            sample_seed=seed if chosen is not None else None,
            candidate_version=candidate_version,
            max_payload_bytes=options.get("max_payload_bytes"),
            created_at=created_at,
            policy=policy,
        )

    emitter.emit(
        {
            "type": "evaluation_completed",
            "name": evaluation.name,
            "status": status,
            "created_at": created_at,
            "local_run_id": result.local_run_id,
            "run_id": result.run_id,
            "dataset": {
                "dataset_id": result.dataset.dataset_id if result.dataset else None,
                "dataset_version_id": result.dataset.dataset_version_id if result.dataset else None,
            },
            "counts": _counts(result),
            "score_summary": {k: v.to_dict() for k, v in result.score_summary.items()},
            "artifact": artifact,
        }
    )
    if cancelled:
        raise _CancelledError()


def _partial_result(
    evaluation: Evaluation,
    collected: list[EvalItemResult],
    candidate_version: str | None,
    local_run_id: str | None = None,
    run_id: str | None = None,
) -> EvalRunResult:
    """Build an incomplete EvalRunResult from the cases that finished before cancel.

    Reuses the ids the run registered with (``local_run_id``/``run_id``) so the partial artifact
    reconciles with the cloud run; falls back to a fresh local id only if the run was cancelled
    before it ever registered.
    """
    data = evaluation.dataset
    if isinstance(data, DatasetSnapshot):
        dataset_id, revision, version_id = data.dataset_id, data.revision, data.base_version_id
    elif isinstance(data, Dataset):
        snap = data.snapshot()
        dataset_id, revision, version_id = snap.dataset_id, snap.revision, data.dataset_version_id
    else:
        dataset_id, revision, version_id = "ds_inline", "rev_partial", None
    return EvalRunResult(
        name=evaluation.name,
        item_results=list(collected),
        score_summary=aggregate_scores(list(collected)),
        upload_state=UploadState(),
        local_run_id=local_run_id or new_run_id(),
        run_id=run_id,
        candidate_version=candidate_version,
        dataset=RunDatasetRef(
            dataset_id=dataset_id,
            revision=revision,
            dataset_version_id=version_id,
            case_count=len(collected),
        ),
    )


def _artifact_paths(options: dict[str, Any], local_run_id: str) -> tuple[Path, Path]:
    if options.get("out_run") and options.get("out_cases"):
        return Path(options["out_run"]), Path(options["out_cases"])
    out_dir = Path(options.get("out_dir") or _DEFAULT_RUN_DIR)
    return out_dir / f"{local_run_id}.json", out_dir / f"{local_run_id}.cases.jsonl"


def _fmt(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _flush_spans() -> None:
    """Drain the span exporter before the process goes away. A no-op in local-only mode (no
    client, export disabled); best-effort otherwise - a flush that fails is a lost trace, not
    a failed run."""
    try:
        traceroot.flush()
    except Exception as exc:
        print(f"traceroot-eval: span flush failed: {_fmt(exc)}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Runner entry point. Exit-code contract (coordinated with the caller):

    - Normal completion (even with task/scorer errors): exit 0. Quality/error outcomes
      are conveyed by the EVENT STREAM, not the exit code.
    - Unrecoverable harness failure: emit a structured ``fatal`` event and STILL exit 0.
      The presence of the ``fatal`` event -- not a nonzero code -- is what marks a run a
      harness failure, so the caller reads the stream as authoritative.
    - Cancellation (SIGINT): finalize a partial ``incomplete`` artifact and exit 130
      (the one code that corroborates the ``fatal``/cancelled signal).

    Every return path flushes first: each result carries a ``trace_id``, so a span left
    unbatched at exit is a run whose trace drill-down is permanently empty.
    """
    paths = list(argv if argv is not None else sys.argv[1:])
    emitter = Emitter(_open_channel())
    try:
        # Parse options INSIDE the error boundary so malformed TRACEROOT_EVAL_OPTIONS produces a
        # structured `fatal` event (and the exit-0 contract) instead of an uncaught traceback.
        options = _load_options()
        cancelled = run_suite(paths, options, emitter)
        if cancelled:
            return 130  # SIGINT: run was finalized as incomplete; the caller maps 130
    except KeyboardInterrupt:  # safety net if SIGINT lands outside an in-flight case
        emitter.emit({"type": "fatal", "kind": "cancelled", "message": "interrupted"})
        return 130
    except Exception as exc:  # last-resort harness failure
        emitter.emit(
            {
                "type": "fatal",
                "kind": "harness_error",
                "message": _fmt(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        _flush_spans()
    return 0


if __name__ == "__main__":
    sys.exit(main())
