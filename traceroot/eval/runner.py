"""Stable runner entry point for the TraceRoot CLI (Phase 0).

Invoked out-of-process by the CLI as::

    <interpreter> -m traceroot.eval.runner <eval paths...>

It imports each eval module, inspects module-level ``Evaluation`` instances (no
registry, no import-side-effect execution), runs them via the user's own installed
SDK, and streams NDJSON events on fd 3 (``TRACEROOT_EVAL_EVENT_FD``) or an event
file (``TRACEROOT_EVAL_EVENT_FILE``). Structured options come from the JSON env var
``TRACEROOT_EVAL_OPTIONS``; eval paths come from argv.

The runner ALWAYS exits 0 when the harness itself functioned - pass/fail is carried
in the events; the CLI decides the process exit code. It emits ``fatal`` before
dying whenever it can.

Local-only by default: unless ``reporting`` is explicitly enabled, the runner
disables TraceRoot span export and never uses a reporting transport, so a run makes
no TraceRoot reporting/dataset-sync/telemetry-export/platform request - even if the
evaluated app called ``traceroot.initialize()``. (User task/scorer code may still
call LLM providers, tools, or other services; the runner neither intercepts nor
restricts that.)

See ``offline-eval/cli-architecture-2026-07-20.md`` (SDK agent handoff).
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
import threading
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import traceroot
import traceroot.eval as ev
from traceroot.constants import SDK_VERSION
from traceroot.eval.evaluation import Evaluation
from traceroot.eval.results import EvalItemResult, EvalRunResult, case_status
from traceroot.eval.types import Dataset, DatasetSnapshot, EvalCase
from traceroot.utils import serialize_value

_DEFAULT_RUN_DIR = ".traceroot/eval/runs"


# =============================================================================
# Event emitter (NDJSON, thread-safe)
# =============================================================================
class Emitter:
    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, default=str, ensure_ascii=False) + "\n"
        with self._lock:
            self._write(line)


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
    spec.loader.exec_module(module)
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
def _case_ids(data: Any) -> list[str]:
    if isinstance(data, DatasetSnapshot):
        return [c.id for c in data.cases]  # type: ignore[misc]
    if isinstance(data, Dataset):
        return [c.id for c in data]  # active cases
    ids: list[str] = []
    for i, item in enumerate(data):
        if isinstance(item, EvalCase):
            ids.append(item.id or f"case-{i}")
        elif isinstance(item, dict):
            ids.append(item.get("id") or f"case-{i}")
    return ids


def _subset(
    data: Any, first: int | None, sample: int | None, seed: int
) -> tuple[set[str] | None, str, bool]:
    """Return (chosen_ids | None, run_mode, is_final)."""
    if first is None and sample is None:
        return None, "full", True
    ids = _case_ids(data)
    if first is not None:
        return set(ids[:first]), "first", False
    n = min(sample, len(ids)) if sample is not None else len(ids)
    chosen = random.Random(seed).sample(ids, n) if ids else []
    return set(chosen), "sample", False


# =============================================================================
# Event / artifact shaping
# =============================================================================
def _score_passed(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 1.0
    return None


def _score_event(score: Any) -> dict[str, Any]:
    return {
        "scorer_name": score.name,
        "scorer_version": score.version,
        "value": score.value,
        "passed": _score_passed(score.value),
        "explanation": score.comment,
    }


def _scorer_error_events(item: EvalItemResult) -> list[dict[str, Any]]:
    return [
        {"scorer_name": name, "scorer_version": "1", "message": msg}
        for name, msg in item.scorer_errors.items()
    ]


def _case_metadata(item: EvalItemResult) -> dict[str, Any]:
    return {
        "case_id": item.case_id,
        "status": case_status(item),
        "scores": [_score_event(s) for s in item.scores],
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
        "scored": result.scored_count,
        "passed": result.passed,
        "failed": result.failed,
        "task_errors": result.task_error_count,
        "scorer_errors": result.scorer_error_count,
        "not_scored": result.not_scored,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
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


def _truncate(value: Any, limit: int | None) -> tuple[Any, bool]:
    if limit is None:
        return value, False
    text = json.dumps(serialize_value(value), ensure_ascii=False)
    if len(text.encode()) <= limit:
        return value, False
    return {"truncated": True, "preview": text[:limit]}, True


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
    provenance: dict[str, Any] | None,
    baseline: Any = None,
    max_payload_bytes: int | None = None,
) -> dict[str, Any]:
    """Write the two-file artifact. Returns the artifact descriptor."""
    truncated_any = False
    case_lines: list[str] = []
    for item in result.item_results:
        inp, t1 = _truncate(item.input, max_payload_bytes)
        out, t2 = _truncate(item.output, max_payload_bytes)
        exp, t3 = _truncate(item.expected, max_payload_bytes)
        truncated_any = truncated_any or t1 or t2 or t3
        record = {
            "schema_version": "1",
            "case_id": item.case_id,
            "status": case_status(item),
            "input": inp,
            "output": out,
            "expected": exp,
            "scores": [_score_event(s) for s in item.scores],
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
        "evaluation_name": result.name,
        "status": status,
        "candidate_version": candidate_version,
        "run_mode": run_mode,
        "is_final": is_final,
        "sample": {"count": sample_count, "seed": sample_seed},
        "provenance": provenance,
        "dataset": result.dataset.to_dict() if result.dataset else None,
        "scorers": [{"name": n, "version": "1"} for n in result.score_summary],
        "counts": _counts(result),
        "scores": {k: v.to_dict() for k, v in result.score_summary.items()},
        "upload": result.upload_state.to_dict(),
        "artifact": artifact,
        "cases": [_case_metadata(it) for it in result.item_results],
    }
    if baseline is not None:
        cmp = result.compare(baseline)
        run_doc["baseline"] = {
            "compatible": cmp.compatible,
            "improvements": len(cmp.improvements),
            "regressions": len(cmp.regressions),
            "unchanged": len(cmp.unchanged),
            "unpaired": len(cmp.unpaired),
        }
    _atomic_write(run_path, json.dumps(serialize_value(run_doc), ensure_ascii=False, indent=2))
    return artifact


# =============================================================================
# Suite execution
# =============================================================================
def run_suite(paths: list[str], options: dict[str, Any], emitter: Emitter) -> None:
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
        return

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
        return

    completed = 0
    for _module_path, evaluation in discovered:
        _run_one(evaluation, options, emitter)
        completed += 1

    emitter.emit({"type": "suite_completed", "evaluations": completed})


def _run_one(evaluation: Evaluation, options: dict[str, Any], emitter: Emitter) -> None:
    first = options.get("first")
    sample = options.get("sample")
    seed = int(options.get("sample_seed", 0) or 0)
    candidate_version = options.get("candidate_version")
    provenance = options.get("provenance")

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

    baseline = None
    baseline_path = options.get("baseline")
    if baseline_path:
        baseline = EvalRunResult.load(baseline_path)

    emitter.emit(
        {
            "type": "evaluation_started",
            "name": evaluation.name,
            "dataset": {"case_count": len(chosen) if chosen is not None else None},
            "candidate_version": candidate_version,
            "run_mode": run_mode,
            "is_final": is_final,
        }
    )

    def on_start(case: EvalCase) -> None:
        emitter.emit({"type": "case_started", "case_id": case.id})

    def on_complete(item: EvalItemResult, _dur: float) -> None:
        emitter.emit({"type": "case_completed", **_case_metadata(item)})

    result = evaluation.run(
        candidate_version=candidate_version,
        select=select,
        max_concurrency=int(options["max_concurrency"]) if options.get("max_concurrency") else 10,
        timeout=options.get("timeout"),
        on_case_start=on_start,
        on_case_complete=on_complete,
    )

    status = _run_status(result, cancelled=False)
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
            provenance=provenance,
            baseline=baseline,
            max_payload_bytes=options.get("max_payload_bytes"),
        )

    emitter.emit(
        {
            "type": "evaluation_completed",
            "name": evaluation.name,
            "status": status,
            "counts": _counts(result),
            "score_summary": {k: v.to_dict() for k, v in result.score_summary.items()},
            "artifact": artifact,
        }
    )


def _artifact_paths(options: dict[str, Any], local_run_id: str) -> tuple[Path, Path]:
    if options.get("out_run") and options.get("out_cases"):
        return Path(options["out_run"]), Path(options["out_cases"])
    out_dir = Path(options.get("out_dir") or _DEFAULT_RUN_DIR)
    return out_dir / f"{local_run_id}.json", out_dir / f"{local_run_id}.cases.jsonl"


def _fmt(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def main(argv: list[str] | None = None) -> int:
    paths = list(argv if argv is not None else sys.argv[1:])
    emitter = Emitter(_open_channel())
    options = _load_options()
    try:
        run_suite(paths, options, emitter)
    except KeyboardInterrupt:  # handled explicitly in Phase 0-C; safety net here
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
