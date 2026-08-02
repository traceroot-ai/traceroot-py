"""Phase 0: the traceroot.eval.runner module (discovery, events, artifacts, local-only)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import traceroot
from traceroot.eval.runner import Emitter, discover, run_suite

# ---------------------------------------------------------------------------
# Eval fixtures written to disk (real modules the runner imports)
# ---------------------------------------------------------------------------
SUCCESS_EVAL = """
from traceroot import Dataset, Evaluation
from traceroot.eval import FakeTransport

def route(x):
    return {"r": "billing" if "charge" in x["m"] else "general"}

def acc(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0

ds = Dataset("tickets")
ds.add(input={"m": "charge"}, id="a", expected={"r": "billing"})
ds.add(input={"m": "hello"}, id="b", expected={"r": "general"})

# Cloud-only: a run reports through a transport. In this test the eval file configures a
# non-network FakeTransport (a real eval would report to the platform with credentials).
billing_eval = Evaluation(
    name="billing-routing", dataset=ds, task=route, scorers=[acc], report_to=FakeTransport()
)
"""

MIXED_EVAL = """
from traceroot import Dataset, Evaluation, Score

def task(x):
    if x == "boom":
        raise ValueError("kaboom")
    return x

def good(ctx):
    return 1.0

def bad(ctx):
    raise RuntimeError("scorer down")

ds = Dataset("d")
ds.add(input="ok", id="ok", expected="ok")
ds.add(input="boom", id="err", expected="boom")

mixed_eval = Evaluation(name="mixed", dataset=ds, task=task, scorers=[good, bad], main_score="good")
"""

INSTRUMENTED_EVAL = """
import traceroot
# The evaluated app already configured TraceRoot instrumentation:
traceroot.initialize(api_key="tr-fake-key", host_url="http://127.0.0.1:9")

from traceroot import Dataset, Evaluation

def echo(x):
    return x

def acc(ctx):
    return 1.0

ds = Dataset("d")
ds.add(input=1, id="a", expected=1)

inst_eval = Evaluation(name="instrumented", dataset=ds, task=echo, scorers=[acc])
"""


CANCEL_EVAL = """
from traceroot import Dataset, Evaluation

_n = {"i": 0}

async def task(x):
    _n["i"] += 1
    if _n["i"] == 2:      # simulate a SIGINT arriving after the first case finished
        raise KeyboardInterrupt()
    return x

def acc(ctx):
    return 1.0

ds = Dataset("d")
ds.add(input=1, id="c0", expected=1)
ds.add(input=2, id="c1", expected=2)
ds.add(input=3, id="c2", expected=3)

cancel_eval = Evaluation(name="cancel", dataset=ds, task=task, scorers=[acc])
"""


@pytest.fixture(autouse=True)
def _isolate():
    yield
    os.environ.pop("TRACEROOT_ENABLED", None)
    try:
        traceroot.shutdown()
    except Exception:
        pass
    traceroot._client = None


def _write_eval(tmp_path, name, body):
    d = tmp_path / "evals"
    d.mkdir(exist_ok=True)
    (d / name).write_text(body)
    return d


def _collect(paths, options):
    events = []
    run_suite(
        [str(p) for p in paths], options, Emitter(lambda line: events.append(json.loads(line)))
    )
    return events


# ---------------------------------------------------------------------------
class TestDiscovery:
    def test_finds_module_level_evaluations(self, tmp_path):
        d = _write_eval(tmp_path, "billing_eval.py", SUCCESS_EVAL)
        found = discover([str(d)])
        assert [e.name for _, e in found] == ["billing-routing"]

    def test_list_mode_emits_definitions_without_running(self, tmp_path):
        d = _write_eval(tmp_path, "billing_eval.py", SUCCESS_EVAL)
        events = _collect([d], {"mode": "list", "reporting": True})
        types = [e["type"] for e in events]
        assert types == ["hello", "definitions"]
        assert events[1]["evaluations"][0]["name"] == "billing-routing"

    def test_no_definitions_still_terminates(self, tmp_path):
        d = tmp_path / "evals"
        d.mkdir()
        (d / "empty.py").write_text("x = 1\n")
        events = _collect([d], {"reporting": True})
        assert events[0]["type"] == "hello"
        assert events[-1] == {"type": "suite_completed", "evaluations": 0, "cancelled": False}


class TestHandshakeEvent:
    def test_hello_is_first(self, tmp_path):
        d = _write_eval(tmp_path, "billing_eval.py", SUCCESS_EVAL)
        events = _collect([d], {"reporting": True})
        hello = events[0]
        assert hello["type"] == "hello"
        assert hello["eval_api_version"] == 1
        assert hello["capabilities"]["cancellation"] is True
        assert hello["sdk_version"]


class TestSuccessStream:
    def test_full_event_sequence(self, tmp_path):
        d = _write_eval(tmp_path, "billing_eval.py", SUCCESS_EVAL)
        events = _collect([d], {"reporting": True, "no_artifact": True})
        types = [e["type"] for e in events]
        assert types[0] == "hello"
        assert "evaluation_started" in types
        assert types.count("case_started") == 2
        assert types.count("case_completed") == 2
        assert types.count("evaluation_completed") == 1
        assert types[-1] == "suite_completed"
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["status"] == "completed"
        assert done["counts"]["passed"] == 2


class TestIndependentErrorFields:
    def test_task_failure_and_scorer_failure_are_independent(self, tmp_path):
        d = _write_eval(tmp_path, "mixed_eval.py", MIXED_EVAL)
        events = _collect([d], {"reporting": True, "no_artifact": True})
        cases = {e["case_id"]: e for e in events if e["type"] == "case_completed"}

        # Task failure: task_error set, scores empty (scorers skipped), status errored.
        err = cases["err"]
        assert err["status"] == "errored"
        assert "kaboom" in err["task_error"]
        assert err["scores"] == []

        # Partial scorer failure: successful score PRESERVED alongside the scorer error.
        ok = cases["ok"]
        assert ok["task_error"] is None
        assert [s["scorer_name"] for s in ok["scores"]] == ["good"]  # survived
        assert [se["scorer_name"] for se in ok["scorer_errors"]] == ["bad"]
        assert ok["scores"][0]["value"] == 1.0  # a scorer error never erased this

    def test_completed_with_errors_status(self, tmp_path):
        d = _write_eval(tmp_path, "mixed_eval.py", MIXED_EVAL)
        events = _collect([d], {"reporting": True, "no_artifact": True})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["status"] == "completed_with_errors"
        assert done["counts"]["task_errors"] == 1
        assert done["counts"]["scorer_errors"] == 1


class TestArtifacts:
    def test_two_file_artifact(self, tmp_path):
        d = _write_eval(tmp_path, "billing_eval.py", SUCCESS_EVAL)
        out = tmp_path / "runs"
        events = _collect([d], {"reporting": True, "out_dir": str(out)})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        run_path = done["artifact"]["run"]
        cases_path = done["artifact"]["cases"]
        assert run_path.endswith(".json") and cases_path.endswith(".cases.jsonl")

        run_doc = json.loads(Path(run_path).read_text())
        assert next(iter(run_doc)) == "schema_version"
        assert run_doc["counts"]["cases"] == 2
        # run.json holds per-case METADATA, not full input payloads
        assert "input" not in run_doc["cases"][0]

        case_lines = [json.loads(x) for x in Path(cases_path).read_text().splitlines()]
        assert len(case_lines) == 2
        assert "input" in case_lines[0]  # full payloads live here
        assert case_lines[0]["schema_version"] == "1"


class TestLocalOnlyNoNetwork:
    def test_instrumented_app_sends_nothing_to_traceroot(self, tmp_path, monkeypatch):
        # Spy on the reporting/dataset HTTP seam - it must never be called.
        calls = []
        monkeypatch.setattr(
            "traceroot.eval.platform._http_json",
            lambda *a, **k: calls.append(a) or {},
        )
        d = _write_eval(tmp_path, "inst_eval.py", INSTRUMENTED_EVAL)

        events = _collect([d], {})  # reporting OFF (default) -> local-only enforced

        # 1. No reporting/dataset-sync HTTP happened.
        assert calls == []
        # 2. Span export is disabled: the app's initialize() produced a DISABLED client,
        #    so no telemetry is exported to TraceRoot (trace_id is None).
        client = traceroot.get_client()
        assert client is not None and client.enabled is False
        case = next(e for e in events if e["type"] == "case_completed")
        assert case["trace_id"] is None
        # 3. The run still executed (not a silent no-op).
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["counts"]["cases"] == 1


class TestInterrupt:
    def test_partial_run_is_finalized_and_persisted(self, tmp_path):
        # A task that raises KeyboardInterrupt mid-suite = a real SIGINT arriving
        # while a later case runs. max_concurrency=1 makes ordering deterministic.
        d = _write_eval(tmp_path, "cancel_eval.py", CANCEL_EVAL)
        out = tmp_path / "runs"
        events = []
        cancelled = run_suite(
            [str(d)],
            {"reporting": True, "out_dir": str(out), "max_concurrency": 1},
            Emitter(lambda line: events.append(json.loads(line))),
        )

        # The suite reports cancellation (drives the CLI's 130 exit).
        assert cancelled is True
        suite = events[-1]
        assert suite == {"type": "suite_completed", "evaluations": 1, "cancelled": True}

        # The interrupted evaluation still emits a terminal event, marked incomplete.
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["status"] == "incomplete"

        # Cases that finished BEFORE the interrupt are preserved (partial, not empty).
        completed_ids = [e["case_id"] for e in events if e["type"] == "case_completed"]
        assert completed_ids == ["c0"]  # c0 finished; c1 raised; c2 never ran
        assert done["counts"]["cases"] == 1

        # Partial artifacts are written to disk (both files), flagged non-final.
        run_doc = json.loads(Path(done["artifact"]["run"]).read_text())
        assert run_doc["is_final"] is False
        assert run_doc["status"] == "incomplete"
        assert run_doc["counts"]["cases"] == 1
        case_lines = Path(done["artifact"]["cases"]).read_text().splitlines()
        assert len(case_lines) == 1
        assert json.loads(case_lines[0])["case_id"] == "c0"

    def test_main_exits_130_on_cancel(self, tmp_path, monkeypatch):
        d = _write_eval(tmp_path, "cancel_eval.py", CANCEL_EVAL)
        event_file = tmp_path / "events.ndjson"
        out = tmp_path / "runs"
        monkeypatch.setenv("TRACEROOT_EVAL_EVENT_FILE", str(event_file))
        monkeypatch.setenv(
            "TRACEROOT_EVAL_OPTIONS",
            json.dumps({"out_dir": str(out), "max_concurrency": 1}),
        )
        from traceroot.eval import runner

        code = runner.main([str(d)])
        assert code == 130  # SIGINT exit convention, chosen by the runner->CLI contract

        events = [json.loads(x) for x in event_file.read_text().splitlines()]
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["status"] == "incomplete"


class TestModuleInvocation:
    def test_python_m_runner_over_event_file(self, tmp_path):
        d = _write_eval(tmp_path, "billing_eval.py", SUCCESS_EVAL)
        event_file = tmp_path / "events.ndjson"
        env = {k: v for k, v in os.environ.items() if not k.startswith("TRACEROOT_")}
        env["TRACEROOT_EVAL_EVENT_FILE"] = str(event_file)
        env["TRACEROOT_EVAL_OPTIONS"] = json.dumps({"no_artifact": True})
        proc = subprocess.run(
            [sys.executable, "-m", "traceroot.eval.runner", str(d)],
            env=env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        events = [json.loads(x) for x in event_file.read_text().splitlines()]
        types = [e["type"] for e in events]
        assert types[0] == "hello"
        assert types[-1] == "suite_completed"
        assert "case_completed" in types
