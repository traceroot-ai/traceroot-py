"""Runner robustness: declared-threshold verdicts, unserializable payloads, hook isolation,
and the flush-before-exit guarantee."""

import json
import os

import pytest

import traceroot
from traceroot.eval.runner import Emitter, main, run_suite

THRESHOLD_EVAL = """
from traceroot import Dataset, Evaluation
from traceroot.eval import scorer

@scorer(value_type="numeric", threshold=0.8)
def quality(ctx):
    return 0.9

ds = Dataset("d")
ds.add(input=1, id="c0", expected=1)

threshold_eval = Evaluation(name="thresh", dataset=ds, task=lambda x: x, scorers=[quality])
"""

LOWER_IS_BETTER_EVAL = """
from traceroot import Dataset, Evaluation
from traceroot.eval import scorer

@scorer(value_type="numeric", direction="lower_is_better", threshold=0.5)
def latency(ctx):
    return 0.5 if ctx.input == "at" else 0.6

ds = Dataset("d")
ds.add(input="at", id="at", expected=None)
ds.add(input="over", id="over", expected=None)

latency_eval = Evaluation(name="lat", dataset=ds, task=lambda x: x, scorers=[latency])
"""

NO_THRESHOLD_EVAL = """
from traceroot import Dataset, Evaluation

def score(ctx):
    return 0.9

ds = Dataset("d")
ds.add(input=1, id="c0", expected=1)

undeclared_eval = Evaluation(name="undeclared", dataset=ds, task=lambda x: x, scorers=[score])
"""

CYCLIC_OUTPUT_EVAL = """
from traceroot import Dataset, Evaluation

def task(x):
    out = {"case": x}
    out["self"] = out          # a reference cycle the JSON artifact cannot represent
    return out

def acc(ctx):
    return 1.0

ds = Dataset("d")
ds.add(input=1, id="c0", expected=1)
ds.add(input=2, id="c1", expected=2)

cyclic_eval = Evaluation(name="cyclic", dataset=ds, task=task, scorers=[acc])
"""

TWO_CASE_EVAL = """
from traceroot import Dataset, Evaluation

def acc(ctx):
    return 1.0

ds = Dataset("d")
ds.add(input=1, id="c0", expected=1)
ds.add(input=2, id="c1", expected=2)

two_eval = Evaluation(name="two", dataset=ds, task=lambda x: x, scorers=[acc])
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


def _scores(events, case_id):
    case = next(e for e in events if e["type"] == "case_completed" and e["case_id"] == case_id)
    return case["scores"]


# ---------------------------------------------------------------------------
class TestDeclaredThresholdVerdict:
    """The local artifact's `passed` must come from the scorer's DECLARED policy - the same
    resolution the platform applies - or the two disagree on the same score."""

    def test_score_above_declared_threshold_passes(self, tmp_path):
        d = _write_eval(tmp_path, "thresh_eval.py", THRESHOLD_EVAL)
        events = _collect([d], {"no_artifact": True})
        # 0.9 >= declared threshold 0.8 -> pass (a hardcoded >= 1.0 would call this a failure).
        assert _scores(events, "c0")[0]["passed"] is True

    def test_lower_is_better_boundary(self, tmp_path):
        d = _write_eval(tmp_path, "lat_eval.py", LOWER_IS_BETTER_EVAL)
        events = _collect([d], {"no_artifact": True})
        assert _scores(events, "at")[0]["passed"] is True  # 0.5 <= 0.5
        assert _scores(events, "over")[0]["passed"] is False  # 0.6 > 0.5

    def test_no_declared_threshold_gets_no_verdict(self, tmp_path):
        d = _write_eval(tmp_path, "undeclared_eval.py", NO_THRESHOLD_EVAL)
        events = _collect([d], {"no_artifact": True})
        # Scored, but the SDK declares no threshold -> no fabricated pass/fail.
        assert _scores(events, "c0")[0]["passed"] is None

    def test_artifact_and_events_agree(self, tmp_path):
        d = _write_eval(tmp_path, "thresh_eval.py", THRESHOLD_EVAL)
        out = tmp_path / "runs"
        events = _collect([d], {"out_dir": str(out)})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        run_doc = json.loads((out / f"{done['local_run_id']}.json").read_text())
        assert run_doc["cases"][0]["scores"][0]["passed"] is True
        cases = out / f"{done['local_run_id']}.cases.jsonl"
        assert json.loads(cases.read_text().splitlines()[0])["scores"][0]["passed"] is True


class TestUnserializablePayload:
    def test_reference_cycle_degrades_the_payload_not_the_run(self, tmp_path):
        d = _write_eval(tmp_path, "cyclic_eval.py", CYCLIC_OUTPUT_EVAL)
        out = tmp_path / "runs"
        events = _collect([d], {"out_dir": str(out)})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["counts"]["cases"] == 2  # the whole run survived
        cases = out / f"{done['local_run_id']}.cases.jsonl"
        record = json.loads(cases.read_text().splitlines()[0])
        assert record["output"]["unserializable"] is True
        assert record["input"] == 1  # untouched payloads keep their real value

    def test_cycle_survives_payload_truncation(self, tmp_path):
        d = _write_eval(tmp_path, "cyclic_eval.py", CYCLIC_OUTPUT_EVAL)
        out = tmp_path / "runs"
        events = _collect([d], {"out_dir": str(out), "max_payload_bytes": 16})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["counts"]["cases"] == 2


class TestHookIsolation:
    def test_a_throwing_emitter_does_not_abort_the_run(self, tmp_path):
        d = _write_eval(tmp_path, "two_eval.py", TWO_CASE_EVAL)
        events = []

        def sink(line: str) -> None:
            event = json.loads(line)
            if event["type"] == "case_started":
                raise BrokenPipeError("event channel closed")
            events.append(event)

        run_suite([str(d)], {"no_artifact": True}, Emitter(sink))
        types = [e["type"] for e in events]
        assert types.count("case_completed") == 2  # every case still ran
        assert types[-1] == "suite_completed"


class TestFlushBeforeExit:
    def test_main_flushes_pending_spans_before_returning(self, tmp_path, monkeypatch):
        d = _write_eval(tmp_path, "two_eval.py", TWO_CASE_EVAL)
        flushed: list[int] = []
        events: list[dict] = []
        monkeypatch.setattr(traceroot, "flush", lambda: flushed.append(len(events)))
        monkeypatch.setenv(
            "TRACEROOT_EVAL_OPTIONS", json.dumps({"reporting": True, "no_artifact": True})
        )
        monkeypatch.setattr(
            "traceroot.eval.runner._open_channel",
            lambda: lambda line: events.append(json.loads(line)),
        )
        assert main([str(d)]) == 0
        # Flushed exactly once, AFTER the last event - no span is left unbatched at exit.
        assert flushed == [len(events)]
        assert events[-1]["type"] == "suite_completed"

    def test_flush_failure_never_fails_the_run(self, tmp_path, monkeypatch):
        d = _write_eval(tmp_path, "two_eval.py", TWO_CASE_EVAL)
        events: list[dict] = []

        def boom() -> None:
            raise RuntimeError("exporter unreachable")

        monkeypatch.setattr(traceroot, "flush", boom)
        monkeypatch.setenv(
            "TRACEROOT_EVAL_OPTIONS", json.dumps({"reporting": True, "no_artifact": True})
        )
        monkeypatch.setattr(
            "traceroot.eval.runner._open_channel",
            lambda: lambda line: events.append(json.loads(line)),
        )
        assert main([str(d)]) == 0
        assert not [e for e in events if e["type"] == "fatal"]
