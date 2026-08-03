"""Runner/event correctness."""

import json
import os
from pathlib import Path

import pytest

import traceroot
from traceroot.eval import Dataset
from traceroot.eval.evaluation import Evaluation
from traceroot.eval.runner import Emitter, _subset, run_suite


@pytest.fixture(autouse=True)
def _isolate():
    yield
    os.environ.pop("TRACEROOT_ENABLED", None)
    try:
        traceroot.shutdown()
    except Exception:
        pass
    traceroot._client = None


def _collect(paths, options):
    events = []
    run_suite([str(p) for p in paths], options, Emitter(lambda ln: events.append(json.loads(ln))))
    return events


CONC_EVAL = """
from traceroot import Dataset, Evaluation
ds = Dataset("d")
for i in range(3):
    ds.add(input=i, id=f"c{i}", expected=i)
def task(x): return x
def acc(ctx): return 1.0
# The definition's own concurrency must be honored when the option is omitted.
conc = Evaluation(name="conc", dataset=ds, task=task, scorers=[acc], max_concurrency=3)
"""

SIMPLE_EVAL = """
from traceroot import Dataset, Evaluation
ds = Dataset("d")
ds.dataset_id = "ds_remote"
ds.dataset_version_id = "dsv_5"
for i in range(4):
    ds.add(input=i, id=f"c{i}", expected=i)
def task(x): return x
def acc(ctx): return 1.0
simple = Evaluation(name="routing", dataset=ds, task=task, scorers=[acc])
"""


def _write(tmp_path, name, body):
    d = tmp_path / "evals"
    d.mkdir(exist_ok=True)
    (d / name).write_text(body)
    return d


class TestSeededSamplingReproducible:
    def test_same_seed_selects_same_content_across_fresh_ids(self):
        # Two independent imports mint fresh ULIDs per case; seeded sampling must still
        # select the same CASES (by stable order/position), not different ones.
        def mk():
            ds = Dataset("d")
            for i in range(10):
                ds.add(input=i)  # auto ULID id, different each construction
            return ds

        ds1, ds2 = mk(), mk()
        assert [c.id for c in ds1] != [c.id for c in ds2]  # ids really differ
        c1, _, _ = _subset(ds1, None, 3, 42)
        c2, _, _ = _subset(ds2, None, 3, 42)
        inputs1 = sorted(c.input for c in ds1 if c.id in c1)
        inputs2 = sorted(c.input for c in ds2 if c.id in c2)
        assert len(inputs1) == 3
        assert inputs1 == inputs2  # same content selected despite different ids


class TestEventEnrichment:
    def test_started_event_reports_full_case_count_and_identity(self, tmp_path):
        d = _write(tmp_path, "simple_eval.py", SIMPLE_EVAL)
        events = _collect([d], {"reporting": True, "no_artifact": True})
        started = next(e for e in events if e["type"] == "evaluation_started")
        assert started["dataset"]["case_count"] == 4  # full run count, not None
        assert started["dataset"]["dataset_id"] == "ds_remote"
        assert started["dataset"]["dataset_version_id"] == "dsv_5"
        assert "created_at" in started

    def test_completed_event_has_run_id_and_created_at(self, tmp_path):
        d = _write(tmp_path, "simple_eval.py", SIMPLE_EVAL)
        events = _collect([d], {"reporting": True, "no_artifact": True})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        assert done["local_run_id"].startswith("run_")  # ULID-based; created_at derived from it
        assert "created_at" in done
        assert done["dataset"]["dataset_id"] == "ds_remote"

    def test_artifact_has_created_at(self, tmp_path):
        d = _write(tmp_path, "simple_eval.py", SIMPLE_EVAL)
        out = tmp_path / "runs"
        events = _collect([d], {"reporting": True, "out_dir": str(out)})
        done = next(e for e in events if e["type"] == "evaluation_completed")
        run_doc = json.loads(Path(done["artifact"]["run"]).read_text())
        assert "created_at" in run_doc
        assert run_doc["local_run_id"].startswith("run_")


class TestConcurrencyOverride:
    def test_definition_concurrency_honored_when_option_absent(self, tmp_path, monkeypatch):
        captured = []
        orig = Evaluation.run
        monkeypatch.setattr(
            Evaluation, "run", lambda self, **kw: captured.append(kw) or orig(self, **kw)
        )
        d = _write(tmp_path, "conc_eval.py", CONC_EVAL)

        _collect([d], {"reporting": True, "no_artifact": True})  # NO max_concurrency option
        assert "max_concurrency" not in captured[0]  # definition's value (3) stands

        captured.clear()
        _collect([d], {"reporting": True, "no_artifact": True, "max_concurrency": 7})
        assert captured[0]["max_concurrency"] == 7  # explicit override applied


class TestFatalExitZero:
    def test_fatal_event_and_exit_zero(self, tmp_path, monkeypatch):
        bad = tmp_path / "evals"
        bad.mkdir()
        (bad / "boom.py").write_text("import nonexistent_module_xyz\n")
        event_file = tmp_path / "events.ndjson"
        monkeypatch.setenv("TRACEROOT_EVAL_EVENT_FILE", str(event_file))
        monkeypatch.setenv("TRACEROOT_EVAL_OPTIONS", json.dumps({"no_artifact": True}))

        from traceroot.eval import runner

        code = runner.main([str(bad)])
        assert code == 0  # fatal -> exit 0; the event stream is authoritative
        events = [json.loads(x) for x in event_file.read_text().splitlines()]
        fatal = next(e for e in events if e["type"] == "fatal")
        assert fatal["kind"] == "import_error"
