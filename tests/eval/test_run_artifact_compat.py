"""One canonical run-artifact reader. EvalRunResult.load understands BOTH the
save() shape and the runner's run.json, so a loaded run reconstructs cases + scores."""

import json

from traceroot.eval import Dataset, EvalCase, EvalRunResult, evaluate
from traceroot.eval.runner import write_artifacts


def _ds(n=3):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
    return ds


def echo(x):
    return x


def acc(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


def _write_runner_artifact(result, tmp_path):
    run_p = tmp_path / "run_x.json"
    cases_p = tmp_path / "run_x.cases.jsonl"
    write_artifacts(
        result,
        run_p,
        cases_p,
        status="completed",
        run_mode="full",
        is_final=True,
        sample_count=None,
        sample_seed=None,
        candidate_version=None,
    )
    return run_p


class TestRunArtifactReader:
    def test_native_save_load_round_trip(self, tmp_path):
        run = evaluate(name="r", dataset=_ds(2), task=echo, scorers=[acc])
        p = tmp_path / "native.json"
        run.save(str(p))
        loaded = EvalRunResult.load(str(p))
        assert loaded.to_dict() == run.to_dict()  # native shape still round-trips

    def test_load_runner_artifact_reconstructs_cases_and_scores(self, tmp_path):
        run = evaluate(name="billing", dataset=_ds(3), task=echo, scorers=[acc])
        run_p = _write_runner_artifact(run, tmp_path)

        loaded = EvalRunResult.load(str(run_p))  # the runner run.json, NOT save()
        assert loaded.name == "billing"
        assert [it.case_id for it in loaded.item_results] == ["c0", "c1", "c2"]
        # scores are preserved (what row-matched comparison needs)
        assert loaded.item_results[0].scores[0].name == "acc"
        assert loaded.item_results[0].scores[0].value == 1.0

    def test_runner_artifact_round_trips_the_scorer_policy(self, tmp_path):
        # A run loaded from run.json is re-uploadable; without the retained policy the re-upload
        # re-registers thresholds-less and its per-score `passed` disagrees with the original run.
        specs = [{"name": "acc", "value_type": "numeric", "threshold": 0.8}]
        run = evaluate(name="billing", dataset=_ds(2), task=echo, scorers=[acc])
        run.scorer_specs = specs
        run_p = _write_runner_artifact(run, tmp_path)

        assert EvalRunResult.load(str(run_p)).scorer_specs == specs


class TestUploadBlock:
    """The ``upload`` block loads whichever counts it carries: an older run file has fewer keys,
    and one written by a newer SDK may have a key this version doesn't know."""

    def _shapes(self, upload, tmp_path):
        run = evaluate(name="r", dataset=_ds(1), task=echo, scorers=[acc])
        native = run.to_dict()
        runner = json.loads(_write_runner_artifact(run, tmp_path).read_text())
        native["upload"] = runner["upload"] = upload
        return {"save": native, "runner": runner}

    def test_the_counts_round_trip_through_both_shapes(self, tmp_path):
        upload = {
            "status": "uploaded",
            "dashboard_url": None,
            "failed_result_count": 2,
            "unconfirmed_result_count": 1,
        }
        for shape, d in self._shapes(upload, tmp_path).items():
            state = EvalRunResult.from_dict(d).upload_state
            assert state.failed_result_count == 2, shape
            assert state.unconfirmed_result_count == 1, shape
            assert state.to_dict() == upload, shape

    def test_an_older_upload_block_reads_zero_counts(self, tmp_path):
        upload = {"status": "uploaded", "dashboard_url": None}
        for shape, d in self._shapes(upload, tmp_path).items():
            state = EvalRunResult.from_dict(d).upload_state
            assert state.failed_result_count == 0, shape
            assert state.unconfirmed_result_count == 0, shape

    def test_an_unknown_upload_key_does_not_fail_the_load(self, tmp_path):
        upload = {
            "status": "uploaded",
            "dashboard_url": "https://app.example/run_1",
            "failed_result_count": 1,
            "a_count_from_a_newer_sdk": 3,
        }
        for shape, d in self._shapes(upload, tmp_path).items():
            state = EvalRunResult.from_dict(d).upload_state
            assert state.dashboard_url == "https://app.example/run_1", shape
            assert state.failed_result_count == 1, shape
