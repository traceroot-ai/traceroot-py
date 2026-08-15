"""One canonical run-artifact reader. EvalRunResult.load understands BOTH the
save() shape and the runner's run.json, so a loaded run reconstructs cases + scores."""

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
