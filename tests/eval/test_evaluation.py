"""DS-2: Evaluation object + evaluate() convenience + enriched run result."""

from traceroot.eval import (
    Dataset,
    EvalRunResult,
    Evaluation,
    RunDatasetRef,
    ScorerContext,
    evaluate,
)


def _ds():
    ds = Dataset("billing", description="routing")
    ds.add(input={"m": "charge"}, id="a", expected={"r": "billing"})
    ds.add(input={"m": "hello"}, id="b", expected={"r": "general"})
    return ds


def route(x):
    return {"r": "billing" if "charge" in x["m"] else "general"}


def acc(ctx: ScorerContext):
    return 1.0 if ctx.output == ctx.expected else 0.0


class TestEvaluationObject:
    def test_run_returns_result(self):
        ev = Evaluation(name="q", dataset=_ds(), task=route, scorers=[acc])
        run = ev.run()
        assert isinstance(run, EvalRunResult)
        assert run.score_summary["acc"].mean == 1.0

    async def test_run_async(self):
        ev = Evaluation(name="q", dataset=_ds(), task=route, scorers=[acc])
        run = await ev.run_async()
        assert run.score_summary["acc"].count == 2

    def test_reuse_produces_distinct_runs(self):
        ev = Evaluation(name="q", dataset=_ds(), task=route, scorers=[acc])
        r1, r2 = ev.run(), ev.run()
        assert r1.local_run_id != r2.local_run_id

    def test_candidate_version_passthrough(self):
        run = Evaluation(
            name="q", dataset=_ds(), task=route, scorers=[acc], candidate_version="git:abc"
        ).run()
        assert run.candidate_version == "git:abc"


class TestConvenience:
    def test_dataset_kwarg(self):
        run = evaluate(name="q", dataset=_ds(), task=route, scorers=[acc])
        assert run.score_summary["acc"].mean == 1.0

    def test_data_alias(self):
        run = evaluate(name="q", data=_ds(), task=route, scorers=[acc])
        assert run.score_summary["acc"].mean == 1.0

    def test_list_data(self):
        run = evaluate(
            name="q",
            dataset=[{"input": {"m": "charge"}, "expected": {"r": "billing"}}],
            task=route,
            scorers=[acc],
        )
        assert run.score_summary["acc"].mean == 1.0

    def test_snapshot_data(self):
        snap = _ds().snapshot()
        run = evaluate(name="q", dataset=snap, task=route, scorers=[acc])
        assert run.dataset.revision == snap.revision


class TestRunDatasetRef:
    def test_records_dataset_identity(self):
        ds = _ds()
        run = evaluate(name="q", dataset=ds, task=route, scorers=[acc])
        assert isinstance(run.dataset, RunDatasetRef)
        assert run.dataset.dataset_id == ds.dataset_id
        assert run.dataset.revision == ds.snapshot().revision
        assert run.dataset.case_count == 2
        assert run.local_run_id.startswith("run_")


class TestSnapshotImmutability:
    def test_mutating_dataset_after_run_does_not_change_run(self):
        ds = _ds()
        run = evaluate(name="q", dataset=ds, task=route, scorers=[acc])
        rev_before = run.dataset.revision
        outputs_before = [it.output for it in run.item_results]
        # mutate the source dataset afterward
        ds.update("a", expected={"r": "changed"})
        ds.add(input={"m": "new"}, id="c")
        ds.remove("b")
        assert run.dataset.revision == rev_before
        assert [it.output for it in run.item_results] == outputs_before
        assert run.dataset.case_count == 2


class TestSelect:
    def test_select_filters_cases(self):
        run = evaluate(
            name="q", dataset=_ds(), task=route, scorers=[acc], select=lambda c: c.id == "a"
        )
        assert run.case_count == 1
        assert run.item_results[0].case_id == "a"


class TestRunResultSurface:
    def _run(self):
        ds = Dataset("d")
        ds.add(input={"m": "charge"}, id="ok", expected={"r": "billing"})
        ds.add(input={"m": "hello"}, id="bad", expected={"r": "billing"})  # will fail

        def boom_task(x):
            if x["m"] == "boom":
                raise ValueError("x")
            return route(x)

        ds.add(input={"m": "boom"}, id="err", expected={"r": "billing"})
        return evaluate(name="q", dataset=ds, task=boom_task, scorers=[acc])

    def test_counts(self):
        run = self._run()
        assert run.case_count == 3
        assert run.passed == 1  # 'ok'
        assert run.failed == 1  # 'bad'
        assert run.task_error_count == 1  # 'err'

    def test_failures_and_errors(self):
        run = self._run()
        assert [it.case_id for it in run.failures()] == ["bad"]
        assert [it.case_id for it in run.errors()] == ["err"]

    def test_results_property_and_summary(self):
        run = self._run()
        assert run.results is run.item_results
        assert "q" in run.summary()

    def test_save_load_round_trip(self, tmp_path):
        run = self._run()
        p = tmp_path / "run.json"
        run.save(str(p))
        restored = EvalRunResult.load(str(p))
        assert restored.local_run_id == run.local_run_id
        assert restored.dataset.revision == run.dataset.revision
        assert restored.case_count == 3
