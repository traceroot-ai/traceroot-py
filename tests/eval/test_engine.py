"""OE-3: execution kernel - evaluate/evaluate_async, concurrency, ordering, isolation."""

import asyncio

import pytest

from traceroot.eval import Dataset, EvalCase, EvalRunResult, Score, evaluate, evaluate_async


def _ds(n):
    ds = Dataset(name="d")
    for i in range(n):
        ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
    return ds


def echo_task(x):
    return x


def exact(ctx):
    return 1.0 if ctx.output == ctx.expected else 0.0


class TestBasicRuns:
    def test_sync_task_sync_scorer(self):
        result = evaluate(name="r", dataset=_ds(3), task=echo_task, scorers=[exact])
        assert isinstance(result, EvalRunResult)
        assert [it.case_id for it in result.item_results] == ["c0", "c1", "c2"]
        assert result.score_summary["exact"].mean == 1.0
        assert result.score_summary["exact"].count == 3

    async def test_async_task_async_scorer(self):
        async def atask(x):
            await asyncio.sleep(0)
            return x

        async def ascore(ctx):
            await asyncio.sleep(0)
            return 1.0 if ctx.output == ctx.expected else 0.0

        result = await evaluate_async(name="r", dataset=_ds(2), task=atask, scorers=[ascore])
        assert result.score_summary["ascore"].mean == 1.0

    def test_mixed_sync_async(self):
        async def ascore(ctx):
            return 1.0

        result = evaluate(name="r", dataset=_ds(2), task=echo_task, scorers=[exact, ascore])
        assert result.score_summary["exact"].mean == 1.0
        assert result.score_summary["ascore"].mean == 1.0

    def test_evaluate_returns_completed_result_not_coroutine(self):
        result = evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=[exact])
        assert isinstance(result, EvalRunResult)
        assert not asyncio.iscoroutine(result)

    def test_evaluate_works_inside_running_loop(self):
        async def outer():
            return evaluate(name="r", dataset=_ds(2), task=echo_task, scorers=[exact])

        result = asyncio.run(outer())
        assert isinstance(result, EvalRunResult)
        assert len(result.item_results) == 2


class TestOrderingAndConcurrency:
    async def test_deterministic_input_ordering(self):
        # Later indices finish first, but results stay in input order.
        async def slow(x):
            await asyncio.sleep((5 - x) * 0.01)
            return x

        result = await evaluate_async(name="r", dataset=_ds(5), task=slow, scorers=[exact])
        assert [it.case_id for it in result.item_results] == ["c0", "c1", "c2", "c3", "c4"]
        assert [it.output for it in result.item_results] == [0, 1, 2, 3, 4]

    async def test_concurrency_is_bounded(self):
        state = {"cur": 0, "peak": 0}

        async def tracked(x):
            state["cur"] += 1
            state["peak"] = max(state["peak"], state["cur"])
            await asyncio.sleep(0.02)
            state["cur"] -= 1
            return x

        await evaluate_async(
            name="r", dataset=_ds(10), task=tracked, scorers=[exact], max_concurrency=2
        )
        assert state["peak"] <= 2


class TestFailureIsolation:
    def test_task_error_isolates_and_skips_scorers(self):
        def boom(x):
            if x == 1:
                raise ValueError("nope")
            return x

        result = evaluate(name="r", dataset=_ds(3), task=boom, scorers=[exact])
        by_id = {it.case_id: it for it in result.item_results}
        assert by_id["c1"].error is not None
        assert "nope" in by_id["c1"].error
        assert by_id["c1"].scores == []  # scorers skipped
        assert by_id["c0"].error is None and by_id["c0"].scores  # others fine
        assert by_id["c2"].error is None

    def test_scorer_error_isolates_siblings_still_score(self):
        def bad(ctx):
            raise RuntimeError("scorer boom")

        result = evaluate(name="r", dataset=_ds(2), task=echo_task, scorers=[exact, bad])
        it = result.item_results[0]
        assert "bad" in it.scorer_errors
        assert "scorer boom" in it.scorer_errors["bad"]
        assert any(s.name == "exact" for s in it.scores)  # sibling still scored

    def test_malformed_scorer_return_is_scorer_error_not_crash(self):
        def malformed(ctx):
            return {"value": 1.0}  # missing 'name'

        result = evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=[malformed])
        it = result.item_results[0]
        assert "malformed" in it.scorer_errors


class TestScoreNormalization:
    def _one(self, scorer):
        return evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=[scorer]).item_results[0]

    def test_scalar(self):
        def s(ctx):
            return 0.5

        scores = self._one(s).scores
        assert len(scores) == 1 and scores[0].name == "s" and scores[0].value == 0.5

    def test_bool_scalar(self):
        def s(ctx):
            return True

        assert self._one(s).scores[0].value is True

    def test_str_scalar(self):
        def s(ctx):
            return "billing"

        assert self._one(s).scores[0].value == "billing"

    def test_dict(self):
        def s(ctx):
            return {"name": "custom", "value": 0.9, "comment": "c"}

        sc = self._one(s).scores[0]
        assert sc.name == "custom" and sc.value == 0.9 and sc.comment == "c"

    def test_single_score(self):
        def s(ctx):
            return Score(name="x", value=1)

        assert self._one(s).scores[0].name == "x"

    def test_list_of_scores(self):
        def s(ctx):
            return [Score("a", 1.0), Score("b", 0.0)]

        # Two emitted metrics -> both scores are recorded (no headline metric is required).
        names = {sc.name for sc in self._one(s).scores}
        assert names == {"a", "b"}

    def test_none_abstains(self):
        def s(ctx):
            return None

        assert self._one(s).scores == []


class TestDataCoercion:
    def test_list_of_eval_cases(self):
        cases = [EvalCase(input=1, id="a", expected=1), EvalCase(input=2, id="b", expected=2)]
        result = evaluate(name="r", dataset=cases, task=echo_task, scorers=[exact])
        assert [it.case_id for it in result.item_results] == ["a", "b"]

    def test_list_of_dicts(self):
        data = [{"input": 1, "expected": 1}, {"input": 2, "expected": 2}]
        result = evaluate(name="r", dataset=data, task=echo_task, scorers=[exact])
        assert result.score_summary["exact"].mean == 1.0

    def test_dict_missing_input_raises(self):
        with pytest.raises((ValueError, TypeError)):
            evaluate(name="r", dataset=[{"expected": 1}], task=echo_task, scorers=[exact])

    def test_dict_unknown_key_raises(self):
        with pytest.raises((ValueError, TypeError)):
            evaluate(name="r", dataset=[{"input": 1, "bogus": 2}], task=echo_task, scorers=[exact])


class TestConfigErrors:
    def test_empty_name(self):
        with pytest.raises(ValueError):
            evaluate(name="", dataset=_ds(1), task=echo_task, scorers=[exact])

    def test_empty_data(self):
        with pytest.raises(ValueError):
            evaluate(name="r", dataset=_ds(0), task=echo_task, scorers=[exact])

    def test_non_callable_task(self):
        with pytest.raises((ValueError, TypeError)):
            evaluate(name="r", dataset=_ds(1), task="nope", scorers=[exact])

    def test_empty_scorers(self):
        with pytest.raises(ValueError):
            evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=[])

    def test_non_callable_scorer(self):
        with pytest.raises((ValueError, TypeError)):
            evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=["nope"])

    def test_bad_max_concurrency(self):
        with pytest.raises(ValueError):
            evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=[exact], max_concurrency=0)


class TestResultShape:
    def test_uploaded_status_and_no_trace_id_without_provider(self):
        # Cloud-only: the run reports (status uploaded). With no tracer provider registered
        # in this test, the no-op span has no valid context, so the item carries no trace id.
        result = evaluate(name="r", dataset=_ds(1), task=echo_task, scorers=[exact])
        assert result.item_results[0].trace_id is None
        assert result.upload_state.status == "uploaded"
