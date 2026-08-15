"""SDK prerequisites: capability handshake, per-case timing/timeout, schema_version."""

import traceroot.eval as ev
from traceroot.eval import Dataset, EvalRunResult, ScorerContext


def _ds(n=3):
    ds = Dataset("d")
    for i in range(n):
        ds.add(input=i, id=f"c{i}", expected=i)
    return ds


def echo(x):
    return x


def acc(ctx: ScorerContext):
    return 1.0 if ctx.output == ctx.expected else 0.0


class TestHandshake:
    def test_api_version_is_int(self):
        assert isinstance(ev.__api_version__, int)
        assert ev.__api_version__ >= 1

    def test_capabilities_keys(self):
        caps = ev.capabilities()
        for key in (
            "snapshot",
            "run_session",
            "compare",
            "dataset_push",
            "sampling",
            "cancellation",
        ):
            assert caps[key] is True


class TestSchemaVersion:
    def test_to_dict_first_key_is_schema_version(self):
        from traceroot.eval import evaluate

        run = evaluate(name="r", dataset=_ds(), task=echo, scorers=[acc])
        d = run.to_dict()
        assert next(iter(d)) == "schema_version"
        assert d["schema_version"] == "1"

    def test_round_trip_lossless(self, tmp_path):
        from traceroot.eval import evaluate

        run = evaluate(name="r", dataset=_ds(), task=echo, scorers=[acc])
        p = tmp_path / "run.json"
        run.save(str(p))
        restored = EvalRunResult.load(str(p))
        assert restored.to_dict() == run.to_dict()


class TestPerCaseTimingAndHooks:
    def test_engine_reports_per_case_via_hooks(self):
        import asyncio

        from traceroot.eval.engine import _run_async

        started, completed = [], []

        async def go():
            return await _run_async(
                name="r",
                data=_ds(2),
                task=echo,
                scorers=[acc],
                on_case_start=lambda case: started.append(case.id),
                on_case_complete=lambda item, dur: completed.append((item.case_id, dur)),
            )

        asyncio.run(go())
        assert sorted(started) == ["c0", "c1"]
        assert sorted(cid for cid, _ in completed) == ["c0", "c1"]
        assert all(isinstance(dur, (int, float)) and dur >= 0 for _, dur in completed)

    def test_timeout_becomes_task_error(self):
        import asyncio

        from traceroot.eval.engine import _run_async

        async def slow(x):
            await asyncio.sleep(1.0)
            return x

        async def go():
            return await _run_async(name="r", data=_ds(1), task=slow, scorers=[acc], timeout=0.05)

        run = asyncio.run(go())
        assert run.item_results[0].error is not None  # timed out -> task error, isolated
        assert run.task_error_count == 1
