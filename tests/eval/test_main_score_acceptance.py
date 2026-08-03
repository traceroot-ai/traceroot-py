"""B1 acceptance matrix, driven by the SHARED fixture (identical copy in traceroot-ts). The
single scorer's declared threshold + direction apply to its emitted metric, one policy across
local + cloud, with an honest terminal failure (no orphaned run) on ambiguity."""

import json
import pathlib

import pytest

from traceroot.eval import evaluate
from traceroot.eval.results import case_status
from traceroot.eval.scorers import scorer
from traceroot.eval.transport import FakeTransport
from traceroot.eval.types import Score

_FIX = json.loads(
    (pathlib.Path(__file__).parent / "fixtures" / "main_score_acceptance.json").read_text()
)


def _build_scorer(spec):
    if "emit_scalar" in spec:

        def fn(ctx, _v=spec["emit_scalar"]):
            return _v

    elif "emit_multi" in spec:

        def fn(ctx, _m=spec["emit_multi"]):
            return [Score(s["name"], s["value"]) for s in _m]

    elif "emit_by_case" in spec:

        def fn(ctx, _by=spec["emit_by_case"]):
            s = _by[ctx.input["i"]]
            return Score(s["name"], s["value"])

    else:

        def fn(ctx, _e=spec["emit"]):
            return Score(_e["name"], _e["value"])

    fn.__name__ = spec["fn_name"]
    return scorer(
        fn, value_type="numeric", threshold=spec["threshold"], direction=spec["direction"]
    )


def _finish(fake):
    return [(c[0], c[1]) for c in fake.calls if c[0] == "finish_run"]


@pytest.mark.parametrize("case", _FIX["main_score_acceptance"], ids=lambda c: c["name"])
def test_main_score_acceptance(case):
    scorers = [_build_scorer(s) for s in case["scorers"]]
    dataset = [{"input": {"i": i}} for i in range(case["cases"])]
    fake = FakeTransport()
    exp = case["expect"]

    def run():
        return evaluate(
            name="acc",
            dataset=dataset,
            task=lambda x: x,
            scorers=scorers,
            main_score=case["main_score"],
            report_to=fake,
        )

    if exp.get("error"):
        with pytest.raises(ValueError):
            run()
        if exp.get("terminal_failed"):
            # completed terminally as 'failed' before raising -- never orphaned in 'running'.
            assert ("finish_run", "failed") in _finish(fake)
        return

    result = run()
    assert result.main_score_name == exp["main_score_name"]
    # Completion payload carries the resolved name (acceptance #8).
    assert fake.last_main_score_name == exp["main_score_name"]
    # The one resolved policy drives every per-case status (local == cloud).
    for item in result.item_results:
        assert case_status(item, result.main_score) == exp["status"]
