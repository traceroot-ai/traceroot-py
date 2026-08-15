"""OE-6: the eval public API is exported from the top-level `traceroot` namespace."""

import traceroot


def test_top_level_exports_importable():
    from traceroot import (  # noqa: F401
        Dataset,
        EvalCase,
        EvalItemResult,
        EvalRunResult,
        Score,
        ScorerContext,
        evaluate,
        evaluate_async,
    )


def test_names_in_dunder_all():
    for name in (
        "Dataset",
        "EvalCase",
        "Score",
        "ScorerContext",
        "evaluate",
        "evaluate_async",
        "EvalRunResult",
        "EvalItemResult",
    ):
        assert name in traceroot.__all__, name


def test_end_to_end_via_top_level():
    ds = traceroot.Dataset(name="billing")
    ds.upsert(traceroot.EvalCase(input={"m": "hi"}, id="c0", expected={"r": "billing"}))

    def task(x):
        return {"r": "billing"}

    def routing(ctx):
        return 1.0 if ctx.output == ctx.expected else 0.0

    result = traceroot.evaluate(name="routing-v2", data=ds, task=task, scorers=[routing])
    assert result.score_summary["routing"].mean == 1.0
    assert result.upload_state.status == "uploaded"
