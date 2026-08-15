"""OE-6: the eval public API is exported from the top-level `traceroot` namespace."""

import traceroot

# The documented public surface. Every symbol a user is told to reference by name -- including
# the exceptions they are told to CATCH -- must be reachable as ``traceroot.<name>``; a symbol
# that only exists on ``traceroot.eval`` is not public API.
PUBLIC_SURFACE = (
    "Dataset",
    "DatasetSnapshot",
    "DatasetConflictError",
    "DatasetPublishAborted",
    "EvalCase",
    "EvalCompletionError",
    "EvalItemResult",
    "EvalRunResult",
    "Score",
    "Scorer",
    "ScorerContext",
    "case_status",
    "evaluate",
    "evaluate_async",
    "llm_judge",
    "pull_dataset",
    "pull_dataset_version",
    "scorer",
)


def test_top_level_exports_importable():
    for name in PUBLIC_SURFACE:
        assert getattr(traceroot, name, None) is not None, name


def test_names_in_dunder_all():
    for name in PUBLIC_SURFACE:
        assert name in traceroot.__all__, name


def test_dunder_all_is_fully_resolvable():
    # Nothing may be advertised in __all__ that `from traceroot import *` cannot resolve.
    for name in traceroot.__all__:
        assert hasattr(traceroot, name), name


def test_documented_exceptions_are_catchable_from_top_level():
    import traceroot.eval.dataset_sync as dataset_sync
    import traceroot.eval.transport as transport
    from traceroot import DatasetConflictError, DatasetPublishAborted, EvalCompletionError

    assert DatasetConflictError is dataset_sync.DatasetConflictError
    assert DatasetPublishAborted is dataset_sync.DatasetPublishAborted
    assert EvalCompletionError is transport.EvalCompletionError


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
