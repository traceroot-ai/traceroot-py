"""DS-5: run-level scorers, deferred/human scores. (Comparison is the backend's job.)"""

from traceroot.eval import (
    Dataset,
    DeferredScore,
    Score,
    ScorerContext,
    evaluate,
)


def _ds(extra=False):
    ds = Dataset("d")
    ds.add(input=1, id="a", expected=1)
    ds.add(input=2, id="b", expected=2)
    if extra:
        ds.add(input=3, id="c", expected=3)
    return ds


def acc(ctx: ScorerContext):
    return 1.0 if ctx.output == ctx.expected else 0.0


def good(x):
    return x


class TestDeferredScore:
    def test_deferred_is_pending_not_zero(self):
        def human(ctx):
            return DeferredScore("human_quality", reason="needs review")

        run = evaluate(name="q", dataset=_ds(), task=good, scorers=[human])
        item = run.item_results[0]
        pending = [s for s in item.scores if s.metadata and s.metadata.get("deferred")]
        assert len(pending) == 1
        assert pending[0].value != 0  # never coerced to zero
        # a deferred-only case is not_scored, and never errored
        assert run.not_scored == len(run.item_results)
        assert run.errored == 0


class TestRunScorers:
    def test_run_level_scorer_produces_run_score(self):
        def pass_rate(view):
            passed = sum(1 for it in view.item_results if it.scores and it.scores[0].value == 1.0)
            return Score("pass_rate", passed / len(view.item_results))

        run = evaluate(name="q", dataset=_ds(), task=good, scorers=[acc], run_scorers=[pass_rate])
        names = {s.name for s in run.run_scores}
        assert "pass_rate" in names
        assert next(s for s in run.run_scores if s.name == "pass_rate").value == 1.0

    def test_run_scorer_error_isolated(self):
        def boom(view):
            raise ValueError("nope")

        run = evaluate(name="q", dataset=_ds(), task=good, scorers=[acc], run_scorers=[boom])
        assert "boom" in run.run_scorer_errors
