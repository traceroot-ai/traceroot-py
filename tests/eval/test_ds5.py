"""DS-5: deferred/human scores. (Comparison is the backend's job.)"""

from traceroot.eval import (
    Dataset,
    DeferredScore,
    evaluate,
)


def _ds(extra=False):
    ds = Dataset("d")
    ds.add(input=1, id="a", expected=1)
    ds.add(input=2, id="b", expected=2)
    if extra:
        ds.add(input=3, id="c", expected=3)
    return ds


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
