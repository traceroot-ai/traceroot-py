"""Phase 1 — progressive scorer API: ordinary functions + simple returns.

A scorer may be an ordinary function taking (input, output, expected=None, metadata=None) OR the
existing single-ScorerContext form. Returns may be a scalar, a {name,value} object, a metric->value
map, a Score, a Score[], or None. Everything normalizes to the same internal Score list.
"""

from traceroot.eval import Dataset, Score, evaluate
from traceroot.eval.transport import FakeTransport


def _ds():
    ds = Dataset("d")
    ds.add(input={"q": "hi"}, id="c0", expected="hi")
    return ds


def _run(scorers, main_score=None):
    return evaluate(
        name="r", dataset=_ds(), task=lambda x: "hi", scorers=scorers,
        main_score=main_score, transport=FakeTransport(),
    )


def _scores(run):
    return {s.name: s.value for s in run.item_results[0].scores}


def test_plain_four_arg_function_scorer():
    """An ordinary function (input, output, expected=None, metadata=None) works as a scorer."""
    def accuracy(input, output, expected=None, metadata=None):
        return 1.0 if output == expected else 0.0

    assert _scores(_run([accuracy])) == {"accuracy": 1.0}


def test_plain_two_arg_function_scorer():
    def exact(input, output):
        return output == input["q"]

    # output "hi" == input["q"] "hi" -> True
    assert _scores(_run([exact])) == {"exact": True}


def test_backward_compatible_single_ctx_scorer():
    """The existing single-ScorerContext form still works (arity 1 == ctx)."""
    def by_ctx(ctx):
        return 1.0 if ctx.output == ctx.expected else 0.0

    assert _scores(_run([by_ctx])) == {"by_ctx": 1.0}


def test_metric_map_return():
    """A dict with no 'name' key is a metric->value map yielding one Score per entry."""
    def multi(input, output):
        return {"len_ok": len(output) > 0, "starts_h": output.startswith("h")}

    assert _scores(_run([multi], main_score="len_ok")) == {"len_ok": True, "starts_h": True}


def test_named_object_return_still_single_score():
    """A dict WITH a 'name' key stays a single named Score (unchanged)."""
    def one(input, output):
        return {"name": "quality", "value": 0.7}

    assert _scores(_run([one])) == {"quality": 0.7}


def test_unsupported_signature_raises_actionable_error():
    def weird(a, b, c, d, e):
        return 1.0

    try:
        _run([weird])
        raise AssertionError("expected an error for an unsupported scorer signature")
    except (TypeError, ValueError) as e:
        assert "signature" in str(e).lower() or "positional" in str(e).lower()
