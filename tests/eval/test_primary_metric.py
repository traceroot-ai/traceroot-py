"""Phase 2 — primary-metric behavior.

Recording multiple metrics must NOT require a primary. When none is selected the run still completes
with every score stored; no overall pass-rate/verdict is invented. A numeric metric with no declared
threshold is scored but carries no fabricated pass/fail. `primary_metric` is an alias for `main_score`.
"""

from traceroot.eval import Dataset, Score, evaluate, scorer
from traceroot.eval.platform import PlatformTransport
from traceroot.eval.transport import FakeTransport


class _Capture(PlatformTransport):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path.endswith("/evaluation-runs"):
            return {"evaluation_run_id": "run1"}
        return {}


def _ds():
    ds = Dataset("d")
    ds.add(input={"q": "hi"}, id="c0", expected="hi")
    return ds


def two_metrics(ctx):
    return {"accuracy": 1.0, "coverage": 0.5}  # two numeric metrics from one scorer


def test_multiple_metrics_without_primary_does_not_error():
    """Multiple numeric metrics + no main_score completes (no MainScoreError) and stores all."""
    run = evaluate(name="r", dataset=_ds(), task=lambda x: "hi",
                   scorers=[two_metrics], transport=FakeTransport())
    scores = {s.name: s.value for s in run.item_results[0].scores}
    assert scores == {"accuracy": 1.0, "coverage": 0.5}


def test_no_primary_invents_no_pass_rate():
    """With no primary metric, no case is marked passed/failed (no invented verdict)."""
    run = evaluate(name="r", dataset=_ds(), task=lambda x: "hi",
                   scorers=[two_metrics], transport=FakeTransport())
    from traceroot.eval.results import case_status
    assert case_status(run.item_results[0]) == "not_scored"


def test_primary_metric_alias():
    """`primary_metric` selects the headline metric just like `main_score`."""
    run = evaluate(name="r", dataset=_ds(), task=lambda x: "hi",
                   scorers=[two_metrics], primary_metric="accuracy", transport=FakeTransport())
    assert run.main_score_name == "accuracy"


@scorer(value_type="numeric")  # numeric, NO threshold declared
def unbounded(ctx):
    return Score("quality", 0.3)


def test_numeric_without_threshold_is_scored_but_no_passfail():
    """A numeric metric with no declared threshold is recorded but has no fabricated pass/fail."""
    t = _Capture("ds_1", api_key="tr-x", host_url="https://h", dataset_version_id="v1")
    evaluate(name="r", dataset=_ds(), task=lambda x: "hi", scorers=[unbounded], report_to=t)
    results = next(b for _m, p, b in t.requests if p.endswith("/results"))
    (score,) = results["scores"]
    assert score["numeric_value"] == 0.3          # scored
    assert "passed" not in score                  # but no fabricated pass/fail
