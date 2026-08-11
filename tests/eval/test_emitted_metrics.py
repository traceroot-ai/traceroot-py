"""Emitted-metric ownership manifest at completion.

A scorer DEFINITION (``grade``) owns the METRIC it emits (``quality``). The platform keys a metric's
policy on the EMITTED-metric name, so completion must carry ``scorers[].emitted_metrics`` recording
that ``grade`` emits ``quality`` with grade's declared policy -- otherwise the platform cannot resolve
``quality``'s threshold/direction and defaults it (wrong pass/fail and comparison verdicts).
"""

from traceroot.eval import Dataset, EvalCase, Score, evaluate, scorer
from traceroot.eval.platform import PlatformTransport


class _Capture(PlatformTransport):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path.endswith("/evaluation-runs"):
            return {"evaluation_run_id": "run1"}
        return {}


@scorer(value_type="numeric", direction="lower_is_better", threshold=0.62)
def grade(ctx):
    return Score("quality", 0.3)  # emitted metric 'quality' != definition name 'grade'


@scorer(value_type="numeric", direction="higher_is_better", threshold=0.0)
def length(ctx):
    return Score("length", 1.0)


def _ds():
    ds = Dataset("d")
    ds.dataset_id = "ds_1"
    ds.dataset_version_id = "v1"
    ds.upsert(EvalCase(input="i", id="c0", expected="e"))
    return ds


def _complete(requests):
    return next(b for _m, p, b in requests if p.endswith("/complete"))


def test_completion_carries_emitted_metric_ownership():
    t = _Capture("ds_1", api_key="tr-x", host_url="https://h", dataset_version_id="v1")
    evaluate(name="r", dataset=_ds(), task=lambda x: "out", scorers=[grade, length],
             report_to=t)
    by = {s["name"]: s for s in _complete(t.requests)["scorers"]}
    # Definition 'grade' declares it emits metric 'quality' with grade's OWN policy -- the platform
    # can now key quality's threshold/direction on the emitted name, not the definition name.
    assert by["grade"]["emitted_metrics"] == [
        {"name": "quality", "value_type": "numeric", "direction": "lower_is_better", "threshold": 0.62}
    ]
    # Function identity stays separate from metric identity.
    assert by["grade"]["name"] == "grade"
    assert by["length"]["emitted_metrics"] == [
        {"name": "length", "value_type": "numeric", "direction": "higher_is_better", "threshold": 0.0}
    ]


def test_single_scorer_name_divergent_metric_manifest():
    """One scorer, emitted name != function name, no explicit main -> the manifest still records
    grade->quality so the platform resolves the sole metric's policy."""
    t = _Capture("ds_1", api_key="tr-x", host_url="https://h", dataset_version_id="v1")
    evaluate(name="r", dataset=_ds(), task=lambda x: "out", scorers=[grade], report_to=t)
    by = {s["name"]: s for s in _complete(t.requests)["scorers"]}
    assert by["grade"]["emitted_metrics"] == [
        {"name": "quality", "value_type": "numeric", "direction": "lower_is_better", "threshold": 0.62}
    ]
