"""A scorer's DECLARED name is its reported identity everywhere.

``@scorer(name="quality")`` over ``def grade(ctx)`` registers the definition as ``quality`` but the
engine named the emitted Score (and the scorer->metric ownership key) after ``__name__`` -- so the
completion manifest looked up ``emitted["quality"]``, found the ownership recorded under ``grade``,
and dropped ``emitted_metrics`` entirely. With two such scorers the platform also cannot resolve a
metric's owner, so every ``passed`` goes out unset.
"""

from traceroot.eval import Dataset, EvalCase, Scorer, evaluate, scorer
from traceroot.eval.platform import PlatformTransport
from traceroot.eval.scorers import scorer_metadata


class _Capture(PlatformTransport):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path.endswith("/evaluation-runs"):
            return {"evaluation_run_id": "run1"}
        return {}


@scorer(name="quality", value_type="numeric", direction="higher_is_better", threshold=0.9)
def grade(ctx):
    return 0.95


@scorer(name="brevity", value_type="numeric", direction="lower_is_better", threshold=0.5)
def shortness(ctx):
    return 0.2


def _ds():
    ds = Dataset("d")
    ds.dataset_id = "ds_1"
    ds.dataset_version_id = "v1"
    ds.upsert(EvalCase(input="i", id="c0", expected="e"))
    return ds


def _body(requests, suffix):
    return next(b for _m, p, b in requests if p.endswith(suffix))


def _run():
    t = _Capture("ds_1", api_key="tr-x", host_url="https://h", dataset_version_id="v1")
    result = evaluate(
        name="r", dataset=_ds(), task=lambda x: "out", scorers=[grade, shortness], report_to=t
    )
    return t, result


def test_declared_name_is_the_emitted_score_name():
    _t, result = _run()
    assert [s.name for s in result.item_results[0].scores] == ["quality", "brevity"]


def test_declared_name_resolves_each_metric_passed():
    t, _result = _run()
    scores = _body(t.requests, "/results")["scores"]
    assert {s["scorer_name"]: s.get("passed") for s in scores} == {
        "quality": True,  # 0.95 >= 0.9
        "brevity": True,  # 0.2 <= 0.5
    }


def test_manifest_entry_name_matches_the_emitted_metric():
    t, _result = _run()
    by = {s["name"]: s for s in _body(t.requests, "/complete")["scorers"]}
    assert by["quality"]["emitted_metrics"] == [
        {
            "name": "quality",
            "value_type": "numeric",
            "direction": "higher_is_better",
            "threshold": 0.9,
        }
    ]
    assert by["brevity"]["emitted_metrics"] == [
        {
            "name": "brevity",
            "value_type": "numeric",
            "direction": "lower_is_better",
            "threshold": 0.5,
        }
    ]


def test_declared_name_wins_over_lambda_name():
    """A lambda's ``__name__`` is ``<lambda>``; the declared name is the reported identity."""
    covers = Scorer.code(lambda ctx: 1, name="covers", value_type="numeric")
    md = scorer_metadata(covers)
    assert md["name"] == "covers"
    assert md["key"] == "covers"
