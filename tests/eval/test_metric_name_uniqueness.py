"""Emitted-metric names must be unique within a run.

A Score row's identity IS its emitted-metric name, and the platform keys that metric's
direction/threshold on the name. Two scorers reporting the same metric name make the policy
ambiguous, so the platform defensively drops the metric to non-directional. The SDK catches the
static case at config time (fail fast, before the run) and warns on the dynamic case (a metric-map
scorer whose emitted name only becomes known during the run).
"""

import pytest

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


def _ds():
    ds = Dataset("d")
    ds.dataset_id = "ds_1"
    ds.dataset_version_id = "v1"
    ds.upsert(EvalCase(input="i", id="c0", expected="e"))
    return ds


def _transport():
    return _Capture("ds_1", api_key="tr-x", host_url="https://h", dataset_version_id="v1")


def _complete(requests):
    return next(b for _m, p, b in requests if p.endswith("/complete"))


# --- 1. config-time, fail fast: two scorers resolving to the same name ---


def test_duplicate_scorer_names_raise_before_the_run_starts():
    @scorer(name="accuracy", threshold=0.5)
    def exact(ctx):
        return 1.0

    @scorer(name="accuracy", threshold=0.9)
    def fuzzy(ctx):
        return 0.5

    ran = []

    def task(x):
        ran.append(x)
        return "out"

    with pytest.raises(ValueError) as exc:
        evaluate(name="r", dataset=_ds(), task=task, scorers=[exact, fuzzy],
                 report_to=_transport())
    msg = str(exc.value)
    assert "'accuracy'" in msg
    assert "metric names must be unique within a run" in msg
    assert "distinct name (or key)" in msg
    # Fail fast: the run never started, so nothing was executed or reported.
    assert ran == []


def test_duplicate_undecorated_function_names_raise():
    """Same resolved name via ``__name__`` (no declared name) is just as ambiguous."""

    def make():
        def relevance(ctx):
            return 1.0

        return relevance

    with pytest.raises(ValueError, match="metric names must be unique within a run"):
        evaluate(name="r", dataset=_ds(), task=lambda x: "out", scorers=[make(), make()],
                 report_to=_transport())


def test_error_lists_every_duplicated_name():
    @scorer(name="a")
    def one(ctx):
        return 1.0

    @scorer(name="a")
    def two(ctx):
        return 1.0

    @scorer(name="b")
    def three(ctx):
        return 1.0

    @scorer(name="b")
    def four(ctx):
        return 1.0

    with pytest.raises(ValueError) as exc:
        evaluate(name="r", dataset=_ds(), task=lambda x: "out",
                 scorers=[one, two, three, four], report_to=_transport())
    assert "'a', 'b'" in str(exc.value)


# --- 2. manifest build, warn: a metric-map scorer colliding with another scorer ---


def test_metric_map_collision_warns_once_and_the_run_still_completes():
    """'quality' is emitted by BOTH scorers; only discoverable after the run, so warn (never fail)."""

    @scorer(name="rubric", value_type="numeric", direction="higher_is_better")
    def rubric(ctx):
        return {"quality": 0.8, "fluency": 0.9}  # metric map: names known only at run time

    @scorer(name="quality", value_type="numeric", direction="lower_is_better", threshold=0.2)
    def quality(ctx):
        return 0.1

    t = _transport()
    with pytest.warns(UserWarning) as rec:
        result = evaluate(name="r", dataset=_ds(), task=lambda x: "out",
                          scorers=[rubric, quality], report_to=t)
    collisions = [w for w in rec if "metric name" in str(w.message)]
    assert len(collisions) == 1
    msg = str(collisions[0].message)
    assert "'quality'" in msg
    assert "'rubric'" in msg  # both contributing scorers are named
    assert "non-directionally" in msg
    # The run succeeded and reported: warning, not failure.
    assert result.case_count == 1
    assert _complete(t.requests)["status"] == "completed"
    assert _complete(t.requests)["scorers"]


# --- 3 + 4. regression guards: no error, no warning on the normal shapes ---


def test_unique_names_produce_no_error_and_no_warning(recwarn):
    @scorer(name="accuracy")
    def accuracy(ctx):
        return 1.0

    @scorer(name="latency")
    def latency(ctx):
        return Score("latency", 0.2)

    t = _transport()
    result = evaluate(name="r", dataset=_ds(), task=lambda x: "out",
                      scorers=[accuracy, latency], report_to=t)
    assert result.case_count == 1
    assert _complete(t.requests)["status"] == "completed"
    assert [str(w.message) for w in recwarn if "metric name" in str(w.message)] == []


def test_single_metric_map_scorer_with_internally_unique_names_does_not_warn(recwarn):
    @scorer(name="rubric")
    def rubric(ctx):
        return {"quality": 0.8, "fluency": 0.9}

    t = _transport()
    result = evaluate(name="r", dataset=_ds(), task=lambda x: "out", scorers=[rubric],
                      report_to=t)
    assert result.case_count == 1
    assert _complete(t.requests)["status"] == "completed"
    assert [str(w.message) for w in recwarn if "metric name" in str(w.message)] == []
    by = {s["name"]: s for s in _complete(t.requests)["scorers"]}
    assert sorted(m["name"] for m in by["rubric"]["emitted_metrics"]) == ["fluency", "quality"]
