"""OE-2: result/summary types + per-scorer aggregation."""

import json

from traceroot.eval import EvalRunResult, Score
from traceroot.eval.results import (
    EvalItemResult,
    ScoreSummary,
    UploadState,
    aggregate_scores,
)


def _item(case_id, scores, scorer_errors=None, error=None):
    return EvalItemResult(
        case_id=case_id,
        input={"m": case_id},
        output={"o": case_id},
        expected=None,
        scores=scores,
        scorer_errors=scorer_errors or {},
        error=error,
        trace_id=None,
    )


class TestUploadState:
    def test_defaults_to_uploaded(self):
        u = UploadState()
        assert u.status == "uploaded"
        assert u.dashboard_url is None


class TestAggregateScores:
    def test_numeric_mean(self):
        items = [_item("a", [Score("acc", 1.0)]), _item("b", [Score("acc", 0.0)])]
        summ = aggregate_scores(items)
        assert summ["acc"].mean == 0.5
        assert summ["acc"].count == 2

    def test_bool_scores_average(self):
        items = [_item("a", [Score("hit", True)]), _item("b", [Score("hit", False)])]
        summ = aggregate_scores(items)
        assert summ["hit"].mean == 0.5
        assert summ["hit"].count == 2

    def test_categorical_str_scores_count_only(self):
        items = [_item("a", [Score("label", "billing")]), _item("b", [Score("label", "tech")])]
        summ = aggregate_scores(items)
        assert summ["label"].mean is None
        assert summ["label"].count == 2

    def test_mixed_numeric_and_str_under_one_name(self):
        items = [_item("a", [Score("x", 1.0)]), _item("b", [Score("x", "skip")])]
        summ = aggregate_scores(items)
        assert summ["x"].mean == 1.0  # mean over numerics only
        assert summ["x"].count == 2  # count over all

    def test_scorer_error_contributes_no_score(self):
        items = [
            _item("a", [Score("acc", 1.0)]),
            _item("b", [], scorer_errors={"acc": "boom"}),
        ]
        summ = aggregate_scores(items)
        assert summ["acc"].mean == 1.0
        assert summ["acc"].count == 1

    def test_empty(self):
        assert aggregate_scores([]) == {}


class TestEvalItemResult:
    def test_to_dict_json_serializable(self):
        item = _item("a", [Score("acc", 1.0, comment="ok")])
        d = item.to_dict()
        assert d["case_id"] == "a"
        assert d["scores"][0]["name"] == "acc"
        json.dumps(d)  # must not raise


class TestEvalRunResult:
    def _run(self):
        items = [_item("a", [Score("acc", 1.0)]), _item("b", [Score("acc", 0.0)])]
        return EvalRunResult(
            name="routing-v2",
            item_results=items,
            score_summary=aggregate_scores(items),
            upload_state=UploadState(),
        )

    def test_to_dict_json_serializable(self):
        d = self._run().to_dict()
        assert d["name"] == "routing-v2"
        assert d["upload"]["status"] == "uploaded"
        assert d["score_summary"]["acc"]["mean"] == 0.5
        assert len(d["item_results"]) == 2
        json.dumps(d)  # must not raise

    def test_str_mentions_name_count_and_means(self):
        s = str(self._run())
        assert "routing-v2" in s
        assert "2" in s  # item count
        assert "acc" in s

    def test_upload_status_is_explicit(self):
        assert self._run().upload_state.status == "uploaded"


class TestScoreSummary:
    def test_fields(self):
        s = ScoreSummary(name="acc", mean=0.5, count=2)
        assert (s.name, s.mean, s.count) == ("acc", 0.5, 2)
