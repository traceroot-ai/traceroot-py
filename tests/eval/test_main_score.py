"""Boolean main-score behavior is consistent across local + cloud status.

True -> 1.0, False -> 0.0 for the numeric MAIN score and passed/failed status; the
individual scorer payload keeps the boolean as bool_value; errors stay distinct.
"""

from traceroot.eval import Score
from traceroot.eval.platform import PlatformTransport
from traceroot.eval.results import EvalItemResult, case_status


def _item(scores, error=None):
    return EvalItemResult("c", {}, {}, {}, scores, {}, error, None)


def _t(main_name="m"):
    return PlatformTransport(
        "ds", scorer_names=["m"], main_score_name=main_name, api_key="tr-x", host_url="https://h"
    )


class TestCloudStatusAndMain:
    def test_true_is_passed_1(self):
        assert _t()._status_and_main(_item([Score("m", True)])) == ("passed", 1.0)

    def test_false_is_failed_0(self):
        assert _t()._status_and_main(_item([Score("m", False)])) == ("failed", 0.0)

    def test_numeric_uses_threshold(self):
        assert _t()._status_and_main(_item([Score("m", 0.5)])) == ("failed", 0.5)
        assert _t()._status_and_main(_item([Score("m", 1.0)])) == ("passed", 1.0)

    def test_categorical_main_is_not_scored(self):
        assert _t()._status_and_main(_item([Score("m", "billing")])) == ("not_scored", None)

    def test_missing_main_is_not_scored(self):
        assert _t()._status_and_main(_item([])) == ("not_scored", None)

    def test_task_error_is_errored(self):
        assert _t()._status_and_main(_item([Score("m", True)], error="boom")) == ("errored", None)

    def test_picks_named_main_over_others(self):
        # only the main_score_name scorer drives status; a bool main is honored.
        scores = [Score("other", 0.0), Score("m", True)]
        assert _t("m")._status_and_main(_item(scores)) == ("passed", 1.0)


class TestBooleanScorerPayloadStaysBool:
    def test_bool_value_not_coerced_to_numeric(self):
        payload = _t()._scores_payload(_item([Score("m", True)]))
        assert payload[0]["bool_value"] is True
        assert "numeric_value" not in payload[0]


class TestLocalCloudConsistency:
    def test_local_case_status_matches_cloud_for_bool(self):
        # local case_status already maps bool -> 1/0; cloud must agree.
        assert case_status(_item([Score("m", True)])) == "passed"
        assert case_status(_item([Score("m", False)])) == "failed"
        assert _t()._status_and_main(_item([Score("m", True)]))[0] == "passed"
        assert _t()._status_and_main(_item([Score("m", False)]))[0] == "failed"
