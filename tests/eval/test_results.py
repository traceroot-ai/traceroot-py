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

    def test_counts_block_is_the_cross_sdk_shape(self):
        """The saved artifact's counts block must be cross-loadable with the TS SDK's, and must
        carry no case-level passed/failed (the SDK derives no such verdict)."""
        items = [
            _item("ok", [Score("acc", 1.0)]),
            _item("taskerr", [], error="boom"),
            _item("scorererr", [], scorer_errors={"grade": "kaboom"}),
        ]
        run = EvalRunResult(
            name="r",
            item_results=items,
            score_summary=aggregate_scores(items),
            upload_state=UploadState(),
        )
        counts = run.to_dict()["counts"]
        assert list(counts) == [
            "case_count",
            "errored",
            "not_scored",
            "task_errors",
            "scorer_errors",
        ]
        assert counts == {
            "case_count": 3,
            "errored": 2,
            "not_scored": 1,
            "task_errors": 1,
            "scorer_errors": 1,
        }

    def test_save_load_round_trip(self, tmp_path):
        r = self._run()
        p = tmp_path / "run.json"
        r.save(str(p))
        loaded = EvalRunResult.load(str(p))
        assert loaded.name == r.name
        assert len(loaded.item_results) == 2
        assert loaded.score_summary["acc"].mean == 0.5
        assert loaded.upload_state.status == r.upload_state.status
        assert not (tmp_path / "run.json.tmp").exists()  # atomic save leaves no temp file behind

    def test_nonfinite_score_round_trips_as_nonfinite_not_categorical(self, tmp_path):
        # serialize_value() stringifies NaN/inf to the JSON tokens "NaN"/"Infinity"; on load they
        # must restore to non-finite FLOATS, not become categorical string scores that would read
        # as legitimate successful metrics. A scorer's non-finite result stays excluded from the mean.
        import math

        items = [_item("a", [Score("acc", math.nan)]), _item("b", [Score("acc", 1.0)])]
        r = EvalRunResult(
            name="r",
            item_results=items,
            score_summary=aggregate_scores(items),
            upload_state=UploadState(),
        )
        p = tmp_path / "run.json"
        r.save(str(p))
        loaded = EvalRunResult.load(str(p))
        v = loaded.item_results[0].scores[0].value
        assert isinstance(v, float) and math.isnan(v)  # restored to float NaN, not the string "NaN"
        assert loaded.score_summary["acc"].mean == 1.0  # non-finite excluded; only 1.0 contributes


class TestReuploadRegistration:
    """The transport a re-upload builds must register every scorer the run saw — including one
    that errored on every case (absent from score_summary). Cross-SDK: the TS results.ts
    upload() must union the same three sources."""

    def test_all_erroring_scorer_is_registered(self, monkeypatch):
        from traceroot.eval import platform as platform_mod
        from traceroot.eval.results import RunDatasetRef

        captured: dict = {}

        class _Capture:
            def __init__(self, dataset_id, **kwargs):
                captured["dataset_id"] = dataset_id
                captured.update(kwargs)
                self.run_id = "run_1"

            def create_run(self, name, dataset_name, metadata=None, client_run_id=None):
                return None

            def record_item_result(self, run, item):
                pass

            def record_scores(self, run, case_id, scores):
                pass

            def finish_run(self, run, status=None, emitted_metrics=None):
                return UploadState()

        monkeypatch.setattr(platform_mod, "PlatformTransport", _Capture)
        items = [_item("c0", [Score("acc", 1.0)], scorer_errors={"flaky": "boom"})]
        run = EvalRunResult(
            name="r",
            item_results=items,
            score_summary=aggregate_scores(items),
            upload_state=UploadState(),
            local_run_id="run_local_1",
            dataset=RunDatasetRef("ds_1", "rev_x", "dsv_1", 1),
        )
        run.upload()
        assert captured["scorer_names"] == ["acc", "flaky"]


class TestScoreSummary:
    def test_fields(self):
        s = ScoreSummary(name="acc", mean=0.5, count=2)
        assert (s.name, s.mean, s.count) == ("acc", 0.5, 2)
