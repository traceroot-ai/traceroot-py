"""Wire robustness at the reporting boundary: replay-safe totals, contract-cap clamping,
and honest metric ownership for a single metric-map scorer.

Network is stubbed via PlatformTransport._request so no real HTTP happens.
"""

import json

from traceroot.eval import capabilities
from traceroot.eval.platform import (
    _EXPLANATION_MAX,
    _METADATA_MAX,
    _PAYLOAD_TEXT_MAX,
    _SCORE_ERROR_MAX,
    _STRING_VALUE_MAX,
    _TASK_ERROR_MAX,
    _TRUNCATION_SUFFIX,
    PlatformTransport,
)
from traceroot.eval.results import EvalItemResult
from traceroot.eval.transport import RunHandle
from traceroot.eval.types import Score


class RecordingTransport(PlatformTransport):
    """PlatformTransport with the HTTP seam replaced by an in-memory recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_run_id": "run_1"}
        return {}

    def bodies(self, suffix: str) -> list[dict]:
        return [b for _, p, b in self.requests if p.endswith(suffix)]


def _t(**kwargs) -> RecordingTransport:
    return RecordingTransport("ds_1", api_key="tr-x", host_url="https://h", **kwargs)


def _item(**kwargs) -> EvalItemResult:
    base = dict(
        case_id="tc1",
        input={"m": 1},
        output={"r": 1},
        expected={"r": 1},
        scores=[],
        scorer_errors={},
        error=None,
        trace_id=None,
    )
    base.update(kwargs)
    return EvalItemResult(**base)  # type: ignore[arg-type]


_RUN = RunHandle(name="e", dataset_name="d", metadata=None)


class TestReplaySafeTotals:
    """Re-recording the SAME case (a retry, or a second upload() on one transport) must
    REPLACE its contribution to the completion totals, never add to it."""

    def test_same_case_recorded_twice_counts_once(self):
        t = _t()
        t.create_run("e", "d", None)
        item = _item(error="boom", scorer_errors={"acc": "bad"})
        t.record_item_result(_RUN, item)
        t.record_item_result(_RUN, item)  # retry / replay of the SAME case id
        t.finish_run(_RUN)
        body = t.bodies("/complete")[0]
        assert body["task_error_count"] == 1
        assert body["scorer_error_count"] == 1

    def test_scored_count_is_replay_safe(self):
        t = _t()
        t.create_run("e", "d", None)
        item = _item(scores=[Score(name="acc", value=1.0)])
        t.record_item_result(_RUN, item)
        t.record_item_result(_RUN, item)
        t.finish_run(_RUN)
        assert t.bodies("/complete")[0]["scored_count"] == 1

    def test_distinct_cases_still_accumulate(self):
        t = _t()
        t.create_run("e", "d", None)
        t.record_item_result(_RUN, _item(case_id="a", error="x"))
        t.record_item_result(_RUN, _item(case_id="b", error="y"))
        t.finish_run(_RUN)
        assert t.bodies("/complete")[0]["task_error_count"] == 2


class TestContractCapClamping:
    """Contract-capped string fields are clamped SDK-side with an explicit marker, so an
    oversized value never 400s and aborts the whole run."""

    def test_explanation_clamped(self):
        t = _t()
        t.create_run("e", "d", None)
        t.record_item_result(_RUN, _item(scores=[Score(name="acc", value=1.0, comment="x" * 9000)]))
        expl = t.bodies("/results")[0]["scores"][0]["explanation"]
        assert len(expl) == _EXPLANATION_MAX
        assert expl.endswith(_TRUNCATION_SUFFIX)

    def test_scorer_error_clamped(self):
        t = _t()
        t.create_run("e", "d", None)
        t.record_item_result(_RUN, _item(scorer_errors={"acc": "e" * 9000}))
        err = t.bodies("/results")[0]["scores"][0]["error"]
        assert len(err) == _SCORE_ERROR_MAX
        assert err.endswith(_TRUNCATION_SUFFIX)

    def test_task_error_clamped(self):
        t = _t()
        t.create_run("e", "d", None)
        t.record_item_result(_RUN, _item(error="b" * 20000))
        te = t.bodies("/results")[0]["task_error"]
        assert len(te) == _TASK_ERROR_MAX
        assert te.endswith(_TRUNCATION_SUFFIX)

    def test_string_value_clamped(self):
        t = _t()
        t.create_run("e", "d", None)
        t.record_item_result(_RUN, _item(scores=[Score(name="label", value="s" * 5000)]))
        sv = t.bodies("/results")[0]["scores"][0]["string_value"]
        assert len(sv) == _STRING_VALUE_MAX
        assert sv.endswith(_TRUNCATION_SUFFIX)

    def test_payload_text_clamped(self):
        t = _t()
        t.create_run("e", "d", None)
        big = "p" * (_PAYLOAD_TEXT_MAX + 10)
        t.record_item_result(_RUN, _item(input=big, output=big, expected=big))
        body = t.bodies("/results")[0]
        for field in ("input", "candidate_output", "expected_output"):
            assert len(body[field]) == _PAYLOAD_TEXT_MAX
            assert body[field].endswith(_TRUNCATION_SUFFIX)

    def test_under_cap_values_pass_through_untouched(self):
        t = _t()
        t.create_run("e", "d", None)
        t.record_item_result(
            _RUN,
            _item(error="short", scores=[Score(name="acc", value=1.0, comment="fine")]),
        )
        body = t.bodies("/results")[0]
        assert body["task_error"] == "short"
        assert body["scores"][0]["explanation"] == "fine"

    def test_run_metadata_clamped_on_registration(self):
        """Run metadata is free-form and user-supplied (a whole prompt, a config dump), and it is
        the ONE field that reaches the backend unclamped. Over the cap, registration 400s and the
        run never starts -- worse than any per-result rejection."""
        t = _t()
        t.create_run("e", "d", {"prompt": "m" * (_METADATA_MAX + 100)})
        meta = t.bodies("/evaluation-runs")[0]["metadata"]
        assert meta["truncated"] is True
        assert len(json.dumps(meta)) <= _METADATA_MAX

    def test_under_cap_run_metadata_passes_through_untouched(self):
        t = _t()
        t.create_run("e", "d", {"commit": "abc123"})
        assert t.bodies("/evaluation-runs")[0]["metadata"] == {"commit": "abc123"}

    def test_scorer_source_clamped_on_registration(self):
        t = _t(scorer_specs=[{"name": "acc", "source": "c" * 60000}])
        t.create_run("e", "d", None)
        ref = t.bodies("/evaluation-runs")[0]["scorers"][0]
        assert len(ref["source"]) == 50_000
        assert ref["source"].endswith(_TRUNCATION_SUFFIX)


class TestSingleScorerOwnership:
    """The single-scorer name-agnostic owner rule applies only when that scorer emitted
    exactly ONE metric. A metric map must not have a verdict stamped on the metric the
    declared threshold does not own."""

    def test_metric_map_does_not_borrow_the_declared_threshold(self):
        t = _t(
            scorer_specs=[{"name": "quality", "threshold": 0.9, "direction": "higher_is_better"}]
        )
        t.create_run("e", "d", None)
        t.record_item_result(
            _RUN,
            _item(
                scores=[
                    Score(name="accuracy", value=0.95),
                    Score(name="latency_ms", value=120.0),
                ]
            ),
        )
        scores = {s["scorer_name"]: s for s in t.bodies("/results")[0]["scores"]}
        assert "passed" not in scores["accuracy"]  # name doesn't match the declaration
        assert "passed" not in scores["latency_ms"]  # 120 >= 0.9 must NOT become a pass

    def test_matching_metric_name_still_gets_its_verdict(self):
        t = _t(
            scorer_specs=[{"name": "quality", "threshold": 0.9, "direction": "higher_is_better"}]
        )
        t.create_run("e", "d", None)
        t.record_item_result(
            _RUN,
            _item(
                scores=[Score(name="quality", value=0.95), Score(name="latency_ms", value=120.0)]
            ),
        )
        scores = {s["scorer_name"]: s for s in t.bodies("/results")[0]["scores"]}
        assert scores["quality"]["passed"] is True
        assert "passed" not in scores["latency_ms"]

    def test_lone_metric_keeps_the_name_agnostic_shortcut(self):
        # `grade` declared, `quality` emitted: a single scorer emitting a SINGLE metric owns it.
        t = _t(scorer_specs=[{"name": "grade", "threshold": 0.9, "direction": "higher_is_better"}])
        t.create_run("e", "d", None)
        t.record_item_result(_RUN, _item(scores=[Score(name="quality", value=0.95)]))
        assert t.bodies("/results")[0]["scores"][0]["passed"] is True


class TestCapabilities:
    def test_handshake_has_no_provenance_flag(self):
        # The typed provenance path the flag referred to is gone; TS never advertised it.
        assert "provenance" not in capabilities()
