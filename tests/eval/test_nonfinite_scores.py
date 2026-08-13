"""A non-finite numeric score is a scorer FAILURE, never a value.

``NaN`` / ``Infinity`` (a 0/0 ratio, a divergent average) are not JSON numbers. Serialized
naively, Python emits a bare ``NaN`` token that the backend's ``JSON.parse`` rejects -- one
poisoned score 400s the whole run -- while TypeScript's ``JSON.stringify`` turns it into
``null``, silently persisting an empty score that poisons the aggregate mean. Both are wrong
and they are wrong DIFFERENTLY, so the reporting boundary converts a non-finite value into an
errored score in both SDKs. Parity with traceroot-ts/tests/eval-nonfinite-scores.test.ts.
"""

from __future__ import annotations

import json

import pytest

from traceroot.eval.platform import PlatformTransport
from traceroot.eval.results import EvalItemResult
from traceroot.eval.transport import RunHandle
from traceroot.eval.types import Score

INFINITY = float("inf")
NAN = float("nan")


class _Recording(PlatformTransport):
    """PlatformTransport with the HTTP seam replaced by a recorder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_run_id": "run_1"}
        return {}


def _transport(**kwargs) -> _Recording:
    t = _Recording("ds_1", api_key="tr-x", host_url="https://h", **kwargs)
    t.run_id = "run_1"
    return t


def _item(value, name="ratio") -> EvalItemResult:
    return EvalItemResult(
        case_id="c0",
        input={"m": 1},
        output={"m": 1},
        expected={"m": 1},
        scores=[Score(name=name, value=value)],
        scorer_errors={},
        error=None,
        trace_id=None,
        duration_ms=1.0,
    )


def _result_body(t: _Recording) -> dict:
    return next(b for _m, p, b in t.requests if p.endswith("/results"))


@pytest.mark.parametrize("value", [NAN, INFINITY, -INFINITY])
class TestNonFiniteIsAnError:
    def test_never_reaches_the_wire_as_a_number(self, value):
        t = _transport()
        t.record_item_result(RunHandle(name="e", dataset_name="d", metadata=None), _item(value))
        body = _result_body(t)
        # allow_nan=False is exactly what the backend's JSON.parse enforces: a bare NaN/Infinity
        # token is not JSON, and a silent null would be a fabricated score.
        json.dumps(body, allow_nan=False)
        (score,) = body["scores"]
        assert "numeric_value" not in score
        assert score.get("bool_value") is None
        assert score.get("string_value") is None
        assert "passed" not in score  # an errored score has no verdict

    def test_is_reported_as_an_errored_score(self, value):
        t = _transport()
        t.record_item_result(RunHandle(name="e", dataset_name="d", metadata=None), _item(value))
        (score,) = _result_body(t)["scores"]
        assert score["scorer_name"] == "ratio"
        assert score["scorer_version"] == "unversioned"
        assert "non-finite" in score["error"]

    def test_errors_the_case_and_counts_toward_the_completion_totals(self, value):
        t = _transport()
        run = RunHandle(name="e", dataset_name="d", metadata=None)
        t.record_item_result(run, _item(value))
        assert _result_body(t)["status"] == "errored"
        t.finish_run(run)
        complete = next(b for _m, p, b in t.requests if p.endswith("/complete"))
        assert complete["scorer_error_count"] == 1
        assert complete["scored_count"] == 0  # a case whose only score errored is not scored
        assert complete["status"] == "completed_with_errors"


class TestFiniteScoresAreUntouched:
    def test_a_finite_number_still_reports_its_value(self):
        t = _transport()
        t.record_item_result(RunHandle(name="e", dataset_name="d", metadata=None), _item(0.0))
        (score,) = _result_body(t)["scores"]
        assert score["numeric_value"] == 0.0
        assert "error" not in score

    def test_the_message_names_the_metric_and_the_value(self):
        t = _transport()
        t.record_item_result(
            RunHandle(name="e", dataset_name="d", metadata=None), _item(NAN, name="quality")
        )
        (score,) = _result_body(t)["scores"]
        # Byte-identical to the TypeScript SDK's message for the same input.
        assert score["error"] == (
            "ValueError: scorer 'quality' returned a non-finite score value (NaN); "
            "a numeric score must be finite"
        )

    @pytest.mark.parametrize(
        ("value", "rendered"), [(INFINITY, "Infinity"), (-INFINITY, "-Infinity")]
    )
    def test_infinities_render_language_neutrally(self, value, rendered):
        """Python spells these 'inf'/'-inf' and JavaScript 'Infinity'; the wire uses ONE
        spelling so the same scorer bug reads identically from either SDK."""
        t = _transport()
        t.record_item_result(RunHandle(name="e", dataset_name="d", metadata=None), _item(value))
        (score,) = _result_body(t)["scores"]
        assert f"({rendered})" in score["error"]
