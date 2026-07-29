"""Phase 5: raw-data ownership boundary.

The SDK reports RAW outcomes + measurements; the backend owns candidate-vs-baseline
comparison entirely (baseline selection AND the deltas). This test captures every request
a full reported run makes and asserts what is and is NOT on the wire: the SDK sends NO
comparison linkage at all -- not even a baseline_run_id.
"""

from traceroot.eval import Dataset, EvalCase, evaluate, scorer
from traceroot.eval.platform import PlatformTransport


class _Capture(PlatformTransport):
    """PlatformTransport that records every request instead of hitting the network."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.requests: list[tuple] = []

    def _request(self, method, path, body=None):
        self.requests.append((method, path, body))
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_run_id": "run_cand"}
        return {}


@scorer(value_type="numeric", direction="higher_is_better", version="v2")
def acc(ctx):
    if ctx.input == "boom":
        raise ValueError("scorer down")  # scorer error on one case
    return 1.0 if ctx.output == ctx.expected else 0.0


def task(x):
    if x == "explode":
        raise RuntimeError("task boom")  # task error on one case
    return x


def _ds():
    ds = Dataset("d")
    ds.dataset_id = "ds_1"
    ds.dataset_version_id = "dsv_1"
    ds.upsert(EvalCase(input="ok", id="c0", expected="ok"))
    ds.upsert(EvalCase(input="boom", id="c1", expected="boom"))  # scorer error
    ds.upsert(EvalCase(input="explode", id="c2", expected="explode"))  # task error
    return ds


def _run():
    t = _Capture(
        "ds_1",
        scorer_names=["acc"],
        candidate_version="sonnet",
        dataset_version_id="dsv_1",
        api_key="tr-x",
        host_url="https://h",
    )
    result = evaluate(
        name="routing",
        dataset=_ds(),
        task=task,
        scorers=[acc],
        candidate_version="sonnet",
        report_to=t,
    )
    return result, t.requests


class TestSendsRawData:
    def test_register_run_carries_identity_and_scorer_metadata(self):
        _, reqs = _run()
        reg = reqs[0][2]
        assert reqs[0][1] == "/api/v1/public/evaluation-runs"
        assert reg["candidate_version"] == "sonnet"
        assert "baseline_run_id" not in reg  # SDK sends NO comparison linkage
        assert reg["main_score_name"] == "acc"
        # scorer comparison metadata + definition (raw description, not a comparison result)
        (sc,) = reg["scorers"]
        assert sc["name"] == "acc" and sc["version"] == "v2"
        assert sc["value_type"] == "numeric" and sc["direction"] == "higher_is_better"
        # definition: a decorated code scorer reports its type, output type, and source
        assert sc["scorer_type"] == "code"
        assert sc["output_type"] == "score"  # derived from numeric
        assert sc["language"] == "python" and "def acc" in sc["source"]
        # run metadata/provenance is NOT sent (backend registration schema is strict)
        assert "metadata" not in reg

    def test_results_carry_raw_outputs_values_errors_duration_trace(self):
        _, reqs = _run()
        results = [b for _m, p, b in reqs if p.endswith("/results")]
        by_id = {b["test_case_id"]: b for b in results}

        ok = by_id["c0"]
        assert ok["candidate_output"] == "ok"  # raw candidate output
        assert ok["status"] == "passed"
        assert ok["main_score"] == 1.0
        assert "duration_ms" in ok  # measurement present (int or null)
        assert "trace_id" in ok  # trace identity link
        assert ok["scores"] == [
            {"scorer_name": "acc", "scorer_version": "v2", "numeric_value": 1.0}
        ]  # raw per-scorer value + declared version

        # scorer error: preserved as a per-scorer error, task_error stays null
        se = by_id["c1"]
        assert se["task_error"] is None
        assert any("scorer down" in (s.get("error") or "") for s in se["scores"])

        # task error: reported as a task error, status errored
        te = by_id["c2"]
        assert te["status"] == "errored"
        assert "task boom" in te["task_error"]

    def test_complete_carries_counts_and_main_score(self):
        _, reqs = _run()
        comp = next(b for _m, p, b in reqs if p.endswith("/complete"))
        assert comp["status"] in ("completed", "completed_with_errors")
        assert "scored_count" in comp and "task_error_count" in comp
        assert "main_score" in comp


class TestDoesNotSendComparisonLabels:
    def test_no_change_delta_baseline_output_or_regression_counts_on_wire(self):
        _, reqs = _run()
        forbidden = {
            "baseline_run_id",
            "baseline_run",
            "change",
            "baseline_output",
            "baseline_score",
            "delta",
            "score_delta",
            "regressions",
            "regression_count",
            "improved",
            "regressed",
            "unchanged",
        }
        for _m, _p, body in reqs:
            if not body:
                continue
            assert forbidden.isdisjoint(body.keys()), f"leaked comparison label in {body.keys()}"
            for s in body.get("scores", []):
                assert forbidden.isdisjoint(s.keys())


class TestLocalOwnership:
    def test_run_metadata_provenance_is_local_only(self):
        result, reqs = _run()
        # provenance lives on the LOCAL result, never on a request body
        assert result.metadata is None or isinstance(result.metadata, dict)
        for _m, _p, body in reqs:
            assert not body or "metadata" not in body
