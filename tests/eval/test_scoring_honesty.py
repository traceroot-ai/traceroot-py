"""Honest scoring semantics.

A missing or errored judgment must NEVER become a numeric zero, and the distinct
outcomes -- quality failure, task error, scorer error, not-scored, pending review --
must stay separable.
"""

import json

from traceroot.eval import Dataset, DeferredScore, EvalCase, Score, Scorer, ScorerContext, evaluate
from traceroot.eval.results import aggregate_scores, case_status


def _one(expected=1, **kw):
    ds = Dataset(name="d")
    ds.upsert(EvalCase(input=1, id="c0", expected=expected, **kw))
    return ds


def echo(x):
    return x


def test_non_finite_score_does_not_poison_the_summary_mean():
    # H1 makes a non-finite score `errored` on the wire, but the LOCAL aggregate must agree: a NaN
    # folded into the mean makes run.json contain a bare `NaN` (invalid per the JSON spec — strict
    # parsers reject it) and disagrees with .summary() and the platform. It must be excluded.
    import math

    ds = Dataset(name="nan-agg")
    ds.upsert(EvalCase(input=1, id="c0", expected=1))
    run = evaluate(name="r", dataset=ds, task=echo, scorers=[lambda ctx: float("nan")], local=True)
    metric = next(iter(run.score_summary.values()))
    assert metric.mean is None  # excluded from the mean, not NaN
    assert metric.count == 1  # still counts as a produced score
    # And the saved artifact is valid JSON — no bare NaN/Infinity token that a strict parser rejects.
    import os
    import tempfile

    path = os.path.join(tempfile.mkdtemp(), "run.json")
    run.save(path)

    def _reject(token):
        raise AssertionError(f"artifact has a bare {token} token (invalid JSON)")

    json.loads(open(path).read(), parse_constant=_reject)


def test_aggregate_excludes_non_finite_but_keeps_finite_values():
    from traceroot.eval.results import EvalItemResult, aggregate_scores

    def _item(cid, value):
        return EvalItemResult(cid, value, value, value, [Score("m", value)], {}, None, None)

    summary = aggregate_scores(
        [_item("a", 1.0), _item("b", float("nan")), _item("c", float("inf"))]
    )
    assert summary["m"].mean == 1.0  # only the finite value contributes
    assert summary["m"].count == 3  # all three still count as produced


def test_summary_shows_per_metric_pass_rate():
    ds = Dataset(name="passrate")
    ds.upsert(EvalCase(input=1, id="a", expected=1))  # will pass
    ds.upsert(EvalCase(input=0, id="b", expected=1))  # will fail

    @Scorer.code(key="hit", value_type="numeric", direction="higher_is_better", threshold=1.0)
    def hit(ctx):
        return Score("hit", 1.0 if ctx.output == ctx.expected else 0.0)

    run = evaluate(name="r", dataset=ds, task=echo, scorers=[hit], local=True)
    line = next(ln for ln in run.summary().splitlines() if ln.strip().startswith("hit"))
    assert "pass=1/2" in line  # one of two cleared the declared threshold
    assert "count=2" in line


def test_summary_pass_rate_is_name_agnostic_for_a_lone_scorer():
    # A lone scorer emitting a lone metric owns it even when the emitted Score name differs from the
    # scorer's declared name — the SAME resolution PlatformTransport uses, so local pass-rate == the
    # platform's. (Regression: a by-name-only lookup would miss this and show no pass-rate.)
    ds = Dataset(name="agnostic")
    ds.upsert(EvalCase(input=1, id="a", expected=1))
    s = Scorer.code(key="x", value_type="numeric", direction="higher_is_better", threshold=1.0)(
        lambda ctx: Score("differently_named", 1.0)
    )
    run = evaluate(name="r", dataset=ds, task=echo, scorers=[s], local=True)
    line = next(ln for ln in run.summary().splitlines() if "mean=" in ln)
    assert "pass=1/1" in line


def test_summary_omits_pass_rate_when_no_threshold_declared():
    ds = Dataset(name="nopolicy")
    ds.upsert(EvalCase(input=1, id="a", expected=1))
    # a plain-function scorer declares no threshold -> nothing can judge pass/fail
    run = evaluate(name="r", dataset=ds, task=echo, scorers=[lambda ctx: 0.5], local=True)
    line = next(ln for ln in run.summary().splitlines() if "mean=" in ln)
    assert "pass=" not in line  # no fabricated verdict
    assert "count=1" in line


def test_evaluate_retains_the_declared_scorer_policy_on_the_result():
    # The run remembers the policy it scored under (name/threshold/direction), so a later
    # result.upload() can re-declare it instead of re-registering policy-less. Captured even for a
    # local run (no transport), which is the "run now, upload later" flow.
    @Scorer.code(key="acc", value_type="numeric", direction="higher_is_better", threshold=0.8)
    def acc(ctx):
        return Score(name="acc", value=1.0)

    run = evaluate(name="r", dataset=_one(), task=echo, scorers=[acc], local=True)
    spec = next(s for s in (run.scorer_specs or []) if s["name"] == "acc")
    assert spec["threshold"] == 0.8
    assert spec["direction"] == "higher_is_better"


class TestErrorVocabulary:
    def test_quality_failure_is_not_scored_at_case_level_not_errored(self):
        # scorer ran and judged poorly -> a real 0 score, no error. Case status carries no
        # headline pass/fail (that lives per-score); with no error the case is "not_scored".
        run = evaluate(name="r", dataset=_one(expected=999), task=echo, scorers=[lambda c: 0.0])
        item = run.item_results[0]
        assert item.error is None
        assert item.scores[0].value == 0.0  # a genuine 0, not coerced away
        assert case_status(item) == "not_scored"

    def test_task_error_is_errored_with_no_scores(self):
        def boom(x):
            raise ValueError("kaboom")

        run = evaluate(name="r", dataset=_one(), task=boom, scorers=[lambda c: 1.0])
        item = run.item_results[0]
        assert "kaboom" in item.error
        assert item.scores == []  # scorers are skipped, NOT scored 0
        assert case_status(item) == "errored"

    def test_one_scorer_error_preserves_other_scores(self):
        def good(c):
            return 1.0

        def broken(c):
            raise RuntimeError("judge down")

        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[good, broken])
        item = run.item_results[0]
        # the good score survives; the broken scorer is recorded as an error, not a 0.
        assert [s.name for s in item.scores] == ["good"]
        assert item.scores[0].value == 1.0
        assert "judge down" in item.scorer_errors["broken"]
        assert case_status(item) == "errored"  # a scorer error makes the case errored

    def test_abstain_none_is_not_scored_not_zero(self):
        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[lambda c: None])
        item = run.item_results[0]
        assert item.scores == []  # abstention produced no score at all
        assert case_status(item) == "not_scored"  # distinct from "failed"

    def test_missing_expected_abstain_is_not_scored(self):
        # A scorer that abstains when it has nothing to compare against must NOT be
        # turned into a 0 by the harness.
        def needs_expected(ctx: ScorerContext):
            return None if ctx.expected is None else (1.0 if ctx.output == ctx.expected else 0.0)

        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="c0"))  # no expected
        run = evaluate(name="r", dataset=ds, task=echo, scorers=[needs_expected])
        item = run.item_results[0]
        assert item.scores == []
        assert case_status(item) == "not_scored"

    def test_deferred_human_score_is_pending_not_zero(self):
        def defer(c):
            return DeferredScore(name="human_quality", reason="awaiting review")

        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[defer])
        item = run.item_results[0]
        assert item.scores[0].value == "pending"  # not a numeric 0
        assert case_status(item) == "not_scored"  # pending review != failure


class TestAggregateHonesty:
    def test_abstentions_and_errors_do_not_deflate_the_mean(self):
        # 2 real 1.0 scores + 1 abstention + 1 scorer error -> mean is 1.0 over the 2
        # that actually produced a score, NOT 0.5 (abstain/error are not zeros).
        def acc(c):
            return 1.0

        def flaky(c):
            if c.input == 2:
                raise RuntimeError("down")  # error on one case
            if c.input == 3:
                return None  # abstain on another
            return 1.0

        ds = Dataset(name="d")
        for i in range(4):
            ds.upsert(EvalCase(input=i, id=f"c{i}", expected=i))
        run = evaluate(name="r", dataset=ds, task=echo, scorers=[acc, flaky])

        summary = aggregate_scores(run.item_results)
        assert summary["acc"].mean == 1.0 and summary["acc"].count == 4
        # flaky produced a score on cases 0 and 1 only (2 -> error, 3 -> abstain)
        assert summary["flaky"].count == 2
        assert summary["flaky"].mean == 1.0

    def test_score_summary_survives_serialization(self):
        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[lambda c: Score("m", 1.0)])
        restored = type(run).from_dict(run.to_dict())
        assert restored.to_dict() == run.to_dict()


class TestScorerVersionHonesty:
    """scorer_version is never invented: absent unless the scorer declares one."""

    def test_unversioned_scorer_reports_none_not_one(self):
        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[lambda c: 1.0])
        assert run.item_results[0].scores[0].version is None  # not "1"

    def test_declared_version_via_scorer_attribute(self):
        def scorer(ctx):
            return 1.0

        scorer.version = "2.1"  # explicitly declared
        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[scorer])
        assert run.item_results[0].scores[0].version == "2.1"

    def test_explicit_score_version_is_preserved(self):
        def scorer(ctx):
            return Score("m", 1.0, version="9")

        run = evaluate(name="r", dataset=_one(), task=echo, scorers=[scorer])
        assert run.item_results[0].scores[0].version == "9"

    def test_runner_events_do_not_invent_version(self):
        from traceroot.eval.results import EvalItemResult
        from traceroot.eval.runner import ScorePolicy, _score_event, _scorer_error_events

        item = EvalItemResult(
            "c", None, None, None, [Score("acc", 1.0)], {"bad": "boom"}, None, None
        )
        event = _score_event(item.scores[0], ScorePolicy(None), single_emission=True)
        assert event["scorer_version"] is None
        assert _scorer_error_events(item)[0]["scorer_version"] is None
        # scorer NAME and independent error are still preserved
        assert _scorer_error_events(item)[0]["scorer_name"] == "bad"
