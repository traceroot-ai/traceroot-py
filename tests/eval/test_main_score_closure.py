"""integration closure: ONE resolver, ONE resolved value threaded through the local
result and the cloud reporter -- late-bound single metric, explicit multi-metric selection,
and the configuration-failure lifecycle (terminal completion, no orphaned runs)."""

import pytest

from traceroot.eval import evaluate
from traceroot.eval.transport import FakeTransport
from traceroot.eval.types import Score

_DS = [{"input": {"m": 1}, "expected": {"m": 1}}]


def echo(x):
    return x


def _finish_calls(fake):
    return [(c[0], c[1]) for c in fake.calls if c[0] == "finish_run"]


def test_late_bound_single_metric_resolves_to_emitted_name():
    # scorer function identity `grade`, emitted metric `quality`, no explicit main_score.
    def grade(ctx):
        return Score(name="quality", value=1.0)

    fake = FakeTransport()
    result = evaluate(name="r", dataset=_DS, task=echo, scorers=[grade], report_to=fake)

    assert result.main_score_name == "quality"  # resolved to the EMITTED name, never `grade`
    assert result.passed == 1  # local status derives from quality
    assert fake.last_main_score_name == "quality"  # cloud completion carries the resolved name
    assert "quality" in result.summary()


def test_explicit_multi_metric_selection_used_everywhere():
    def accuracy(ctx):  # the first scorer AND first emitted metric
        return Score("accuracy", 0.0)

    def fmt(ctx):
        return Score("format", 1.0)

    fake = FakeTransport()
    result = evaluate(
        name="r",
        dataset=_DS,
        task=echo,
        scorers=[accuracy, fmt],
        main_score="format",
        report_to=fake,
    )

    assert result.main_score_name == "format"
    assert fake.last_main_score_name == "format"
    # Status derives from format (1.0 -> passed), NOT accuracy (0.0 -> would be failed).
    assert result.passed == 1 and result.failed == 0


def test_missing_main_score_completes_run_terminally_then_raises():
    def acc(ctx):
        return Score("accuracy", 1.0)

    fake = FakeTransport()
    with pytest.raises(ValueError, match="accuracy"):  # lists the emitted metrics
        evaluate(
            name="r", dataset=_DS, task=echo, scorers=[acc], main_score="missing", report_to=fake
        )

    # The run WAS registered, then completed in a terminal 'failed' state before raising --
    # never left orphaned in 'running'.
    assert ("finish_run", "failed") in _finish_calls(fake)


def test_multi_scorer_no_main_on_report_path_records_not_scored(monkeypatch):
    """Parity with TS (eval-main-score.test.ts) and the backend contract (main_score_name is
    nullable): a REPORTED multi-scorer run with no main_score records every score with no invented
    verdict (not_scored). It must NOT raise -- main_score is optional on the reporting path exactly
    as it is on the explicit-transport path."""
    import traceroot.eval.engine as engine

    fake = FakeTransport()
    # Force the auto-report branch (transport is None) to yield a transport with no network/creds.
    monkeypatch.setattr(engine, "_auto_transport", lambda *a, **k: fake)

    def s1(ctx):
        return Score("acc", 1.0)

    def s2(ctx):
        return Score("fmt", 1.0)

    result = evaluate(name="r", dataset=_DS, task=echo, scorers=[s1, s2])  # no main_score, no transport

    assert result.main_score_name is None  # no headline invented
    assert result.not_scored == 1  # both scores recorded, no pass/fail verdict
    assert _finish_calls(fake) == [("finish_run", None)]  # completed normally, not 'failed'


def test_no_successful_scores_is_unscored_not_ambiguous():
    def boom(ctx):
        raise RuntimeError("scorer down")

    fake = FakeTransport()
    result = evaluate(name="r", dataset=_DS, task=echo, scorers=[boom], report_to=fake)

    assert result.main_score_name is None  # unscored, not a main-score misconfiguration
    assert result.not_scored == 1
    assert _finish_calls(fake) == [("finish_run", None)]  # completed normally, not 'failed'
