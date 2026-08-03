"""deterministic main-score / scorer-name resolution.

Kills the footgun where a scorer whose function name differs from its emitted Score name
silently resolved to not_scored. Covers scalar, explicit Score, Score array, errored
scorer, and the single-vs-multiple-metric rules."""

import pytest

from traceroot.eval.platform import PlatformTransport
from traceroot.eval.results import EvalItemResult
from traceroot.eval.types import Score


def _tx(scorer_names, main_score_name=None):
    return PlatformTransport(
        dataset_id="ds",
        scorer_names=scorer_names,
        main_score_name=main_score_name,
        api_key="k",
        host_url="http://h",
    )


def _item(scores, error=None):
    return EvalItemResult(
        case_id="c0",
        input={},
        output={},
        expected=None,
        scores=scores,
        scorer_errors={},
        error=error,
        trace_id=None,
    )


def test_single_scorer_is_name_agnostic_footgun_fixed():
    # Scorer 'grade' emits Score(name='quality'); a single scorer resolves name-agnostically,
    # so this is scored (not silently not_scored).
    t = _tx(["grade"])
    status, main = t._status_and_main(_item([Score("quality", 0.9)]))
    assert main == 0.9
    assert status in ("passed", "failed")


def test_scalar_return_uses_declared_name_and_scores():
    # The engine names a scalar with the scorer's declared name; a single scorer scores it.
    t = _tx(["acc"])
    _, main = t._status_and_main(_item([Score("acc", True)]))  # bool -> 1.0
    assert main == 1.0


def test_explicit_main_matches_by_name():
    t = _tx(["a", "b"], main_score_name="a")
    _, main = t._status_and_main(_item([Score("a", 1.0), Score("b", 0.0)]))
    assert main == 1.0


def test_configured_main_never_emitted_raises():
    from traceroot.eval.results import MainScoreError, resolve_main_score_name

    # The one resolver (used by both local result + cloud completion) fails loudly.
    with pytest.raises(MainScoreError, match="never emitted"):
        resolve_main_score_name("grade", ["quality"])


def test_multiple_scorers_no_main_has_no_headline_metric():
    # The engine forbids this for a reported run; a directly-built transport simply has no
    # headline metric (never a silent, arbitrary pick of the first scorer's function name).
    t = _tx(["a", "b"])
    assert t.main_score_name is None
    status, main = t._status_and_main(_item([Score("a", 1.0), Score("b", 0.0)]))
    assert main is None and status == "not_scored"


def test_score_array_single_scorer_takes_first_numeric():
    t = _tx(["multi"])
    _, main = t._status_and_main(_item([Score("a", 0.7), Score("b", 0.2)]))
    assert main == 0.7


def test_errored_case_is_errored_not_scored():
    t = _tx(["acc"])
    status, main = t._status_and_main(_item([], error="boom"))
    assert status == "errored" and main is None


def test_categorical_named_main_is_not_scored():
    # A named main that only ever produced a categorical value has no numeric headline.
    t = _tx(["label", "acc"], main_score_name="label")
    status, main = t._status_and_main(_item([Score("label", "billing"), Score("acc", 1.0)]))
    assert main is None and status == "not_scored"
