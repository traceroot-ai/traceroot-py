"""Phase 5: llm_judge score parsing is an unambiguous exact-value parser, not "first number
wins". A malformed/ambiguous response is an isolated scorer error with the raw text
preserved, never a wrong silent score."""

import pytest

from traceroot.eval.scorers import _parse_judge_output


def _p(text):
    return _parse_judge_output(text, "score")


def test_exact_number_response():
    assert _p("1.0") == 1.0
    assert _p("0.8") == 0.8
    assert _p("  0.0  ") == 0.0
    assert _p("-2") == -2.0


def test_trailing_period_tolerated():
    assert _p("1.0.") == 1.0


def test_single_number_in_prose():
    assert _p("The score is 0.8") == 0.8


def test_footgun_multiple_numbers_does_not_become_first():
    # "Step 3: the score is 0.8" must NOT become 3.
    with pytest.raises(ValueError, match="single numeric score"):
        _p("Step 3: the score is 0.8")


def test_multiple_numbers_raise():
    with pytest.raises(ValueError):
        _p("1 out of 10")


def test_no_number_raises():
    with pytest.raises(ValueError):
        _p("no idea")


def test_error_preserves_raw_response_for_diagnosis():
    with pytest.raises(ValueError, match="Step 3"):
        _p("Step 3: the score is 0.8")


def test_classification_is_passthrough():
    assert _parse_judge_output("  billing  ", "classification") == "billing"
