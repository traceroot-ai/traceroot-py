"""the extensible ``required_inputs`` scorer descriptor (replaces the
never-shipped ``reference_based`` boolean). Declared explicitly for code scorers, derived
from template placeholders for ``llm_judge``, unknown (omitted) for a bare callable."""

import pytest

from traceroot.eval.platform import PlatformTransport
from traceroot.eval.scorers import describe_scorers, llm_judge, scorer, scorer_metadata


def test_code_scorer_declares_required_inputs():
    @scorer(value_type="numeric", required_inputs=["output"])
    def has_conclusion(ctx):
        return 1.0

    assert scorer_metadata(has_conclusion)["required_inputs"] == ["output"]


def test_bare_callable_is_unknown_and_omitted():
    def plain(ctx):
        return 1.0

    # A bare callable declares nothing -> unknown (None -> omitted at the wire boundary).
    assert scorer_metadata(plain)["required_inputs"] is None


def test_decorated_scorer_without_declaration_is_unknown():
    @scorer(value_type="numeric")
    def acc(ctx):
        return 1.0

    assert scorer_metadata(acc)["required_inputs"] is None


def test_llm_judge_output_only_derivation():
    # "Has conclusion section": examines only candidate output; must NOT require expected.
    judge = llm_judge(
        name="no_conclusion",
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": "Grade whether the answer has a conclusion."},
            {"role": "user", "content": "ANSWER:\n{{output}}"},
        ],
    )
    assert scorer_metadata(judge)["required_inputs"] == ["output"]


def test_llm_judge_derives_canonical_order():
    judge = llm_judge(
        name="match",
        model="m",
        messages=[{"role": "user", "content": "{{expected}} vs {{output}} for {{input}}"}],
    )
    assert scorer_metadata(judge)["required_inputs"] == ["input", "output", "expected"]


def test_llm_judge_explicit_override_wins():
    judge = llm_judge(
        name="j",
        model="m",
        messages=[{"role": "user", "content": "{{output}}"}],
        required_inputs=["input", "output"],
    )
    assert scorer_metadata(judge)["required_inputs"] == ["input", "output"]


def test_required_inputs_rejects_unknown_field():
    with pytest.raises(ValueError):
        scorer(required_inputs=["bogus"])
    with pytest.raises(ValueError):
        llm_judge(
            name="j",
            model="m",
            messages=[{"role": "user", "content": "x"}],
            required_inputs=["nope"],
        )


def test_required_inputs_serialized_on_the_wire():
    @scorer(value_type="numeric", required_inputs=["output"])
    def has_conclusion(ctx):
        return 1.0

    t = PlatformTransport(
        dataset_id="ds",
        api_key="k",
        host_url="http://h",
        scorer_specs=describe_scorers([has_conclusion]),
    )
    assert t._scorer_refs()[0]["required_inputs"] == ["output"]
