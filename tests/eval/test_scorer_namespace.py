"""Unified `Scorer` namespace: Scorer.code + Scorer.llm_judge (static & dynamic), aliases, hashing."""

import pytest

from traceroot.eval import Scorer, evaluate, llm_judge, scorer
from traceroot.eval.scorers import scorer_metadata
from traceroot.eval.transport import FakeTransport
from traceroot.eval.types import ScorerContext

MODEL = "claude-sonnet-5"


def _ctx(output, query="hi"):
    return ScorerContext(input={"query": query}, output=output, expected=None, metadata=None)


# ── Scorer.code ───────────────────────────────────────────────────────────────
def test_scorer_code_decorator():
    @Scorer.code(name="covers", key="covers", threshold=1.0, direction="higher_is_better")
    def covers(input, output, expected=None):
        return 1.0

    md = scorer_metadata(covers)
    assert md["key"] == "covers" and md["threshold"] == 1.0 and md["scorer_type"] == "code"


def test_scorer_code_adapter_over_callable():
    def exact(input, output):
        return output == input["query"]

    graded = Scorer.code(exact, key="exact", threshold=1.0)
    assert scorer_metadata(graded)["key"] == "exact"


# ── Scorer.llm_judge — static (no function, rubric only) ──────────────────────
def test_static_judge_runs_and_hashes_config():
    judge = Scorer.llm_judge(
        name="no_conclusion",
        model=MODEL,
        rubric="Return 1 if the answer has no conclusion, otherwise 0.",
        threshold=1.0,
        direction="higher_is_better",
        complete=lambda model, messages: "1.0",  # deterministic, offline
    )
    md = scorer_metadata(judge)
    assert md["scorer_type"] == "llm_judge"
    assert md["version"].startswith("cfg_")  # declarative config is the versioned source
    score = judge(_ctx("A concise answer."))
    assert score.value == 1.0


def test_static_judge_config_hash_is_deterministic():
    def mk():
        return Scorer.llm_judge(name="j", model=MODEL, rubric="Grade it 0..1.", threshold=1.0)

    assert scorer_metadata(mk())["version"] == scorer_metadata(mk())["version"]


def _cfg_version(**overrides):
    base = dict(name="j", model=MODEL, rubric="Grade it 0..1.", threshold=1.0)
    return scorer_metadata(Scorer.llm_judge(**{**base, **overrides}))["version"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("rubric", "Grade it 0..10."),  # the rubric IS the judge's source
        ("model", "claude-opus-4-5"),
        ("threshold", 0.5),
        ("direction", "lower_is_better"),
        ("value_type", "categorical"),
        ("output_type", "classification"),
        ("metadata", {"team": "eval"}),
    ],
)
def test_changed_config_changes_the_version(field, value):
    """Determinism alone is satisfied by a constant, so pin the other half: every hashed field
    must MOVE the ``cfg_``. A judge whose version survives a rubric edit is a version that lies."""
    assert _cfg_version(**{field: value}) != _cfg_version()


def test_human_name_is_not_part_of_the_version():
    """`name` is a label, not source: renaming a judge must not read as 'someone edited the
    rubric'. (`key` is the cross-language identity and is likewise outside the config hash.)"""
    assert _cfg_version(name="readability") == _cfg_version()
    assert _cfg_version(key="other_key") == _cfg_version()


# ── Scorer.llm_judge — dynamic (decorator over a builder) ─────────────────────
def test_dynamic_judge_builder_variable_map():
    seen = {}

    @Scorer.llm_judge(
        name="nc",
        model=MODEL,
        rubric="Evaluate {{ANSWER}} against {{USER_REQUEST}}.",
        threshold=1.0,
        complete=lambda model, messages: (seen.update(prompt=messages), "1")[1],
    )
    def nc(input, output, expected=None):
        return {"ANSWER": output, "USER_REQUEST": input["query"]}

    score = nc(_ctx("the answer text", query="the question"))
    assert score.value == 1.0
    rendered = " ".join(m["content"] for m in seen["prompt"])
    assert "the answer text" in rendered and "the question" in rendered  # builder vars rendered


def test_dynamic_judge_builder_messages_list():
    seen = {}

    @Scorer.llm_judge(
        name="nc",
        model=MODEL,
        rubric="unused",
        threshold=1.0,
        complete=lambda model, messages: (seen.update(p=messages), "0")[1],
    )
    def nc(input, output, expected=None):
        return [{"role": "user", "content": f"Judge: {output}"}]

    assert nc(_ctx("XYZ")).value == 0.0
    assert seen["p"] == [{"role": "user", "content": "Judge: XYZ"}]


def test_dynamic_judge_unsupported_builder_return_is_actionable_error():
    @Scorer.llm_judge(
        name="nc", model=MODEL, rubric="r", threshold=1.0, complete=lambda m, msgs: "1"
    )
    def nc(input, output, expected=None):
        return 0.5  # a number is NOT a valid builder return (that's the judge's job)

    try:
        nc(_ctx("x"))
        raise AssertionError("expected an actionable error for a numeric builder return")
    except TypeError as e:
        assert "dynamic builder" in str(e) and "dict" in str(e)


# ── Aliases ───────────────────────────────────────────────────────────────────
def test_old_scorer_alias_is_scorer_code():
    assert Scorer.code is scorer  # `scorer` is the same constructor


def test_old_llm_judge_alias_still_works():
    j = llm_judge(
        name="j",
        model=MODEL,
        messages=[{"role": "user", "content": "{{output}}"}],
        threshold=1.0,
        complete=lambda m, msgs: "1",
    )
    assert scorer_metadata(j)["scorer_type"] == "llm_judge"
    assert j(_ctx("hello")).value == 1.0


def test_static_judge_runs_in_evaluate():
    ds = [{"input": {"query": "q"}, "id": "c0"}]
    judge = Scorer.llm_judge(
        name="nc", model=MODEL, rubric="Grade 0..1.", threshold=1.0, complete=lambda m, msgs: "1"
    )
    run = evaluate(
        name="r", dataset=ds, task=lambda x: "ans", scorers=[judge], transport=FakeTransport()
    )
    assert run.item_results[0].scores[0].value == 1.0
