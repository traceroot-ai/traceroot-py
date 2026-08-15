"""First-class scorer comparison metadata (name/version/value_type/direction/threshold).

Authoring is via the @scorer decorator (or supported callable attributes); plain callables
keep working. Defaults: numeric/boolean -> higher_is_better, categorical -> none; an explicit
direction always wins; direction is NEVER inferred from the scorer name; version is never
fabricated.
"""

import pytest

from traceroot.eval import Score, ScorerContext, describe_scorers, llm_judge, scorer
from traceroot.eval.scorers import scorer_metadata


def test_plain_callable_has_name_no_version_no_type():
    def routing_accuracy(ctx):
        return 1.0

    m = scorer_metadata(routing_accuracy)
    assert m["name"] == "routing_accuracy"
    assert m["version"] is None  # never fabricated
    assert m["value_type"] is None  # unknown until it runs / declared
    assert m["direction"] is None  # cannot default without a value_type


def test_numeric_defaults_higher_is_better():
    @scorer(value_type="numeric")
    def acc(ctx):
        return 1.0

    m = scorer_metadata(acc)
    assert (m["value_type"], m["direction"]) == ("numeric", "higher_is_better")


def test_boolean_defaults_higher_is_better():
    @scorer(value_type="boolean")
    def safe(ctx):
        return True

    assert scorer_metadata(safe)["direction"] == "higher_is_better"


def test_categorical_defaults_none():
    @scorer(value_type="categorical")
    def route(ctx):
        return "billing"

    assert scorer_metadata(route)["direction"] == "none"


def test_explicit_direction_wins():
    @scorer(value_type="numeric", direction="lower_is_better", threshold=100)
    def latency(ctx):
        return 42.0

    m = scorer_metadata(latency)
    assert m["direction"] == "lower_is_better"
    assert m["threshold"] == 100


def test_direction_not_inferred_from_name():
    # a scorer literally named "latency" must NOT be guessed as lower_is_better.
    @scorer(value_type="numeric")
    def latency(ctx):
        return 42.0

    assert scorer_metadata(latency)["direction"] == "higher_is_better"


def test_declared_name_and_version_win():
    @scorer(name="Routing accuracy", version="v3")
    def acc(ctx):
        return 1.0

    m = scorer_metadata(acc)
    assert m["name"] == "Routing accuracy" and m["version"] == "v3"


def test_invalid_value_type_or_direction_raise():
    with pytest.raises(ValueError):
        scorer(value_type="nonsense")
    with pytest.raises(ValueError):
        scorer(direction="up")


def test_describe_scorers_list_and_value_type_hint():
    @scorer(direction="lower_is_better")
    def latency(ctx):
        return 1.0

    def acc(ctx):
        return Score("acc", 1.0)

    specs = describe_scorers([acc, latency], value_types={"latency": "numeric"})
    by = {s["name"]: s for s in specs}
    # a runtime value_type hint fills an undeclared value_type
    assert by["latency"]["value_type"] == "numeric"
    assert by["latency"]["direction"] == "lower_is_better"
    assert "value_type" not in by["acc"]  # undeclared fields are omitted, not exposed as null


class TestScorerSpecsOnWire:
    def test_create_run_emits_rich_scorer_descriptors(self):
        from traceroot.eval.platform import PlatformTransport

        @scorer(value_type="numeric", direction="higher_is_better", threshold=0.9, version="v2")
        def acc(ctx):
            return 1.0

        t = _recording(PlatformTransport)
        t.scorer_specs = describe_scorers([acc])
        t.create_run("r", "d", None)
        (sc,) = t.reqs[0][2]["scorers"]
        # comparison metadata
        assert sc["name"] == "acc" and sc["version"] == "v2"
        assert sc["value_type"] == "numeric" and sc["direction"] == "higher_is_better"
        assert sc["threshold"] == 0.9
        # definition fields ride the same descriptor
        assert sc["scorer_type"] == "code" and sc["output_type"] == "score"
        assert sc["language"] == "python" and "def acc" in sc["source"]

    def test_declared_threshold_drives_per_score_passed(self):
        from traceroot.eval.platform import PlatformTransport
        from traceroot.eval.results import EvalItemResult

        @scorer(value_type="numeric", threshold=0.8)
        def acc(ctx):
            return 0.85

        t = _recording(PlatformTransport)
        t.scorer_specs = describe_scorers([acc])
        t.create_run("r", "d", None)
        # 0.85 >= declared threshold 0.8 -> the emitted score's own `passed` is True
        # (would be False against the 1.0 default).
        item = EvalItemResult("c", {}, {}, {}, [Score("acc", 0.85)], {}, None, None)
        t.record_item_result(None, item)
        (score,) = t.reqs[-1][2]["scores"]
        assert score["passed"] is True

    def test_plain_names_still_work_without_specs(self):
        from traceroot.eval.platform import PlatformTransport

        t = _recording(PlatformTransport)
        t.create_run("r", "d", None)
        assert t.reqs[0][2]["scorers"] == [{"name": "acc", "version": "unversioned"}]


class TestRetainedSpecsMatchTheRunsEffectivePolicy:
    """``result.scorer_specs`` is what an explicit ``upload()`` re-declares, so it must be the
    policy the run actually registered under -- the transport's, when the caller pre-set one."""

    def test_a_pre_set_transport_policy_wins_over_the_callable_derived_one(self):
        from traceroot.eval import Dataset, EvalCase, FakeTransport, evaluate

        @scorer(value_type="numeric", threshold=0.9)
        def acc(ctx):
            return 1.0

        custom = [{"name": "acc", "value_type": "numeric", "threshold": 0.25}]
        t = FakeTransport()
        t.scorer_specs = custom  # a caller-declared policy: the run registers under THIS
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="c0", expected=1))

        run = evaluate(name="r", data=ds, task=lambda x: x, scorers=[acc], transport=t)

        assert run.scorer_specs == custom  # not describe_scorers([acc]) (threshold 0.9)

    def test_without_a_pre_set_policy_the_callable_derived_specs_are_retained(self):
        from traceroot.eval import Dataset, EvalCase, FakeTransport, evaluate

        @scorer(value_type="numeric", threshold=0.9)
        def acc(ctx):
            return 1.0

        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="c0", expected=1))

        run = evaluate(
            name="r", data=ds, task=lambda x: x, scorers=[acc], transport=FakeTransport()
        )

        assert run.scorer_specs == describe_scorers([acc])


def _recording(cls):
    t = cls("ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h")
    t.reqs = []
    t._request = lambda m, p, b=None: t.reqs.append((m, p, b)) or {"evaluation_run_id": "run_1"}
    return t


# --- scorer definition (read-only Scorer detail) ------------------------------------


def test_code_scorer_reports_type_source_and_output_type():
    @scorer(output_type="score", threshold=1.0, description="Exact match", metadata={"team": "q"})
    def exact_match(ctx):
        return 1.0 if ctx.output == ctx.expected else 0.0

    m = scorer_metadata(exact_match)
    assert m["scorer_type"] == "code"
    assert m["language"] == "python"
    assert "def exact_match(ctx):" in m["source"]
    assert "@scorer" not in m["source"]  # decorator line stripped
    assert m["output_type"] == "score"
    assert m["threshold"] == 1.0
    assert m["description"] == "Exact match"
    assert m["metadata"] == {"team": "q"}


def test_multiline_decorator_is_fully_stripped_from_source():
    # A multi-line @scorer(...) must not leak its argument lines into the captured source;
    # the definition should begin at the function's own `def` header.
    @scorer(
        value_type="numeric",
        direction="higher_is_better",
        threshold=1.0,
        description="both cities present",
        metadata={"kind": "coverage"},
    )
    def covers_both_cities(ctx):
        text = (ctx.output or "").lower()
        return 1.0 if "sf" in text and "tokyo" in text else 0.0

    src = scorer_metadata(covers_both_cities)["source"]
    assert src.startswith("def covers_both_cities(ctx):")  # begins at def, no decorator args
    assert "@scorer" not in src
    assert "value_type=" not in src and "metadata=" not in src  # no leaked decorator arg lines


def test_output_type_derived_from_value_type_but_explicit_wins():
    @scorer(value_type="categorical")
    def route(ctx):
        return "billing"

    assert scorer_metadata(route)["output_type"] == "classification"

    @scorer(value_type="categorical", output_type="score")
    def weird(ctx):
        return "x"

    assert scorer_metadata(weird)["output_type"] == "score"  # explicit wins


def test_undeclared_definition_fields_are_none_then_omitted():
    def plain(ctx):
        return 1.0

    m = scorer_metadata(plain)
    assert m["scorer_type"] == "code"  # always present
    assert m["output_type"] is None  # no value_type -> not derived
    assert m["description"] is None and m["metadata"] is None
    # code scorers still report their source (it is the real definition, not inference)
    assert "def plain" in m["source"]


def test_llm_judge_reports_model_and_messages_not_source():
    concise = llm_judge(
        name="concise",
        version="1",
        model="claude-sonnet-5",
        messages=[
            {"role": "system", "content": "Rate 0..1."},
            {"role": "user", "content": "ANSWER:\n{{output}}"},
        ],
        output_type="score",
        threshold=0.8,
        description="conciseness",
        metadata={"team": "quality"},
    )
    m = scorer_metadata(concise)
    assert m["scorer_type"] == "llm_judge"
    assert m["model"] == "claude-sonnet-5"
    assert m["messages"][1]["content"] == "ANSWER:\n{{output}}"  # authored template, verbatim
    assert m["output_type"] == "score" and m["threshold"] == 0.8
    assert "source" not in m and "language" not in m


def test_llm_judge_executes_with_injected_complete_and_renders_placeholders():
    seen = {}

    def fake_complete(model, messages):
        seen["model"] = model
        seen["rendered"] = messages
        return "The score is 0.7"  # judge contract: a single unambiguous number

    concise = llm_judge(
        name="concise",
        model="claude-sonnet-5",
        messages=[{"role": "user", "content": "ANSWER:\n{{output}}"}],
        complete=fake_complete,
    )
    ctx = ScorerContext(input="q", output="a concise answer", expected=None, metadata=None)
    score = concise(ctx)
    assert score.name == "concise" and score.value == 0.7  # parsed numeric
    assert seen["model"] == "claude-sonnet-5"
    assert seen["rendered"][0]["content"] == "ANSWER:\na concise answer"  # {{output}} rendered
