"""Cross-language normalized-definition parity + source/config provenance (Python side).

Equivalent Python and TypeScript definitions normalize to the SAME descriptor fields (identity +
policy); provenance (language/source) and the config-hash version are language-specific. The TS
counterpart (eval-definition-parity.test.ts) asserts the same fixture.
"""

import json
from pathlib import Path

from traceroot.eval import Scorer
from traceroot.eval.scorers import scorer_metadata

FIX = json.loads((Path(__file__).parent / "fixtures" / "scorer_definition_parity.json").read_text())


def test_code_definition_matches_parity_fixture():
    @Scorer.code(key="grade", value_type="numeric", direction="higher_is_better", threshold=1.0)
    def grade(input, output, expected=None):
        return 1.0

    md = scorer_metadata(grade)
    for k, v in FIX["code"].items():
        assert md[k] == v, f"{k}: {md.get(k)!r} != {v!r}"


def test_static_judge_definition_matches_parity_fixture():
    j = Scorer.llm_judge(
        key="nc", name="nc", model="claude-sonnet-5",
        rubric="Return 1 if the answer has no conclusion, otherwise 0.",
        output_type="score", threshold=1.0, direction="higher_is_better",
    )
    md = scorer_metadata(j)
    for k, v in FIX["static_judge"].items():
        assert md[k] == v, f"{k}: {md.get(k)!r} != {v!r}"


def test_plain_callable_captures_source_without_decorator():
    """A plain function passed to evaluate() captures provenance WITHOUT any decorator/wrapper."""
    def plain(input, output, expected=None):
        return 1.0

    md = scorer_metadata(plain)
    assert md["language"] == "python"
    assert "def plain" in md["source"]
    assert md["key"] == "plain"  # documented fallback: the function name


def test_dynamic_judge_captures_config_and_builder_provenance():
    @Scorer.llm_judge(name="nc", model="claude-sonnet-5", rubric="Evaluate {{X}}.",
                      threshold=1.0, complete=lambda m, msgs: "1")
    def nc(input, output, expected=None):
        return {"X": output}

    md = scorer_metadata(nc)
    assert md["version"].startswith("cfg_")   # declarative config hashed
    assert "def nc" in md["source"]           # AND the builder callback's provenance captured


def test_static_judge_needs_no_source_and_no_function():
    """A static judge requires no function; its config-hash version is its source (no code source)."""
    j = Scorer.llm_judge(name="s", model="claude-sonnet-5", rubric="Grade 0..1.", threshold=1.0)
    md = scorer_metadata(j)
    assert md["version"].startswith("cfg_")
    assert "source" not in md  # a static judge captures no code source
