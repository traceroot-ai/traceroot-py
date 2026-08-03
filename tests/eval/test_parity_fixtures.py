"""Cross-language parity: assert the Python implementation matches the shared golden fixture
that traceroot-ts validates against too (an identical copy lives in each repo)."""

import json
import pathlib

import pytest

from traceroot.eval.scorers import _parse_judge_output, llm_judge, scorer_metadata

_FIX = json.loads((pathlib.Path(__file__).parent / "fixtures" / "parity.json").read_text())


@pytest.mark.parametrize("case", _FIX["judge_parsing"], ids=lambda c: repr(c["text"]))
def test_judge_parsing_parity(case):
    if case.get("error"):
        with pytest.raises(ValueError):
            _parse_judge_output(case["text"], "score")
    else:
        assert _parse_judge_output(case["text"], "score") == case["expected"]


@pytest.mark.parametrize("case", _FIX["required_inputs"], ids=lambda c: c["content"][:20])
def test_required_inputs_parity(case):
    judge = llm_judge(name="j", model="m", messages=[{"role": "user", "content": case["content"]}])
    assert scorer_metadata(judge)["required_inputs"] == case["expected"]
