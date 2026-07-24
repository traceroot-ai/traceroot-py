"""OE-1: EvalCase, Score, ScorerContext data types."""

import dataclasses

import pytest

from traceroot.eval import EvalCase, Score, ScorerContext


class TestEvalCase:
    def test_input_is_required_others_optional(self):
        case = EvalCase(input={"message": "hi"})
        assert case.input == {"message": "hi"}
        assert case.id is None
        assert case.expected is None
        assert case.metadata is None
        assert case.source_trace_id is None
        assert case.source_span_id is None
        assert case.score_target_span_id is None

    def test_all_fields_settable(self):
        case = EvalCase(
            input={"m": 1},
            id="case-001",
            expected={"route": "billing"},
            metadata={"category": "dup"},
            source_trace_id="trace-123",
            source_span_id="span-456",
            score_target_span_id="span-789",
        )
        assert case.id == "case-001"
        assert case.expected == {"route": "billing"}
        assert case.metadata == {"category": "dup"}
        assert case.source_trace_id == "trace-123"
        assert case.source_span_id == "span-456"
        assert case.score_target_span_id == "span-789"

    def test_is_frozen(self):
        case = EvalCase(input=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            case.input = 2  # type: ignore[misc]

    def test_input_required_positionally(self):
        with pytest.raises(TypeError):
            EvalCase()  # type: ignore[call-arg]


class TestScore:
    def test_name_and_value_required(self):
        s = Score(name="acc", value=1.0)
        assert s.name == "acc"
        assert s.value == 1.0
        assert s.comment is None
        assert s.metadata is None

    def test_value_accepts_float_int_bool_str(self):
        assert Score(name="a", value=0.5).value == 0.5
        assert Score(name="a", value=3).value == 3
        assert Score(name="a", value=True).value is True
        assert Score(name="a", value="billing").value == "billing"

    def test_comment_and_metadata(self):
        s = Score(name="acc", value=0.0, comment="wrong", metadata={"k": "v"})
        assert s.comment == "wrong"
        assert s.metadata == {"k": "v"}

    def test_is_frozen(self):
        s = Score(name="acc", value=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.value = 0.0  # type: ignore[misc]


class TestScorerContext:
    def test_carries_all_four_fields(self):
        ctx = ScorerContext(
            input={"m": 1},
            output={"route": "billing"},
            expected={"route": "billing"},
            metadata={"c": "x"},
        )
        assert ctx.input == {"m": 1}
        assert ctx.output == {"route": "billing"}
        assert ctx.expected == {"route": "billing"}
        assert ctx.metadata == {"c": "x"}

    def test_expected_and_metadata_may_be_none(self):
        ctx = ScorerContext(input=1, output=2, expected=None, metadata=None)
        assert ctx.expected is None
        assert ctx.metadata is None

    def test_is_frozen(self):
        ctx = ScorerContext(input=1, output=2, expected=None, metadata=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.output = 3  # type: ignore[misc]
