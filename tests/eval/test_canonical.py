"""One canonical form, shared with TypeScript (C2/C3/C4).

Every vector here is asserted against the SAME fixture bytes by
``traceroot-ts/packages/traceroot/tests/eval-canonical.test.ts``, so a divergence in either SDK
fails a test in that SDK instead of silently forking a dataset on the platform.
"""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from tests.eval.parity_vectors import CASE_FIXTURE, SCORER_FIXTURE, decode
from traceroot.eval import Dataset, EvalCase, Scorer
from traceroot.eval.canonical import CanonicalizationError, canonical_json, es_number, normalize
from traceroot.eval.scorers import scorer_metadata

# --- the shared fixture ---------------------------------------------------------------------


@pytest.mark.parametrize("vector", CASE_FIXTURE["cases"], ids=lambda v: v["_comment"][:48])
def test_canonical_json_matches_shared_fixture(vector):
    assert canonical_json(decode(vector["input"])) == vector["canonical_json"]


def test_case_ids_and_revision_match_shared_fixture():
    d = Dataset(CASE_FIXTURE["dataset_key"])
    assert d.dataset_id == CASE_FIXTURE["dataset_id"]
    for vector in CASE_FIXTURE["cases"]:
        produced = d.add(input=decode(vector["input"]))
        assert produced.id == vector["id"], vector["_comment"]
    assert d.snapshot().revision == CASE_FIXTURE["revision"]


def test_judge_config_version_matches_shared_fixture():
    """The config hash is the judge's version: the SAME rubric must produce the SAME ``cfg_`` in
    both SDKs, or a scorer looks edited every time the authoring language changes."""
    fix = SCORER_FIXTURE["static_judge"]
    md = scorer_metadata(
        Scorer.llm_judge(
            key=fix["key"],
            name=fix["key"],
            model=fix["model"],
            rubric="Return 1 if the answer has no conclusion, otherwise 0.",
            output_type=fix["output_type"],
            threshold=fix["threshold"],
            direction=fix["direction"],
        )
    )
    assert md["version"] == fix["version"]


def test_judge_config_version_with_metadata_matches_shared_fixture():
    fix = SCORER_FIXTURE["judge_with_metadata"]
    md = scorer_metadata(
        Scorer.llm_judge(
            key=fix["key"],
            name=fix["key"],
            model=fix["model"],
            rubric="Grade {{output}} against {{expected}}.",
            output_type=fix["output_type"],
            threshold=fix["threshold"],
            direction=fix["direction"],
            value_type=fix["value_type"],
            metadata=fix["metadata"],
        )
    )
    assert md["version"] == fix["version"]


def test_changed_rubric_changes_the_config_version():
    """Half the versioning contract: a constant-returning hash would pass everything above."""

    def build(rubric: str) -> str:
        return scorer_metadata(Scorer.llm_judge(name="j", model="m", rubric=rubric, threshold=1.0))[
            "version"
        ]

    assert build("Grade strictly.") != build("Grade leniently.")


# --- ECMAScript Number::toString --------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, "0"),
        (-0.0, "0"),
        (1.0, "1"),
        (-1.0, "-1"),
        (2.5, "2.5"),
        (0.1, "0.1"),
        (100.0, "100"),
        (1e-6, "0.000001"),
        (1e-7, "1e-7"),
        (1.5e-7, "1.5e-7"),
        (1e-5, "0.00001"),
        (1e16, "10000000000000000"),
        (1e20, "100000000000000000000"),
        (1e21, "1e+21"),
        (1e22, "1e+22"),
        (-1.5e-9, "-1.5e-9"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (1 / 3, "0.3333333333333333"),
    ],
)
def test_es_number_matches_javascript_string(value, expected):
    """These are JavaScript's `String(x)` outputs, verbatim. Python's own repr differs for most of
    them (`1.0`, `1e-05`, `1e+21`), which is exactly why the hashing path cannot use it."""
    assert es_number(value) == expected


def test_large_int_renders_as_the_double_it_becomes_on_the_wire():
    """Beyond 2**53 a Python int has no faithful JS counterpart; render the double it will come
    back as, so the hash survives the round trip instead of changing on the first pull."""
    assert canonical_json(2**53) == "9007199254740992"
    assert canonical_json(10**21) == canonical_json(1e21) == "1e+21"


def test_wire_value_of_a_big_int_is_the_double_that_was_hashed():
    """``normalize()`` is the WIRE form and ``canonical_json()`` is the HASHED form. An int beyond
    2**53 renders as the double it becomes, so the wire value must be that same double -- shipping
    the exact int would push content that no longer hashes to its own revision."""
    assert normalize(2**53) == 2**53  # still exact: representable
    assert isinstance(normalize(2**53), int)
    big = 2**53 + 1
    assert normalize(big) == float(big)
    assert isinstance(normalize(big), float)
    # what goes on the wire re-hashes to the revision that was published
    assert canonical_json(normalize(big)) == canonical_json(big)
    # bool is an int subclass and must not fall into the number branch
    assert normalize(True) is True and normalize(False) is False


def test_int_with_no_double_representation_is_rejected():
    with pytest.raises(CanonicalizationError, match="too large to canonicalize"):
        normalize(10**400)


# --- normalization rules ------------------------------------------------------------------


def test_normalize_is_the_wire_form_and_is_idempotent():
    value = {
        "d": datetime(2020, 1, 1, tzinfo=UTC),
        "s": {2, 1},
        "b": b"hi",
        "n": float("nan"),
    }
    once = normalize(value)
    assert once == {"d": "2020-01-01T00:00:00.000Z", "s": [1, 2], "b": [104, 105], "n": "NaN"}
    # Re-normalizing what went on the wire must not change it — this is what makes a pulled
    # dataset re-hash to the revision that published it.
    assert canonical_json(once) == canonical_json(value)


def test_datetime_renders_like_javascript_toisostring():
    # Naive == UTC (a JS Date is an instant; there is no naive form to mirror).
    assert normalize(datetime(2020, 1, 1, 12, 30)) == "2020-01-01T12:30:00.000Z"
    assert normalize(date(2020, 1, 1)) == "2020-01-01T00:00:00.000Z"
    # An offset-aware datetime converts to UTC; precision truncates to milliseconds.
    aware = datetime(
        2020, 1, 1, 12, 30, 45, 123999, tzinfo=timezone(timedelta(hours=5, minutes=30))
    )
    assert normalize(aware) == "2020-01-01T07:00:45.123Z"


def test_object_keys_sort_by_code_point_not_utf16():
    # U+E000 (private use) sorts BEFORE U+1D11E by code point, and AFTER it by UTF-16 code unit.
    astral, pua = chr(0x1D11E), chr(0xE000)
    assert canonical_json({astral: 1, pua: 2}) == f'{{"{pua}":2,"{astral}":1}}'


def test_numeric_string_keys_sort_as_strings():
    assert canonical_json({"10": "a", "2": "b"}) == '{"10":"a","2":"b"}'


def test_aliased_reference_hashes_as_content_but_a_cycle_is_rejected():
    shared = {"x": 1}
    assert canonical_json({"a": shared, "b": shared}) == canonical_json(
        {"a": {"x": 1}, "b": {"x": 1}}
    )
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(CanonicalizationError, match="reference cycle"):
        canonical_json(cyclic)


@pytest.mark.parametrize(
    "value", [Decimal("1.5"), object(), lambda x: x, {1, 2}.__iter__(), complex(1, 2)]
)
def test_non_json_values_are_rejected_not_hashed_in_a_language_specific_shape(value):
    with pytest.raises(CanonicalizationError):
        canonical_json(value)


@pytest.mark.parametrize(
    "value",
    [
        "\ud800",  # a lone high surrogate as a value
        {"k": "lo\udc00"},  # ... nested in a value
        {"\ud834": "v"},  # ... as a mapping key
    ],
)
def test_a_lone_surrogate_is_rejected_not_hashed(value):
    # A lone UTF-16 surrogate can't encode to UTF-8: without an explicit guard Python raises a raw
    # UnicodeEncodeError while TypeScript silently hashes the escaped form. Both must reject it as a
    # CanonicalizationError so the SDKs never disagree.
    with pytest.raises(CanonicalizationError, match="surrogate"):
        canonical_json(value)


def test_a_valid_astral_character_still_canonicalizes():
    # A real non-BMP character (a surrogate PAIR / single code point) is valid text, not a lone
    # surrogate, and must keep hashing normally.
    assert canonical_json("\U0001f600") == '"\U0001f600"'


def test_dataset_add_rejects_a_non_serializable_payload_naming_the_field():
    with pytest.raises(CanonicalizationError, match="test case expected"):
        Dataset("d").add(input=1, expected=Decimal("1.5"))
    with pytest.raises(CanonicalizationError, match="test case input"):
        Dataset("d").upsert(EvalCase(input=object(), id="a"))


def test_nan_and_infinity_are_sentinel_strings_in_both_sdks():
    assert canonical_json([float("nan"), math.inf, -math.inf]) == '["NaN","Infinity","-Infinity"]'
