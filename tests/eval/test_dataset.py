"""OE-1: Dataset stable-id upsert, ordering, identity, JSON."""

import json

from traceroot.eval import Dataset, EvalCase


class TestDatasetUpsert:
    def test_explicit_id_replaces_in_place(self):
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="a", expected="x"))
        ds.upsert(EvalCase(input=2, id="a", expected="y"))
        assert len(ds) == 1
        assert ds.get("a").input == 2
        assert ds.get("a").expected == "y"

    def test_upsert_returns_stored_case(self):
        ds = Dataset(name="d")
        returned = ds.upsert(EvalCase(input=1, id="a"))
        assert returned.id == "a"
        assert returned.input == 1

    def test_anonymous_gets_stable_ulid(self):
        # DS-1: anonymous cases get a stable ULID tc_ id (not positional case-{n}),
        # so ids survive serialization and remote push.
        ds = Dataset(name="d")
        r0 = ds.upsert(EvalCase(input="a"))
        r1 = ds.upsert(EvalCase(input="b"))
        assert r0.id.startswith("tc_") and r1.id.startswith("tc_")
        assert r0.id != r1.id
        assert len(ds) == 2

    def test_reupsert_returned_case_is_idempotent(self):
        ds = Dataset(name="d")
        stored = ds.upsert(EvalCase(input="a"))
        ds.upsert(stored)
        assert len(ds) == 1

    def test_reupsert_anonymous_object_appends(self):
        # Documented: a still-anonymous (id=None) object cannot be deduped.
        ds = Dataset(name="d")
        anon = EvalCase(input="a")
        ds.upsert(anon)
        ds.upsert(anon)
        assert len(ds) == 2


class TestDatasetOrderingAndAccess:
    def test_iter_is_insertion_order(self):
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="z"))
        ds.upsert(EvalCase(input=2, id="a"))
        ds.upsert(EvalCase(input=3, id="m"))
        assert [c.input for c in ds] == [1, 2, 3]

    def test_replace_keeps_original_position(self):
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="z"))
        ds.upsert(EvalCase(input=2, id="a"))
        ds.upsert(EvalCase(input=99, id="z"))  # replace first
        assert [c.input for c in ds] == [99, 2]

    def test_len(self):
        ds = Dataset(name="d")
        assert len(ds) == 0
        ds.upsert(EvalCase(input=1, id="a"))
        assert len(ds) == 1

    def test_get_miss_returns_none(self):
        ds = Dataset(name="d")
        assert ds.get("nope") is None


class TestDatasetToDict:
    def test_to_dict_is_json_serializable(self):
        ds = Dataset(name="billing")
        ds.upsert(
            EvalCase(input={"m": "hi"}, id="a", expected={"r": "billing"}, metadata={"c": "x"})
        )
        d = ds.to_dict()
        assert d["name"] == "billing"
        # round-trips through json without error
        s = json.dumps(d)
        assert "billing" in s
        assert d["cases"][0]["id"] == "a"

    def test_to_dict_includes_provenance(self):
        ds = Dataset(name="d")
        ds.upsert(EvalCase(input=1, id="a", source_trace_id="t1", source_span_id="s1"))
        case_dict = ds.to_dict()["cases"][0]
        assert case_dict["source_trace_id"] == "t1"
        assert case_dict["source_span_id"] == "s1"
