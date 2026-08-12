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

    def test_anonymous_gets_stable_content_id(self):
        # DS-1: anonymous cases get a stable content-derived tc_ id (not positional case-{n}),
        # so ids survive serialization and remote push. Content-derived, not random: a ULID here
        # would fork the same case into a new server case on every process.
        ds = Dataset(name="d")
        r0 = ds.upsert(EvalCase(input="a"))
        r1 = ds.upsert(EvalCase(input="b"))
        assert r0.id.startswith("tc_") and r1.id.startswith("tc_")
        assert r0.id != r1.id
        assert len(ds) == 2
        assert Dataset(name="d").upsert(EvalCase(input="a")).id == r0.id

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


class TestContentBasedCaseIds:
    """Case ids are derived from the case INPUT (not position): inserting/removing/reordering
    other cases must not shift a case's id, and duplicate inputs stay collision-free."""

    def test_insert_does_not_shift_other_case_ids(self):
        a = Dataset("k")
        a.add(input={"q": "a"})
        a.add(input={"q": "b"})
        base = {c.input["q"]: c.id for c in a.cases()}

        b = Dataset("k")
        b.add(input={"q": "z"})  # a new case inserted BEFORE a and b
        b.add(input={"q": "a"})
        b.add(input={"q": "b"})
        after = {c.input["q"]: c.id for c in b.cases()}

        assert after["a"] == base["a"]  # unchanged despite z inserted first (positional would shift)
        assert after["b"] == base["b"]

    def test_reorder_yields_same_revision(self):
        a = Dataset("k")
        a.add(input=1)
        a.add(input=2)
        b = Dataset("k")
        b.add(input=2)
        b.add(input=1)
        assert a.snapshot().revision == b.snapshot().revision  # order-independent content address

    def test_duplicate_inputs_get_distinct_ids_collision_safe(self):
        d = Dataset("k")
        x0 = d.add(input={"q": "x"})
        x1 = d.add(input={"q": "x"})
        assert x0.id != x1.id
        # removing the first then re-adding the same input must not collide with the survivor
        d.remove(x0.id)
        x2 = d.add(input={"q": "x"})
        assert x2.id not in (x1.id,)

    def test_editing_expected_keeps_case_id(self):
        d = Dataset("k")
        c = d.add(input={"q": "a"}, expected="old")
        d2 = Dataset("k")
        c2 = d2.add(input={"q": "a"}, expected="new")
        assert c.id == c2.id  # identity is the input; editing expected is the SAME case (an update)
