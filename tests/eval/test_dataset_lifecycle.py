"""DS-1: local dataset lifecycle + immutable snapshot (network-free)."""

import dataclasses

import pytest

from traceroot.eval import Dataset, DatasetSnapshot, EvalCase


class TestDatasetCreation:
    def test_generates_dataset_id_and_stores_description(self):
        ds = Dataset(name="billing", description="routing set")
        assert ds.name == "billing"
        assert ds.description == "routing set"
        assert ds.dataset_id.startswith("ds_")

    def test_dataset_ids_are_unique(self):
        assert Dataset("a").dataset_id != Dataset("b").dataset_id


class TestAdd:
    def test_add_generates_stable_tc_id(self):
        ds = Dataset("d")
        case = ds.add(input={"m": 1}, expected={"r": 1})
        assert case.id.startswith("tc_")
        assert ds.get(case.id).input == {"m": 1}

    def test_add_with_explicit_id(self):
        ds = Dataset("d")
        case = ds.add(input=1, id="tc-custom")
        assert case.id == "tc-custom"

    def test_add_duplicate_id_raises(self):
        ds = Dataset("d")
        ds.add(input=1, id="a")
        with pytest.raises(ValueError):
            ds.add(input=2, id="a")

    def test_add_carries_all_fields(self):
        ds = Dataset("d")
        c = ds.add(
            input=1, expected=2, metadata={"k": "v"}, source_trace_id="t", source_span_id="s"
        )
        assert (c.expected, c.metadata, c.source_trace_id, c.source_span_id) == (
            2,
            {"k": "v"},
            "t",
            "s",
        )


class TestUpsertUpdate:
    def test_upsert_replaces(self):
        ds = Dataset("d")
        ds.upsert(EvalCase(input=1, id="a"))
        ds.upsert(EvalCase(input=2, id="a"))
        assert len(ds) == 1
        assert ds.get("a").input == 2

    def test_upsert_anonymous_gets_ulid(self):
        ds = Dataset("d")
        c = ds.upsert(EvalCase(input=1))
        assert c.id.startswith("tc_")

    def test_update_changes_fields(self):
        ds = Dataset("d")
        ds.add(input=1, id="a", expected=1)
        updated = ds.update("a", expected=99, metadata={"x": 1})
        assert updated.expected == 99
        assert updated.metadata == {"x": 1}
        assert ds.get("a").input == 1  # unchanged field preserved

    def test_update_missing_raises(self):
        with pytest.raises(KeyError):
            Dataset("d").update("nope", expected=1)


class TestArchiveRemove:
    def test_archive_excludes_from_active(self):
        ds = Dataset("d")
        ds.add(input=1, id="a")
        ds.add(input=2, id="b")
        ds.archive("a")
        assert len(ds) == 1
        assert [c.id for c in ds] == ["b"]
        assert ds.get("a") is not None  # still retrievable
        assert ds.get("a").archived is True

    def test_cases_include_archived(self):
        ds = Dataset("d")
        ds.add(input=1, id="a")
        ds.archive("a")
        assert len(ds.cases(include_archived=True)) == 1
        assert ds.cases() == []

    def test_remove_deletes(self):
        ds = Dataset("d")
        ds.add(input=1, id="a")
        ds.remove("a")
        assert ds.get("a") is None
        assert len(ds) == 0

    def test_remove_missing_raises(self):
        with pytest.raises(KeyError):
            Dataset("d").remove("nope")


class TestSnapshot:
    def _ds(self):
        ds = Dataset("d", description="x")
        ds.add(input=1, id="a", expected=1)
        ds.add(input=2, id="b", expected=2)
        return ds

    def test_snapshot_is_immutable_and_typed(self):
        snap = self._ds().snapshot()
        assert isinstance(snap, DatasetSnapshot)
        assert snap.revision.startswith("rev_")
        assert len(snap) == 2
        with pytest.raises(dataclasses.FrozenInstanceError):
            snap.cases = ()  # frozen

    def test_snapshot_excludes_archived(self):
        ds = self._ds()
        ds.archive("a")
        assert len(ds.snapshot()) == 1

    def test_mutating_dataset_does_not_change_snapshot(self):
        ds = self._ds()
        snap = ds.snapshot()
        ds.update("a", expected=999)
        ds.add(input=3, id="c")
        assert len(snap) == 2  # snapshot unchanged
        snap_a = next(c for c in snap if c.id == "a")
        assert snap_a.expected == 1  # original value

    def test_identical_content_same_revision(self):
        assert self._ds().snapshot().revision == self._ds().snapshot().revision

    def test_different_content_different_revision(self):
        ds1 = self._ds()
        ds2 = self._ds()
        ds2.update("a", expected=42)
        assert ds1.snapshot().revision != ds2.snapshot().revision

    def test_archived_toggle_changes_revision(self):
        ds1 = self._ds()
        ds2 = self._ds()
        ds2.archive("a")
        assert ds1.snapshot().revision != ds2.snapshot().revision


class TestNoNetwork:
    def test_local_ops_make_no_http(self, respx_mock):
        # respx_mock with zero routes -> any HTTP call raises. All local ops below
        # must make none.
        ds = Dataset("d", description="x")
        ds.add(input=1, id="a", expected=1)
        ds.upsert(EvalCase(input=2, id="b"))
        ds.update("a", expected=2)
        ds.archive("b")
        ds.remove("b")
        snap = ds.snapshot()
        _ = (ds.to_dict(), snap.to_dict(), list(ds), len(ds))
