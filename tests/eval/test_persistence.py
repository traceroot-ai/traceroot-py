"""DS-1: local dataset persistence (JSON / JSONL), stable ids through round-trip."""

import json

from traceroot.eval import Dataset


def _ds():
    ds = Dataset("billing", description="routing set")
    ds.add(
        input={"m": "hi"},
        id="tc_keep",
        expected={"r": "billing"},
        metadata={"suite": "smoke"},
        source_trace_id="t1",
        source_span_id="s1",
    )
    ds.add(input={"m": "bye"}, id="tc_arch")
    ds.archive("tc_arch")
    return ds


class TestDictRoundTrip:
    def test_to_dict_from_dict_preserves_ids_and_archived(self):
        ds = _ds()
        restored = Dataset.from_dict(ds.to_dict())
        assert restored.dataset_id == ds.dataset_id
        assert restored.name == "billing"
        assert restored.description == "routing set"
        assert restored.get("tc_keep").source_trace_id == "t1"
        assert restored.get("tc_arch").archived is True
        assert len(restored) == 1  # active only
        assert len(restored.cases(include_archived=True)) == 2

    def test_to_dict_is_json_serializable(self):
        json.dumps(_ds().to_dict())


class TestJson:
    def test_json_str_round_trip(self):
        ds = _ds()
        restored = Dataset.from_json(ds.to_json())
        assert restored.dataset_id == ds.dataset_id
        assert restored.get("tc_keep").expected == {"r": "billing"}


class TestSaveLoad:
    def test_save_load_json(self, tmp_path):
        ds = _ds()
        p = tmp_path / "d.json"
        ds.save(str(p))
        restored = Dataset.load(str(p))
        assert restored.dataset_id == ds.dataset_id
        assert [c.id for c in restored.cases(include_archived=True)] == ["tc_keep", "tc_arch"]

    def test_save_load_jsonl(self, tmp_path):
        ds = _ds()
        p = tmp_path / "d.jsonl"
        ds.save(str(p))
        # header line + one line per case
        lines = p.read_text().strip().splitlines()
        assert json.loads(lines[0])["type"] == "dataset"
        assert json.loads(lines[1])["type"] == "case"
        restored = Dataset.load(str(p))
        assert restored.dataset_id == ds.dataset_id
        assert restored.get("tc_arch").archived is True

    def test_revision_stable_through_save_load(self, tmp_path):
        ds = _ds()
        before = ds.snapshot().revision
        p = tmp_path / "d.jsonl"
        ds.save(str(p))
        after = Dataset.load(str(p)).snapshot().revision
        assert before == after  # ids + content preserved -> same content revision
