"""A published dataset re-hashes to the revision that published it.

``publishedRevision == snapshot.revision`` is the single predicate the whole idempotent-push
design rests on, and both suites used to stub it out wholesale. Here NOTHING in the SDK is
stubbed except the HTTP transport itself: push -> the captured wire payload -> a version-shaped
response -> ``pull_dataset_version`` -> ``_dataset_from_version`` -> recomputed revision. If the
value hashed is not the value sent, the second push publishes a no-change version and this fails.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from traceroot.eval import Dataset


class FakeBackend:
    """The three endpoints push/pull touch, backed by a dict. Every payload crosses a real
    ``json.dumps``/``json.loads`` so a value that cannot survive JSON fails here, not in prod."""

    def __init__(self) -> None:
        self.versions: dict[str, dict] = {}
        self.current: dict[str, str] = {}
        self.published = 0

    def __call__(self, method: str, url: str, api_key: str, body: dict | None = None) -> dict:
        path = url.split("//", 1)[-1].split("/", 1)[1]
        if method == "GET" and "/dataset-versions/" in path:
            return json.loads(json.dumps(self.versions[path.rsplit("/", 1)[-1]]))
        if method == "GET":
            dataset_id = path.rsplit("/", 1)[-1]
            if dataset_id not in self.current:
                raise RuntimeError("GET -> HTTP 404: not found")
            return {"name": "rt", "current_dataset_version_id": self.current[dataset_id]}
        if path.endswith("/versions"):
            dataset_id = path.split("/datasets/")[1].split("/")[0]
            self.published += 1
            version_id = f"dsv_{self.published}"
            # Store what the SDK actually sent, through a real JSON round trip.
            changes = json.loads(json.dumps(body))["changes"]
            self.versions[version_id] = {
                "dataset_id": dataset_id,
                "dataset_version_id": version_id,
                "items": changes,
            }
            self.current[dataset_id] = version_id
            return {"dataset_version_id": version_id, "version_number": self.published}
        return {}


@pytest.fixture()
def backend(monkeypatch):
    fake = FakeBackend()
    monkeypatch.setattr("traceroot.eval.platform._http_json", fake)
    return fake


def _sync():
    from traceroot.eval.dataset_sync import PlatformDatasetSync

    sync = PlatformDatasetSync.__new__(PlatformDatasetSync)
    sync.api_key = "k"
    sync.host_url = "https://h"
    return sync


def _dataset() -> Dataset:
    d = Dataset("roundtrip")
    d.add(
        input={"when": datetime(2020, 1, 1, 12, 0, tzinfo=UTC), "tags": {"b", "a"}},
        expected={"score": 1.0, "eps": 1e-07},
        metadata={"10": "x", "2": "y", "raw": b"hi"},
    )
    d.add(input={"plain": "case"})
    return d


def test_push_then_pull_recomputes_the_same_revision(backend):
    sync = _sync()
    local = _dataset()
    snapshot = local.snapshot()
    sync.push_dataset(snapshot, None)
    assert backend.published == 1

    pulled_revision = sync._published_revision(snapshot.dataset_id, "dsv_1")
    assert pulled_revision == snapshot.revision


def test_second_push_of_unchanged_content_is_a_noop_and_never_prompts(backend):
    """The user-visible consequence: a Date/Set/bytes case used to publish a brand-new
    version on every push, unprompted-and-unbounded, because the pull never matched."""
    sync = _sync()

    def refuse(_info):
        raise AssertionError("unchanged content must not prompt")

    sync.push_dataset(_dataset().snapshot(), None, on_existing=refuse)
    result = sync.push_dataset(_dataset().snapshot(), None, on_existing=refuse)

    assert backend.published == 1  # no second version
    assert result.dataset_version_id == "dsv_1"


def test_the_wire_payload_is_the_canonical_form_that_was_hashed(backend):
    sync = _sync()
    sync.push_dataset(_dataset().snapshot(), None)
    items = backend.versions["dsv_1"]["items"]
    sent = next(i for i in items if "when" in i["input"])
    assert sent["input"] == {"when": "2020-01-01T12:00:00.000Z", "tags": ["a", "b"]}
    assert sent["metadata"]["raw"] == [104, 105]


def test_save_load_round_trip_preserves_the_revision(tmp_path):
    """Same invariant on the local persistence path: a reloaded dataset must not look edited."""
    local = _dataset()
    path = str(tmp_path / "ds.json")
    local.save(path)
    assert Dataset.load(path).snapshot().revision == local.snapshot().revision
