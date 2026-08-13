"""Dataset ergonomics: the paths where a dataset silently loses its identity or its edit.

Each test here pins a behaviour a user relies on but that no fixture exercised: an existence
check that only treats a real 404 as "new", an anonymous ``upsert`` that converges like ``add``,
an accurate cloud-only error, a metadata-only push that is not a no-op, and a dataset ``key``
that survives save/load and pull. Parity with
traceroot-ts/tests/eval-dataset-ergonomics.test.ts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval.parity_vectors import DATASET_KEY_FIXTURE
from traceroot.eval import Dataset, EvalCase
from traceroot.eval.dataset_sync import PlatformDatasetSync
from traceroot.eval.ids import stable_dataset_id


def _snap(name: str = "d") -> object:
    d = Dataset(name)
    d.add(input={"q": "a"})
    return d.snapshot()


def _sync(on_get):
    """A PlatformDatasetSync whose HTTP seam is `on_get` for GETs and a canned publish otherwise."""
    sync = PlatformDatasetSync.__new__(PlatformDatasetSync)
    sync.api_key = "k"
    sync.host_url = "https://h"
    sync.calls = []

    def _request(method, path, body=None):
        sync.calls.append((method, path, body))
        if method == "GET":
            return on_get(path)
        if path.endswith("/versions"):
            return {"dataset_version_id": "dsv_10", "version_number": 2}
        return {}

    sync._request = _request
    return sync


class TestExistenceCheckErrors:
    """A failed existence GET must not be read as 'this dataset does not exist'.

    Swallowing every error there skips BOTH the idempotent no-op and the confirmation prompt,
    so an expired key / 500 / timeout publishes an unprompted new version.
    """

    def test_non_404_propagates_instead_of_becoming_a_new_dataset(self):
        def boom(path):
            raise RuntimeError("GET https://h/... -> HTTP 500: upstream exploded")

        sync = _sync(boom)
        with pytest.raises(RuntimeError, match="HTTP 500"):
            sync.push_dataset(_snap(), None, on_existing=lambda info: True)
        # And nothing was published on the way out.
        assert not any(p.endswith("/versions") for _m, p, _b in sync.calls)

    def test_auth_failure_propagates(self):
        def unauthorized(path):
            raise RuntimeError("GET https://h/... -> HTTP 401: bad key")

        with pytest.raises(RuntimeError, match="HTTP 401"):
            _sync(unauthorized).push_dataset(_snap(), None)

    def test_404_still_means_new(self):
        seen = {"confirmed": False}

        def missing(path):
            raise RuntimeError("GET https://h/... -> HTTP 404: not found")

        def confirm(info):
            seen["confirmed"] = True
            return True

        sync = _sync(missing)
        res = sync.push_dataset(_snap(), None, on_existing=confirm)
        assert res.dataset_version_id == "dsv_10"
        assert seen["confirmed"] is False  # a genuinely new dataset never prompts


class TestAnonymousUpsertConverges:
    """``upsert`` of an id-less case must converge exactly like ``add``.

    A random ULID makes the same case fork into a new server case on every process, which is
    the one thing content-addressed ids exist to prevent.
    """

    def test_two_processes_upserting_the_same_input_agree(self):
        a = Dataset("billing").upsert(EvalCase(input={"m": "hi"}))
        b = Dataset("billing").upsert(EvalCase(input={"m": "hi"}))
        assert a.id == b.id

    def test_anonymous_upsert_matches_add(self):
        added = Dataset("billing").add(input={"m": "hi"})
        upserted = Dataset("billing").upsert(EvalCase(input={"m": "hi"}))
        assert upserted.id == added.id

    def test_duplicate_inputs_get_distinct_occurrence_ids(self):
        d = Dataset("billing")
        first = d.upsert(EvalCase(input={"m": "hi"}))
        second = d.upsert(EvalCase(input={"m": "hi"}))
        assert first.id != second.id
        assert len(d) == 2

    def test_an_explicit_id_still_wins(self):
        c = Dataset("billing").upsert(EvalCase(input={"m": "hi"}, id="c0"))
        assert c.id == "c0"

    def test_key_scopes_the_id(self):
        a = Dataset("billing", key="k1").upsert(EvalCase(input={"m": "hi"}))
        b = Dataset("billing", key="k2").upsert(EvalCase(input={"m": "hi"}))
        assert a.id != b.id


class TestAutoProvisionPinsTheBaseVersion:
    """``evaluate()``'s auto-provision must leave the dataset FULLY pinned, not just versioned:
    ``base_version_id`` is the optimistic-concurrency base for the NEXT push, so leaving it None
    makes a later explicit ``push()`` send ``base_version_id: null`` against a dataset that now
    has a version -- a spurious conflict."""

    @pytest.mark.no_default_transport
    def test_base_version_is_advanced(self, monkeypatch):
        import traceroot.eval.dataset_sync as sync_mod
        import traceroot.eval.engine as engine
        import traceroot.eval.platform as platform
        from traceroot.eval import evaluate
        from traceroot.eval.transport import FakeTransport

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))

        class _FakeSync:
            def push_dataset(self, snapshot, base, *, on_existing=None):
                return sync_mod.PushResult("uploaded", snapshot.dataset_id, "dsv_1")

        monkeypatch.setattr(sync_mod, "PlatformDatasetSync", lambda *a, **k: _FakeSync())
        monkeypatch.setattr(engine, "_auto_transport", lambda *a, **k: FakeTransport())

        d = Dataset("auto")
        d.add(input={"m": 1})
        evaluate(name="r", dataset=d, task=lambda x: x, scorers=[lambda ctx: 1.0])

        assert d.dataset_version_id == "dsv_1"
        assert d.base_version_id == "dsv_1"


class TestCloudOnlyErrorIsAccurate:
    """ "No credentials" must only be said when there are no credentials."""

    @pytest.mark.no_default_transport
    def test_credentials_present_but_data_is_not_synced(self, monkeypatch):
        import traceroot.eval.platform as platform
        from traceroot.eval import evaluate

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
        with pytest.raises(RuntimeError) as exc:
            evaluate(
                name="r",
                data=[EvalCase(input=1, id="c0", expected=1)],
                task=lambda x: x,
                scorers=[lambda ctx: 1.0],
            )
        message = str(exc.value)
        assert "no credentials" not in message.lower()
        assert "synced dataset" in message

    @pytest.mark.no_default_transport
    def test_no_credentials_says_so(self, monkeypatch):
        import traceroot.eval.platform as platform
        from traceroot.eval import evaluate

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("", "https://h"))
        d = Dataset("d")
        d.add(input=1, expected=1)
        with pytest.raises(RuntimeError, match="no credentials were found"):
            evaluate(name="r", data=d, task=lambda x: x, scorers=[lambda ctx: 1.0])


class TestMetadataOnlyPush:
    """A name/description edit is not part of the content revision, so the unchanged-revision
    short circuit must still carry it to the platform rather than drop it silently."""

    def _existing(self, path):
        return {"name": "old-name", "current_dataset_version_id": "dsv_9"}

    def test_description_change_reaches_the_platform(self):
        d = Dataset("d")
        d.add(input={"q": "a"})
        snap_before = d.snapshot()
        sync = _sync(self._existing)
        sync._published_revision = lambda ds, v: snap_before.revision  # content unchanged

        d.description = "now documented"
        res = sync.push_dataset(d.snapshot(), None)

        upserts = [b for m, p, b in sync.calls if m == "POST" and p.endswith("/datasets")]
        assert upserts, "the name/description upsert must be sent even when content is unchanged"
        assert upserts[0]["description"] == "now documented"
        # Still an idempotent no-op version-wise: no new version was published.
        assert res.dataset_version_id == "dsv_9"
        assert not any(p.endswith("/versions") for _m, p, _b in sync.calls)

    def test_declined_publish_mutates_nothing(self):
        from traceroot.eval.dataset_sync import DatasetPublishAborted

        sync = _sync(self._existing)
        sync._published_revision = lambda ds, v: "rev_something_else"  # content changed
        with pytest.raises(DatasetPublishAborted):
            sync.push_dataset(_snap(), None, on_existing=lambda info: False)
        assert not any(m == "POST" for m, _p, _b in sync.calls)


class TestDatasetKeySurvives:
    """``key`` is what every case id is derived from, so losing it silently de-converges every
    case added to a reloaded or pulled dataset."""

    def test_to_dict_round_trip_preserves_key(self):
        d = Dataset("Billing v2 (renamed)", key="billing")
        assert Dataset.from_dict(d.to_dict()).key == "billing"

    def test_case_added_after_reload_matches_local_authoring(self, tmp_path: Path):
        d = Dataset("Billing v2 (renamed)", key="billing")
        d.add(input={"m": "first"})
        path = str(tmp_path / "ds.json")
        d.save(path)

        reloaded = Dataset.load(path)
        assert reloaded.add(input={"m": "second"}).id == d.add(input={"m": "second"}).id

    def test_jsonl_round_trip_preserves_key(self, tmp_path: Path):
        d = Dataset("Billing v2 (renamed)", key="billing")
        d.add(input={"m": "first"})
        path = str(tmp_path / "ds.jsonl")
        d.save(path)
        header = json.loads(Path(path).read_text(encoding="utf-8").splitlines()[0])
        assert header["key"] == "billing"
        assert Dataset.load(path).key == "billing"

    def test_pull_recovers_the_key_from_the_dataset_name(self, monkeypatch):
        import traceroot.eval.platform as platform

        local = Dataset("billing")
        first = local.add(input={"m": "first"})

        def fake_get(url, key):
            if "/dataset-versions/" in url:
                return {
                    "dataset_id": local.dataset_id,
                    "dataset_version_id": "dsv_1",
                    "items": [{"test_case_id": first.id, "input": {"m": "first"}}],
                }
            return {"name": "billing", "current_dataset_version_id": "dsv_1"}

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
        monkeypatch.setattr(platform, "_http_get_json", fake_get)

        pulled = platform.pull_dataset(local.dataset_id)
        assert pulled.key == "billing"
        # The pulled dataset extends exactly like the local one.
        assert pulled.add(input={"m": "second"}).id == local.add(input={"m": "second"}).id

    def test_pull_prefers_the_key_echoed_by_the_platform(self, monkeypatch):
        """The dataset response now CARRIES the key, so a renamed dataset is no longer a dead end:
        the echoed key is adopted verbatim and every case added afterwards converges with local
        authoring. The ids are pinned by the shared fixture, so Python and TypeScript agree."""
        import traceroot.eval.platform as platform

        fix = DATASET_KEY_FIXTURE
        local = Dataset(fix["display_name"], key=fix["dataset_key"])
        first = local.add(input=fix["existing_case"]["input"])
        assert local.dataset_id == fix["dataset_id"]
        assert first.id == fix["existing_case"]["id"]

        def fake_get(url, key):
            if "/dataset-versions/" in url:
                return {
                    "dataset_id": fix["dataset_id"],
                    "dataset_version_id": "dsv_1",
                    "items": [{"test_case_id": first.id, "input": fix["existing_case"]["input"]}],
                }
            return {
                "name": fix["display_name"],
                "key": fix["dataset_key"],
                "current_dataset_version_id": "dsv_1",
            }

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
        monkeypatch.setattr(platform, "_http_get_json", fake_get)

        pulled = platform.pull_dataset(fix["dataset_id"])
        assert pulled.key == fix["dataset_key"]
        added = pulled.add(input=fix["added_case"]["input"])
        assert added.id == fix["added_case"]["id"]
        assert added.id == local.add(input=fix["added_case"]["input"]).id

    def test_pull_with_no_current_version_raises_a_clear_error(self, monkeypatch):
        """A dataset that has never published a version has no current version to substitute.
        Without the guard a None version id reaches pull_dataset_version and fails obscurely; it
        must instead say what is actually wrong."""
        import traceroot.eval.platform as platform

        def fake_get(url, key):
            # The dataset exists but carries no current_dataset_version_id.
            return {"name": "billing"}

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
        monkeypatch.setattr(platform, "_http_get_json", fake_get)

        with pytest.raises(ValueError, match="no published version to pull"):
            platform.pull_dataset("ds_1")

    def test_pull_never_adopts_a_name_that_is_not_the_key(self, monkeypatch):
        """A display-name rename (key != name) must not be mistaken for the key: the id derived
        from it would be wrong. The dataset id is used instead -- stable, and never a lie."""
        import traceroot.eval.platform as platform

        local = Dataset("Billing v2 (renamed)", key="billing")

        def fake_get(url, key):
            if "/dataset-versions/" in url:
                return {
                    "dataset_id": local.dataset_id,
                    "dataset_version_id": "dsv_1",
                    "items": [{"test_case_id": "tc_x", "input": {"m": "first"}}],
                }
            return {"name": "Billing v2 (renamed)", "current_dataset_version_id": "dsv_1"}

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
        monkeypatch.setattr(platform, "_http_get_json", fake_get)

        pulled = platform.pull_dataset(local.dataset_id)
        assert pulled.key != "Billing v2 (renamed)"
        assert pulled.key == local.dataset_id  # the unambiguous fallback
        assert stable_dataset_id("Billing v2 (renamed)") != local.dataset_id  # name is not the key

    def test_pull_falls_back_to_the_heuristic_when_the_key_is_null(self, monkeypatch):
        """Datasets that predate the ``key`` contract (or were authored in the UI) echo
        ``key: null``, so the name-hash recovery must stay in place for them."""
        import traceroot.eval.platform as platform

        local = Dataset("billing")
        first = local.add(input={"m": "first"})

        def fake_get(url, key):
            if "/dataset-versions/" in url:
                return {
                    "dataset_id": local.dataset_id,
                    "dataset_version_id": "dsv_1",
                    "items": [{"test_case_id": first.id, "input": {"m": "first"}}],
                }
            return {"name": "billing", "key": None, "current_dataset_version_id": "dsv_1"}

        monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
        monkeypatch.setattr(platform, "_http_get_json", fake_get)

        assert platform.pull_dataset(local.dataset_id).key == "billing"


class TestDatasetKeyIsSentOnPush:
    """The key is the dataset's identity, so it belongs on the wire: without it the platform can
    only guess (name-hash), and a UI rename or an explicit key leaves every later case divergent."""

    def _new_dataset(self, path):
        raise RuntimeError("GET https://h/... -> HTTP 404: not found")

    def _upserts(self, calls):
        return [b for m, p, b in calls if m == "POST" and p.endswith("/datasets")]

    def test_push_sends_the_key(self):
        sync = _sync(self._new_dataset)
        sync.push_dataset(_snap("billing"), None)
        assert self._upserts(sync.calls)[0]["key"] == "billing"  # default key == name

    def test_push_sends_an_explicit_key(self):
        d = Dataset(DATASET_KEY_FIXTURE["display_name"], key=DATASET_KEY_FIXTURE["dataset_key"])
        d.add(input=DATASET_KEY_FIXTURE["existing_case"]["input"])
        sync = _sync(self._new_dataset)
        sync.push_dataset(d.snapshot(), None)
        body = self._upserts(sync.calls)[0]
        assert body["key"] == DATASET_KEY_FIXTURE["dataset_key"]
        assert body["name"] == DATASET_KEY_FIXTURE["display_name"]  # the key is NOT the name

    def test_metadata_only_push_sends_the_key_too(self):
        """The unchanged-revision short circuit sends the metadata upsert; it must carry the key
        as well, or the backfill never happens for an already-published dataset."""
        d = Dataset("Billing", key="billing")
        d.add(input={"q": "a"})
        sync = _sync(lambda path: {"name": "Billing", "current_dataset_version_id": "dsv_9"})
        sync._published_revision = lambda ds, v: d.snapshot().revision  # content unchanged
        sync.push_dataset(d.snapshot(), None)
        assert self._upserts(sync.calls)[0]["key"] == "billing"
