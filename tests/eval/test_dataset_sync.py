"""DS-4: explicit dataset publish + run upload + optimistic-concurrency conflict.

Remote dataset endpoints do not exist yet (see contract delta A2/A4); these use
the fake sync transport. Real wiring is a base-URL/impl swap once the server ships.
"""

import pytest

from traceroot.eval import Dataset, Score
from traceroot.eval.dataset_sync import (
    DatasetConflictError,
    FakeDatasetSync,
    LocalDatasetSync,
)
from traceroot.eval.platform import PlatformTransport
from traceroot.eval.results import EvalItemResult, EvalRunResult, RunDatasetRef, UploadState


def _ds():
    ds = Dataset("billing", description="x")
    ds.add(input={"m": "charge"}, id="tc0", expected={"r": "billing"})
    ds.add(input={"m": "hello"}, id="tc1")
    return ds


class TestPushLocalDefault:
    def test_default_push_is_local_only_no_network(self, respx_mock):
        result = _ds().push()  # no transport -> LocalDatasetSync, no HTTP
        assert result.status == "local_only"
        assert result.dataset_version_id is None

    def test_construction_and_mutation_do_not_publish(self):
        sync = FakeDatasetSync()
        ds = _ds()
        ds.update("tc0", expected={"r": "changed"})
        ds.archive("tc1")
        # nothing published until an explicit push
        assert sync.pushes == []


class TestPushFake:
    def test_push_creates_one_immutable_version(self):
        sync = FakeDatasetSync()
        ds = _ds()
        result = ds.push(sync)
        assert result.status == "uploaded"
        assert result.dataset_version_id is not None
        assert result.version_number == 1
        assert len(sync.pushes) == 1  # one push -> one version
        assert ds.dataset_version_id == result.dataset_version_id  # dataset now pinned

    def test_retry_same_base_is_idempotent(self):
        sync = FakeDatasetSync()
        ds = _ds()
        r1 = ds.push(sync)
        r2 = ds.push(sync, base_version_id=r1.dataset_version_id)  # unchanged content
        assert r2.dataset_version_id == r1.dataset_version_id  # no duplicate version

    def test_stale_base_raises_conflict(self):
        sync = FakeDatasetSync()
        ds = _ds()
        ds.push(sync)  # -> v1, dataset pinned to v1
        # someone else advances the remote to v2
        sync.force_current_version("dsv_other")
        ds.add(input={"m": "new"}, id="tc2")
        with pytest.raises(DatasetConflictError) as exc:
            ds.push(sync, base_version_id="dsv_stale")
        assert exc.value.current_version_id == "dsv_other"


class _StubPlatform(PlatformTransport):
    """PlatformTransport with the HTTP seam replaced by a recorder (returns uploaded)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.paths: list[str] = []

    def _request(self, method, path, body=None):
        self.paths.append(path)
        if path == "/api/v1/public/evaluation-runs":
            return {"evaluation_run_id": "run_1"}
        if path.endswith("/complete"):
            return {"status": body["status"]}
        return {"evaluation_result_id": "r"}


class TestRunUpload:
    def _run(self):
        items = [
            EvalItemResult(
                "tc0",
                {"m": "charge"},
                {"r": "billing"},
                {"r": "billing"},
                [Score("acc", 1.0)],
                {},
                None,
                "trace-1",
            ),
            EvalItemResult(
                "tc1",
                {"m": "hello"},
                {"r": "general"},
                {"r": "billing"},
                [Score("acc", 0.0)],
                {},
                None,
                None,
            ),
        ]
        return EvalRunResult(
            name="routing",
            item_results=items,
            score_summary={},
            upload_state=UploadState(),
            local_run_id="run_local_1",
            candidate_version="git:abc",
            dataset=RunDatasetRef("ds_1", "rev_x", "dsv_1", 2),
        )

    def test_upload_replays_results_and_marks_uploaded(self):
        run = self._run()
        transport = _StubPlatform(
            "ds_1", scorer_names=["acc"], api_key="tr-x", host_url="https://h"
        )
        out = run.upload(transport)
        assert out.upload_state.status == "uploaded"
        assert out.run_id == "run_1"
        # created run, upserted both results (preserving tc ids), completed
        assert transport.paths[0] == "/api/v1/public/evaluation-runs"
        assert sum(1 for p in transport.paths if p.endswith("/results")) == 2
        assert transport.paths[-1].endswith("/complete")

    def test_upload_without_transport_or_creds_raises(self, monkeypatch):
        import traceroot

        # get_client() re-initializes a client from the env when _client is None, so an ambient
        # TRACEROOT_API_KEY would make this pass for the wrong reason. Clear it to genuinely
        # exercise the no-credentials path.
        monkeypatch.delenv("TRACEROOT_API_KEY", raising=False)
        traceroot.shutdown()
        traceroot._client = None
        with pytest.raises((ValueError, RuntimeError)):
            self._run().upload()


def test_local_sync_is_local_only():
    result = LocalDatasetSync().push_dataset(_ds().snapshot(), None)
    assert result.status == "local_only"


class TestPlatformSyncGuards:
    class _StubSync:
        def __init__(self):
            from traceroot.eval.dataset_sync import PlatformDatasetSync

            self.inner = PlatformDatasetSync.__new__(PlatformDatasetSync)
            self.inner.host_url = "https://h"
            self.inner.api_key = "tr-x"
            self.inner._request = lambda *a, **k: {}  # no network

        def push(self, snapshot, base=None):
            return self.inner.push_dataset(snapshot, base)

    def test_empty_dataset_raises_not_network(self):
        from traceroot.eval import Dataset

        empty = Dataset("d")  # no active cases -> no changes -> backend requires >=1
        with pytest.raises(ValueError, match="no active cases"):
            self._StubSync().push(empty.snapshot(), None)


class TestExistingDatasetConfirmation:
    """Re-pushing an existing dataset name adds a NEW version to the SAME dataset; before doing so
    the SDK double-checks (default: interactive prompt; overridable via on_existing)."""

    def _sync(self, exists: bool, published_rev: str | None = "rev_different"):
        from traceroot.eval.dataset_sync import PlatformDatasetSync

        sync = PlatformDatasetSync.__new__(PlatformDatasetSync)
        sync.api_key = "k"
        sync.host_url = "https://h"
        sync.calls: list = []

        def _request(method, path, body=None):
            sync.calls.append((method, path))
            if method == "GET":
                if exists:
                    return {"name": "d", "current_dataset_version_id": "dsv_9"}
                raise RuntimeError("GET .../datasets/x -> HTTP 404: not found")
            if path.endswith("/versions"):
                return {"dataset_version_id": "dsv_10", "version_number": 2}
            return {}

        sync._request = _request  # type: ignore[assignment]
        # Stub the published-revision fetch: `published_rev` drives changed-vs-unchanged detection.
        sync._published_revision = lambda ds, v: published_rev  # type: ignore[assignment]
        return sync

    def _snap(self):
        d = Dataset("d")
        d.add(input={"q": "a"})
        return d.snapshot()

    def test_unchanged_content_is_a_noop_without_prompting(self):
        seen = {"called": False}

        def confirm(info):
            seen["called"] = True
            return True

        snap = self._snap()
        sync = self._sync(exists=True, published_rev=snap.revision)  # content matches current version
        res = sync.push_dataset(snap, None, on_existing=confirm)
        assert res.dataset_version_id == "dsv_9"  # reuses the current version
        assert seen["called"] is False  # unchanged -> never prompts
        assert not any(p.endswith("/versions") for _m, p in sync.calls)  # no new version published

    def test_existing_declined_aborts_without_publishing(self):
        from traceroot.eval.dataset_sync import DatasetPublishAborted

        sync = self._sync(exists=True)  # published_rev differs -> content changed
        with pytest.raises(DatasetPublishAborted):
            sync.push_dataset(self._snap(), None, on_existing=lambda info: False)
        # NO version was published
        assert not any(p.endswith("/versions") for _m, p in sync.calls)

    def test_existing_accepted_publishes_new_version(self):
        sync = self._sync(exists=True)
        res = sync.push_dataset(self._snap(), None, on_existing=lambda info: True)
        assert res.dataset_version_id == "dsv_10"
        assert any(p.endswith("/versions") for _m, p in sync.calls)

    def test_new_dataset_does_not_prompt(self):
        seen = {"called": False}

        def confirm(info):
            seen["called"] = True
            return True

        sync = self._sync(exists=False)
        sync.push_dataset(self._snap(), None, on_existing=confirm)
        assert seen["called"] is False  # 404 -> new dataset -> confirmation is never invoked

    def test_default_confirmer_proceeds_when_non_interactive(self, monkeypatch):
        from traceroot.eval.dataset_sync import _confirm_new_version

        monkeypatch.delenv("TRACEROOT_ASSUME_YES", raising=False)

        class _FakeStdin:
            def isatty(self):
                return False

        monkeypatch.setattr("sys.stdin", _FakeStdin())
        assert _confirm_new_version({"name": "d", "current_dataset_version_id": "dsv_9"}) is True


def test_evaluate_auto_syncs_a_local_dataset(monkeypatch):
    """evaluate() provisions a locally-authored Dataset automatically (no manual push/ensure_synced):
    when it has no server version and credentials resolve, it pushes and attaches the version."""
    from traceroot.eval import Dataset, evaluate
    import traceroot.eval.platform as platform
    import traceroot.eval.dataset_sync as ds

    monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
    pushed = {"n": 0}

    class _FakeSync:
        def push_dataset(self, snapshot, base, *, on_existing=None):
            pushed["n"] += 1
            return ds.PushResult("uploaded", snapshot.dataset_id, "dsv_1")

    monkeypatch.setattr(ds, "PlatformDatasetSync", lambda *a, **k: _FakeSync())

    d = Dataset("auto")
    d.add(input={"m": 1})
    assert d.dataset_version_id is None  # local, unsynced
    evaluate(name="r", dataset=d, task=lambda x: x, scorers=[lambda ctx: 1.0])
    assert pushed["n"] == 1  # auto-synced inside evaluate()
    assert d.dataset_version_id == "dsv_1"  # version attached
