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

    def test_upload_without_transport_or_creds_raises(self):
        import traceroot

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
