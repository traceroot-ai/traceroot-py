"""``evaluate()``'s auto-publish must never stop to ask.

A versioning decision cannot gate an eval run. When a locally-authored ``Dataset`` already exists
on the platform and its content CHANGED, ``evaluate()`` publishes the new version silently, so the
run pins exactly what it scored. The explicit, user-initiated ``Dataset.push()`` KEEPS the
confirmation -- that is where deliberate version management lives. Unchanged content is an
idempotent no-op on both paths. Parity with traceroot-ts/tests/eval-evaluate-autopublish.test.ts.
"""

from __future__ import annotations

import pytest

import traceroot.eval.dataset_sync as sync_mod
import traceroot.eval.engine as engine
import traceroot.eval.platform as platform
from traceroot.eval import Dataset, evaluate
from traceroot.eval.dataset_sync import DatasetPublishAborted, PlatformDatasetSync
from traceroot.eval.transport import FakeTransport


def _dataset() -> Dataset:
    d = Dataset("auto")
    d.add(input={"m": 1}, expected={"m": 1})
    return d


def _existing_sync(published_revision: str):
    """A ``PlatformDatasetSync`` whose remote ALREADY has ``dsv_9`` published at
    ``published_revision`` -- so a differing local revision is a real content change."""
    sync = PlatformDatasetSync.__new__(PlatformDatasetSync)
    sync.api_key = "k"
    sync.host_url = "https://h"
    sync.calls = []

    def _request(method, path, body=None):
        sync.calls.append((method, path, body))
        if method == "GET":
            return {"name": "auto", "current_dataset_version_id": "dsv_9"}
        if path.endswith("/versions"):
            return {"dataset_version_id": "dsv_10", "version_number": 2}
        return {}

    sync._request = _request
    sync._published_revision = lambda ds, v: published_revision
    return sync


def _published(sync) -> bool:
    return any(p.endswith("/versions") for _m, p, _b in sync.calls)


@pytest.fixture
def declining_prompt(monkeypatch):
    """Stands in for an interactive TTY that says no. Reaching it at all is the failure."""
    seen: list[dict] = []

    def _decline(info):
        seen.append(info)
        return False

    monkeypatch.setattr(sync_mod, "_confirm_new_version", _decline)
    return seen


@pytest.fixture
def wired(monkeypatch):
    """Credentials + a reporting transport, so ``evaluate()`` reaches the auto-publish."""
    monkeypatch.setattr(platform, "_resolve_credentials", lambda a, b: ("k", "https://h"))
    monkeypatch.setattr(engine, "_auto_transport", lambda *a, **k: FakeTransport())


class TestEvaluateNeverPrompts:
    @pytest.mark.no_default_transport
    def test_changed_existing_dataset_publishes_without_prompting(
        self, monkeypatch, wired, declining_prompt
    ):
        d = _dataset()
        sync = _existing_sync("rev_from_an_older_push")
        monkeypatch.setattr(sync_mod, "PlatformDatasetSync", lambda *a, **k: sync)

        result = evaluate(name="r", data=d, task=lambda x: x, scorers=[lambda ctx: 1.0])

        assert declining_prompt == []  # the confirmer was never consulted
        assert _published(sync)  # ...and the changed content really was versioned
        assert d.dataset_version_id == "dsv_10"
        assert d.base_version_id == "dsv_10"
        # The run pins exactly what it scored: the version the auto-publish just created.
        assert result.dataset.dataset_version_id == "dsv_10"

    @pytest.mark.no_default_transport
    def test_unchanged_content_is_a_silent_no_op(self, monkeypatch, wired, declining_prompt):
        d = _dataset()
        sync = _existing_sync(d.snapshot().revision)
        monkeypatch.setattr(sync_mod, "PlatformDatasetSync", lambda *a, **k: sync)

        result = evaluate(name="r", data=d, task=lambda x: x, scorers=[lambda ctx: 1.0])

        assert declining_prompt == []
        assert not _published(sync)  # no new version for identical content
        assert result.dataset.dataset_version_id == "dsv_9"


class TestExplicitPushStillConfirms:
    """The deliberate publish boundary is unchanged: a direct ``push()`` asks, and a decline
    aborts without touching the remote."""

    def test_declined_push_aborts(self, declining_prompt):
        sync = _existing_sync("rev_from_an_older_push")
        with pytest.raises(DatasetPublishAborted):
            _dataset().push(sync)
        assert len(declining_prompt) == 1
        assert not _published(sync)

    def test_confirmed_push_publishes(self, monkeypatch):
        approved: list[dict] = []
        monkeypatch.setattr(
            sync_mod, "_confirm_new_version", lambda info: bool(approved.append(info)) or True
        )
        sync = _existing_sync("rev_from_an_older_push")
        d = _dataset()
        assert d.push(sync).dataset_version_id == "dsv_10"
        assert len(approved) == 1

    def test_unchanged_content_never_asks(self, declining_prompt):
        d = _dataset()
        sync = _existing_sync(d.snapshot().revision)
        assert d.push(sync).dataset_version_id == "dsv_9"
        assert declining_prompt == []
        assert not _published(sync)
