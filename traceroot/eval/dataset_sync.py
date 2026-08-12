"""Explicit dataset publication seam (DS-4).

One explicit ``Dataset.push()`` = one immutable server dataset version, with
optimistic-concurrency conflict detection (``base_version_id``). Local mutations
never create versions; they accumulate until an explicit push.

The default is a no-op ``LocalDatasetSync`` (local-only); ``FakeDatasetSync`` drives
tests; ``PlatformDatasetSync`` publishes to the live backend endpoints
(``POST /api/v1/public/datasets`` + ``.../{id}/versions``), verified against the
implemented contract (see ``offline-eval/BACKEND-REQUIREMENTS.md`` A2/A4).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable
from urllib.parse import quote

from traceroot.eval.canonical import normalize
from traceroot.eval.types import DatasetSnapshot


class DatasetConflictError(Exception):
    """Raised when a push's ``base_version_id`` is behind the remote's current version."""

    def __init__(self, base_version_id: str | None, current_version_id: str | None) -> None:
        super().__init__(
            f"dataset changed remotely: base={base_version_id!r} current={current_version_id!r}. "
            "Pull the latest version, review the diff, and retry intentionally."
        )
        self.base_version_id = base_version_id
        self.current_version_id = current_version_id


class DatasetPublishAborted(RuntimeError):
    """Raised when publishing a new version to an ALREADY-EXISTING dataset is declined at the
    confirmation step (the push is a no-op: no version was created)."""


def _confirm_new_version(info: dict[str, Any]) -> bool:
    """Default double-check before adding a version to an EXISTING dataset.

    A dataset's identity is its name, so re-pushing the same name updates the SAME dataset with a
    NEW version rather than forking. That's usually intended, but it can surprise someone who reused
    a name by accident -- so on an interactive terminal we ask first (default ``no``: neither an
    accidental Enter nor an EOF ever publishes). Non-interactive contexts (CI, pipes) proceed
    silently so automation is never blocked, and ``TRACEROOT_ASSUME_YES=1`` skips the prompt
    everywhere. Pass an explicit ``on_existing`` to ``push_dataset`` to fully customize."""
    import os
    import sys

    if os.environ.get("TRACEROOT_ASSUME_YES", "").strip().lower() in ("1", "true", "yes"):
        return True
    if not (getattr(sys, "stdin", None) and sys.stdin.isatty()):
        return True
    name = info.get("name") or info.get("dataset_id") or "?"
    ver = info.get("current_dataset_version_id")
    try:
        answer = input(
            f"Dataset {name!r} already exists (current version {ver}). Publish a NEW version? [y/N] "
        )
    except EOFError:
        return False  # no answer at all (^D / stream closed) -> the same default no
    return answer.strip().lower() in ("y", "yes")


@dataclasses.dataclass
class PushResult:
    """Outcome of ``Dataset.push`` - explicit about local vs uploaded state."""

    status: str  # "local_only" | "uploaded"
    dataset_id: str
    dataset_version_id: str | None = None
    version_number: int | None = None


@runtime_checkable
class DatasetSyncTransport(Protocol):
    def push_dataset(
        self,
        snapshot: DatasetSnapshot,
        base_version_id: str | None,
        *,
        on_existing: Callable[[dict[str, Any]], bool] | None = None,
    ) -> PushResult: ...


class LocalDatasetSync:
    """Default no-op: the dataset stays local; nothing is published."""

    def push_dataset(
        self,
        snapshot: DatasetSnapshot,
        base_version_id: str | None,
        *,
        on_existing: Callable[[dict[str, Any]], bool] | None = None,
    ) -> PushResult:
        return PushResult(status="local_only", dataset_id=snapshot.dataset_id)


class FakeDatasetSync:
    """Deterministic in-memory sync for tests: versions, idempotency, conflicts."""

    def __init__(self) -> None:
        # Per-dataset version/idempotency/conflict state, so ONE instance can model several
        # independent server datasets. A single global cursor previously cross-contaminated them
        # (a second dataset inherited the first's version -> false conflicts or skipped updates).
        self._by_dataset: dict[str, dict[str, Any]] = {}
        self._last_dataset_id: str | None = None
        self.pushes: list[tuple[str, str, str]] = []  # (dataset_id, revision, version_id)

    def _state_for(self, dataset_id: str) -> dict[str, Any]:
        return self._by_dataset.setdefault(
            dataset_id, {"version_id": None, "counter": 0, "revision": None}
        )

    @property
    def current_version_id(self) -> str | None:
        """Current server version for the most recently pushed dataset."""
        if self._last_dataset_id is None:
            return None
        return self._by_dataset.get(self._last_dataset_id, {}).get("version_id")

    def force_current_version(self, version_id: str, dataset_id: str | None = None) -> None:
        """Seed a server version to simulate the server moving ahead (test helper). Applies to the
        given dataset, or the most recently pushed one."""
        did = dataset_id or self._last_dataset_id
        if did is None:
            raise ValueError("force_current_version: push a dataset first, or pass a dataset_id")
        self._state_for(did)["version_id"] = version_id

    def push_dataset(
        self,
        snapshot: DatasetSnapshot,
        base_version_id: str | None,
        *,
        on_existing: Callable[[dict[str, Any]], bool] | None = None,
    ) -> PushResult:
        # In-memory sync has no remote existence to double-check; on_existing is accepted for
        # interface parity and only consulted when a prior version already exists here.
        s = self._state_for(snapshot.dataset_id)
        if s["version_id"] is not None and on_existing is not None:
            if not on_existing({"name": snapshot.name, "current_dataset_version_id": s["version_id"]}):
                raise DatasetPublishAborted(
                    f"publish to existing dataset {snapshot.name!r} declined; no new version created."
                )
        self._last_dataset_id = snapshot.dataset_id
        # Optimistic concurrency: base must match this dataset's current version.
        if s["version_id"] is not None and base_version_id != s["version_id"]:
            raise DatasetConflictError(base_version_id, s["version_id"])
        # Idempotency: unchanged content re-push returns the same version.
        if snapshot.revision == s["revision"]:
            return PushResult("uploaded", snapshot.dataset_id, s["version_id"], s["counter"])
        s["counter"] += 1
        s["version_id"] = f"dsv_{s['counter']}"
        s["revision"] = snapshot.revision
        self.pushes.append((snapshot.dataset_id, snapshot.revision, s["version_id"]))
        return PushResult("uploaded", snapshot.dataset_id, s["version_id"], s["counter"])


class PlatformDatasetSync:
    """Real dataset publish against the live backend (A2/A4).

    Upserts the dataset (``POST /api/v1/public/datasets``), then publishes the active
    cases as one immutable version (``POST .../{id}/versions``) with a
    ``base_version_id`` for optimistic concurrency. Raises ``DatasetConflictError`` on
    a 409 stale base and a clear ``ValueError`` on the 413 batch cap. Note: only
    active cases are sent as upserts; archive/delete of cases removed since the base
    version is not yet emitted (documented limitation).
    """

    def __init__(self, *, api_key: str | None = None, host_url: str | None = None) -> None:
        from traceroot.eval.platform import _resolve_credentials

        self.api_key, self.host_url = _resolve_credentials(api_key, host_url)
        if not self.api_key:
            raise ValueError("PlatformDatasetSync needs an API key.")
        self.host_url = self.host_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        from traceroot.eval.platform import _http_json

        return _http_json(method, f"{self.host_url}{path}", self.api_key, body)

    def _existing_dataset(self, dataset_id: str) -> dict[str, Any] | None:
        """The remote dataset's metadata if it ALREADY exists with a published version, else None.

        ONLY a 404 means "no such dataset". Any other failure (401/403/5xx/timeout) says nothing
        about whether the dataset exists, and reading it as "new" would skip BOTH the idempotent
        no-op and the confirmation -- publishing an unprompted version off a transient error. So
        everything but a 404 propagates.
        """
        try:
            meta = self._request(
                "GET", f"/api/v1/public/datasets/{quote(dataset_id, safe='')}"
            )
        except RuntimeError as exc:
            if " HTTP 404:" in str(exc):
                return None
            raise
        return meta if meta.get("current_dataset_version_id") else None

    def _published_revision(self, dataset_id: str, version_id: str) -> str | None:
        """The content revision of the dataset's current published version, for change detection.
        None when it can't be fetched (then we fall through to confirm rather than assume unchanged)."""
        try:
            from traceroot.eval.platform import pull_dataset_version

            current = pull_dataset_version(
                version_id, dataset_id=dataset_id, api_key=self.api_key, host_url=self.host_url
            )
            return current.snapshot().revision
        except Exception:
            return None

    def _upsert_dataset(self, snapshot: DatasetSnapshot) -> None:
        """Upsert the dataset's un-versioned metadata (name/description). Not a version."""
        self._request(
            "POST",
            "/api/v1/public/datasets",
            {
                "dataset_id": snapshot.dataset_id,
                "name": snapshot.name,
                "description": snapshot.description,
            },
        )

    def push_dataset(
        self,
        snapshot: DatasetSnapshot,
        base_version_id: str | None,
        *,
        on_existing: Callable[[dict[str, Any]], bool] | None = None,
    ) -> PushResult:
        # A dataset's identity is its name: re-pushing the same name updates the SAME dataset with a
        # new version. If the content is UNCHANGED from the current version, this is a no-op (reuse
        # it, no prompt). If it CHANGED, double-check before creating a new version (default: prompt
        # on an interactive TTY, proceed otherwise) so a reused name never silently adds a version.
        existing = self._existing_dataset(snapshot.dataset_id)
        if existing is not None:
            current_version = existing["current_dataset_version_id"]
            if self._published_revision(snapshot.dataset_id, current_version) == snapshot.revision:
                # Identical content -> idempotent no-op; keep the current version, never prompt.
                # ``name``/``description`` are dataset metadata, NOT part of the content revision,
                # so an edit to either lands here with an unchanged revision. Returning without the
                # upsert would make ``ds.description = ...; ds.push()`` a silent no-op, so send the
                # metadata first and only skip the VERSION.
                self._upsert_dataset(snapshot)
                return PushResult("uploaded", snapshot.dataset_id, current_version)
            confirm = on_existing if on_existing is not None else _confirm_new_version
            if not confirm(existing):
                # A declined publish leaves the remote entirely untouched -- metadata included.
                raise DatasetPublishAborted(
                    f"publish to existing dataset {snapshot.name!r} declined; no new version created."
                )
        self._upsert_dataset(snapshot)
        changes: list[dict[str, Any]] = []
        for c in snapshot.cases:
            # Native JSON at the HTTP boundary: input/expected/metadata are sent as
            # their real values. The backend owns the single JSON-encode into its
            # storage column (input/expected schema is z.unknown()); the SDK does NOT
            # pre-encode -- doing so would double-encode.
            #
            # The value SENT is the SAME canonical form that was HASHED into the
            # revision, so a pulled version re-hashes to the revision that published
            # it. Sending a differently-shaped value (a datetime here, an ISO string
            # in storage) would make publishedRevision != snapshot.revision forever,
            # and every push would publish a no-change version.
            change: dict[str, Any] = {
                "op": "upsert",
                "test_case_id": c.id,
                "input": normalize(c.input),
            }
            if c.expected is not None:
                change["expected"] = normalize(c.expected)
            if c.metadata is not None:
                change["metadata"] = normalize(c.metadata)
            if c.source_trace_id is not None:
                change["source_trace_id"] = c.source_trace_id
            if c.source_span_id is not None:
                change["source_span_id"] = c.source_span_id
            changes.append(change)
        if not changes:
            # The backend requires >= 1 change; an all-archived/empty snapshot has none.
            raise ValueError("cannot publish a dataset version with no active cases")
        try:
            resp = self._request(
                "POST",
                f"/api/v1/public/datasets/{quote(snapshot.dataset_id, safe='')}/versions",
                {"base_version_id": base_version_id, "changes": changes},
            )
        except RuntimeError as exc:
            if " HTTP 413:" in str(exc):
                raise ValueError(
                    f"too many changes in one push ({len(changes)}); the backend caps a "
                    "version at ~1000 changes. Split the dataset or publish in stages."
                ) from exc
            if " HTTP 409:" in str(exc):
                current = None
                try:
                    import json

                    detail = str(exc).split(" HTTP 409:", 1)[1].strip()
                    current = json.loads(detail).get("current_version_id")
                except (ValueError, IndexError):
                    pass
                raise DatasetConflictError(base_version_id, current) from exc
            raise
        return PushResult(
            status="uploaded",
            dataset_id=snapshot.dataset_id,
            dataset_version_id=resp.get("dataset_version_id"),
            version_number=resp.get("version_number"),
        )
