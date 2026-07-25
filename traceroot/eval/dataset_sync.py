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
from typing import Any, Protocol, runtime_checkable

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
        self, snapshot: DatasetSnapshot, base_version_id: str | None
    ) -> PushResult: ...


class LocalDatasetSync:
    """Default no-op: the dataset stays local; nothing is published."""

    def push_dataset(self, snapshot: DatasetSnapshot, base_version_id: str | None) -> PushResult:
        return PushResult(status="local_only", dataset_id=snapshot.dataset_id)


class FakeDatasetSync:
    """Deterministic in-memory sync for tests: versions, idempotency, conflicts."""

    def __init__(self) -> None:
        self.current_version_id: str | None = None
        self._version_counter = 0
        self._last_revision: str | None = None
        self.pushes: list[tuple[str, str, str]] = []  # (dataset_id, revision, version_id)

    def force_current_version(self, version_id: str) -> None:
        self.current_version_id = version_id

    def push_dataset(self, snapshot: DatasetSnapshot, base_version_id: str | None) -> PushResult:
        # Optimistic concurrency: base must match the remote's current version.
        if self.current_version_id is not None and base_version_id != self.current_version_id:
            raise DatasetConflictError(base_version_id, self.current_version_id)
        # Idempotency: unchanged content re-push returns the same version.
        if snapshot.revision == self._last_revision:
            return PushResult(
                "uploaded", snapshot.dataset_id, self.current_version_id, self._version_counter
            )
        self._version_counter += 1
        self.current_version_id = f"dsv_{self._version_counter}"
        self._last_revision = snapshot.revision
        self.pushes.append((snapshot.dataset_id, snapshot.revision, self.current_version_id))
        return PushResult(
            "uploaded", snapshot.dataset_id, self.current_version_id, self._version_counter
        )


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

    def push_dataset(self, snapshot: DatasetSnapshot, base_version_id: str | None) -> PushResult:
        self._request(
            "POST",
            "/api/v1/public/datasets",
            {
                "dataset_id": snapshot.dataset_id,
                "name": snapshot.name,
                "description": snapshot.description,
            },
        )
        from traceroot.eval.platform import _encode_field

        changes: list[dict[str, Any]] = []
        for c in snapshot.cases:
            # input/expected are sent as canonical JSON text so their type survives
            # the backend's TEXT column; metadata is a native JSONB column (sent raw).
            change: dict[str, Any] = {
                "op": "upsert",
                "test_case_id": c.id,
                "input": _encode_field(c.input),
            }
            if c.expected is not None:
                change["expected"] = _encode_field(c.expected)
            if c.metadata is not None:
                change["metadata"] = c.metadata
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
                f"/api/v1/public/datasets/{snapshot.dataset_id}/versions",
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
