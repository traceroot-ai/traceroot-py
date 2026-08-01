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
from urllib.parse import quote

from traceroot.eval.types import DatasetSnapshot
from traceroot.utils import serialize_value


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

    def push_dataset(self, snapshot: DatasetSnapshot, base_version_id: str | None) -> PushResult:
        s = self._state_for(snapshot.dataset_id)
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
        changes: list[dict[str, Any]] = []
        for c in snapshot.cases:
            # Native JSON at the HTTP boundary: input/expected/metadata are sent as
            # their real Python values. The backend owns the single JSON-encode into
            # its storage column (input/expected schema is z.unknown()); the SDK does
            # NOT pre-encode -- doing so would double-encode. serialize_value only
            # normalizes non-JSON-native objects (datetime, dataclasses, ...) to
            # JSON-safe values; a dict stays a dict, "42" stays the string "42".
            change: dict[str, Any] = {
                "op": "upsert",
                "test_case_id": c.id,
                "input": serialize_value(c.input),
            }
            if c.expected is not None:
                change["expected"] = serialize_value(c.expected)
            if c.metadata is not None:
                change["metadata"] = serialize_value(c.metadata)
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
