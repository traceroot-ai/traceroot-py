"""Core data types for offline evaluation.

Local-first: dataset construction, mutation, snapshotting, and serialization
perform NO network I/O. See ``offline-eval/architecture-v2-rebaseline.md``.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from traceroot.eval.canonical import (
    CanonicalizationError,
    canonical_hash,
    canonical_json,
    normalize,
)
from traceroot.eval.ids import stable_case_id, stable_dataset_id


@dataclasses.dataclass(frozen=True)
class EvalCase:
    """A runnable test case.

    Only ``input`` is required. ``expected`` is always optional and is never
    inferred from a source span. ``source_trace_id`` / ``source_span_id`` are
    provenance only. ``score_target_span_id`` is a reserved hook. ``archived``
    marks a case retained for lineage but excluded from evaluation/snapshots.
    """

    input: Any
    id: str | None = None
    expected: Any | None = None
    metadata: dict[str, Any] | None = None
    source_trace_id: str | None = None
    source_span_id: str | None = None
    score_target_span_id: str | None = None
    archived: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class Score:
    """A single scorer result. ``value`` may be numeric, boolean, or categorical.

    ``comment`` is the human-readable explanation; ``version`` is the scorer version.
    """

    name: str
    value: float | int | bool | str
    comment: str | None = None
    metadata: dict[str, Any] | None = None
    # The scorer version, when the scorer explicitly declares one. None means
    # unversioned -- V1 never invents a "1" for a scorer that did not declare a version.
    version: str | None = None


@dataclasses.dataclass(frozen=True)
class DeferredScore:
    """A scorer's signal that a score needs later (e.g. human) review.

    Recorded as a pending score - never coerced to a numeric zero. A case whose
    only score is deferred is ``not_scored``, distinct from a score of 0.
    """

    name: str
    reason: str | None = None


@dataclasses.dataclass(frozen=True)
class ScorerContext:
    """The single object argument passed to every scorer."""

    input: Any
    output: Any
    expected: Any | None
    metadata: dict[str, Any] | None


# Content fields that define a snapshot's identity (archived + volatile excluded).
# Every field here MUST also travel on the wire and come back on a pull -- a field that is hashed
# but not stored makes the published revision permanently unequal to the local one, so every push
# publishes a no-change version. ``score_target_span_id`` is a reserved hook that the platform does
# not persist yet, so it is deliberately NOT part of the identity.
_CONTENT_FIELDS = (
    "id",
    "input",
    "expected",
    "metadata",
    "source_trace_id",
    "source_span_id",
)


def _validate_payload(input: Any, expected: Any, metadata: Any) -> None:
    """Reject a case payload the canonicalizer cannot represent, AT AUTHORING TIME.

    Identity (case id + dataset revision) and the wire form are both the canonical form, so a
    value with no canonical form has no stable identity. Failing here names the offending field
    instead of surfacing as a mystery hash difference between the two SDKs later.
    """
    for field, value in (("input", input), ("expected", expected), ("metadata", metadata)):
        if value is None:
            continue
        try:
            normalize(value)
        except CanonicalizationError as e:
            raise CanonicalizationError(f"test case {field}: {e}") from None


def _content_revision(cases: tuple[EvalCase, ...]) -> str:
    # Content-addressed and ORDER-INDEPENDENT: sort by id so re-authoring the same set of cases in
    # a different order yields the SAME revision (with content-based ids, reordering is not a content
    # change). Only a real content change (add/remove/edit a case) advances the revision.
    ordered = sorted(cases, key=lambda c: c.id or "")
    content = [{k: getattr(c, k) for k in _CONTENT_FIELDS} for c in ordered]
    return "rev_" + canonical_hash(content, 16)


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    """An immutable, content-addressed snapshot of a dataset's active cases."""

    dataset_id: str
    name: str
    description: str | None
    revision: str
    cases: tuple[EvalCase, ...]
    base_version_id: str | None = None
    # The authoring key the dataset id was hashed from. Carried so a push can SEND it: the
    # platform cannot derive it (a renamed or explicitly keyed dataset hashes from something
    # the name no longer spells), and without it a later pull can only guess.
    key: str | None = None

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases)

    def __len__(self) -> int:
        return len(self.cases)

    def get(self, case_id: str) -> EvalCase | None:
        return next((c for c in self.cases if c.id == case_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "key": self.key,
            "description": self.description,
            "revision": self.revision,
            "base_version_id": self.base_version_id,
            "cases": [c.to_dict() for c in self.cases],
        }


class Dataset:
    """A local, mutable, ordered collection of :class:`EvalCase` keyed by stable id.

    Construction and mutation perform no network I/O. Identity is the dataset's
    ``key`` (defaults to ``name``), NOT which SDK or process created it: the
    ``dataset_id`` is a deterministic ``ds_``+sha256 of the key, so re-constructing
    the same dataset -- another process, Python or TypeScript -- yields the SAME id
    and the platform converges their runs instead of forking a new dataset each run.
    Pass an explicit ``key`` to keep identity stable across a display-name rename.
    ``dataset_version_id`` is set only when this Dataset mirrors a pushed/pulled
    remote version; changed content under the same key becomes a new VERSION.
    """

    def __init__(
        self, name: str, description: str | None = None, *, key: str | None = None
    ) -> None:
        self.name = name
        self.description = description
        self.key = key or name
        self.dataset_id = stable_dataset_id(self.key)
        self.dataset_version_id: str | None = None
        self.base_version_id: str | None = None
        self._cases: dict[str, EvalCase] = {}

    # --- authoring / mutation (network-free) ---
    def _content_id(self, input: Any) -> str:
        """The stable content id for ``input``: dataset key + canonical input + first free
        occurrence. Shared by ``add`` and ``upsert`` so both converge on the same id."""
        canonical = canonical_json(input)
        occurrence = 0
        cid = stable_case_id(self.key, canonical, occurrence)
        while cid in self._cases:
            occurrence += 1
            cid = stable_case_id(self.key, canonical, occurrence)
        return cid

    def add(
        self,
        input: Any,
        *,
        expected: Any | None = None,
        metadata: dict[str, Any] | None = None,
        source_trace_id: str | None = None,
        source_span_id: str | None = None,
        id: str | None = None,
    ) -> EvalCase:
        """Add a new case (strict: a duplicate id raises).

        Without an explicit id, the case gets a STABLE id from the dataset key + its INPUT
        content (not its position), so re-authoring the same dataset converges (the platform
        matches by id) and inserting/removing/reordering other cases does not shift this case's
        id. Duplicate inputs are disambiguated by occurrence (the first free slot), which also
        keeps ids collision-free across removes.
        """
        _validate_payload(input, expected, metadata)
        cid = id if id else self._content_id(input)
        if cid in self._cases:
            raise ValueError(f"test case id already exists: {cid!r}")
        case = EvalCase(
            input=input,
            id=cid,
            expected=expected,
            metadata=metadata,
            source_trace_id=source_trace_id,
            source_span_id=source_span_id,
        )
        self._cases[cid] = case
        return case

    def upsert(self, case: EvalCase) -> EvalCase:
        """Add or replace by id.

        An anonymous case gets the SAME content-derived id ``add()`` would give it, not a fresh
        random one: a random id would fork the case into a new server case on every process, which
        is exactly what content-addressed ids exist to prevent.
        """
        _validate_payload(case.input, case.expected, case.metadata)
        if case.id is None:
            case = dataclasses.replace(case, id=self._content_id(case.input))
        self._cases[case.id] = case
        return case

    def update(self, id: str, **changes: Any) -> EvalCase:
        """Replace fields of an existing case; raises KeyError if absent."""
        if id not in self._cases:
            raise KeyError(id)
        # Reject id changes: the map is keyed by id, so silently changing it would leave the case
        # under the old key (get(new_id) fails) and let snapshots hold ids that don't match the index.
        if changes.get("id", id) != id:
            raise ValueError("test case id cannot be changed via update()")
        updated = dataclasses.replace(self._cases[id], **changes)
        # Validate the MERGED case, exactly as add()/upsert() validate theirs: an edit writes the
        # same payload fields, so an unrepresentable value must fail here (naming the field) rather
        # than at snapshot()/push, where nothing recalls which edit introduced it.
        _validate_payload(updated.input, updated.expected, updated.metadata)
        self._cases[id] = updated
        return updated

    def archive(self, id: str) -> None:
        """Soft-archive a case: retained for lineage, excluded from active set."""
        if id not in self._cases:
            raise KeyError(id)
        self._cases[id] = dataclasses.replace(self._cases[id], archived=True)

    def remove(self, id: str) -> None:
        """Hard-delete a case; raises KeyError if absent."""
        del self._cases[id]

    # --- access ---
    def get(self, case_id: str) -> EvalCase | None:
        return self._cases.get(case_id)

    def cases(self, *, include_archived: bool = False) -> list[EvalCase]:
        return [c for c in self._cases.values() if include_archived or not c.archived]

    def __iter__(self) -> Iterator[EvalCase]:
        return iter(self.cases())

    def __len__(self) -> int:
        return len(self.cases())

    # --- snapshot ---
    def snapshot(self) -> DatasetSnapshot:
        """Immutable snapshot of the active cases with a content revision."""
        original = self.cases()
        # Compute the revision from the ORIGINAL cases, not the deepcopy: the serializer's
        # opaque-object fallback can include object identity, and a fresh deepcopy would change that
        # identity (and thus the revision) on every snapshot, breaking the stable content-addressed
        # contract. The originals keep a consistent identity within the process.
        revision = _content_revision(tuple(original))
        # Deep-copy the stored payloads so a later in-place mutation of the dataset can't change
        # what this immutable snapshot holds (EvalCase is frozen but its payloads are shared refs).
        active = tuple(copy.deepcopy(c) for c in original)
        return DatasetSnapshot(
            dataset_id=self.dataset_id,
            name=self.name,
            description=self.description,
            revision=revision,
            cases=active,
            base_version_id=self.base_version_id,
            key=self.key,
        )

    # --- serialization (network-free) ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            # The key is what every case id is derived from, so it must survive save/load:
            # without it a reloaded dataset falls back to `name` and every case added
            # afterwards gets an id that no longer converges with locally authored ones.
            "key": self.key,
            "description": self.description,
            "base_version_id": self.base_version_id,
            "dataset_version_id": self.dataset_version_id,  # keep the remote binding through save/load
            "cases": [c.to_dict() for c in self._cases.values()],  # incl. archived
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dataset:
        ds = cls(name=d["name"], description=d.get("description"), key=d.get("key"))
        ds.dataset_id = d.get("dataset_id", ds.dataset_id)
        ds.base_version_id = d.get("base_version_id")
        ds.dataset_version_id = d.get("dataset_version_id")
        for cd in d.get("cases", []):
            case = EvalCase(**cd)
            ds._cases[case.id] = case  # type: ignore[index]
        return ds

    def to_json(self) -> str:
        return json.dumps(normalize(self.to_dict()), ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> Dataset:
        return cls.from_dict(json.loads(text))

    def save(self, path: str) -> None:
        """Write to disk. ``.jsonl`` = header line + one line per case; else ``.json``."""
        if path.endswith(".jsonl"):
            header = {
                "type": "dataset",
                "dataset_id": self.dataset_id,
                "name": self.name,
                "key": self.key,
                "description": self.description,
                "base_version_id": self.base_version_id,
                "dataset_version_id": self.dataset_version_id,
                "schema": 1,
            }
            lines = [json.dumps(normalize(header), ensure_ascii=False)]
            for c in self._cases.values():
                lines.append(
                    json.dumps(normalize({"type": "case", **c.to_dict()}), ensure_ascii=False)
                )
            Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
        else:
            Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> Dataset:
        text = Path(path).read_text(encoding="utf-8")
        if path.endswith(".jsonl"):
            records = [json.loads(line) for line in text.strip().splitlines()]
            header = records[0]
            d = {
                k: header.get(k)
                for k in (
                    "dataset_id",
                    "name",
                    "key",
                    "description",
                    "base_version_id",
                    "dataset_version_id",
                )
            }
            d["cases"] = [{k: v for k, v in rec.items() if k != "type"} for rec in records[1:]]
            return cls.from_dict(d)
        return cls.from_json(text)

    def push(
        self,
        transport: Any = None,
        *,
        base_version_id: str | None = None,
        on_existing: Any = None,
    ) -> Any:
        """Explicitly publish this dataset as ONE immutable server version.

        Local mutations never create versions; this is the deliberate publish
        boundary. ``transport`` defaults to the platform transport
        (``PlatformDatasetSync``), so ``push()`` publishes to TraceRoot — the same
        cloud-by-default behaviour as ``evaluate()``. When no credentials are
        configured it raises a clear error rather than silently staying local; for a
        deliberate offline push pass ``transport=LocalDatasetSync()``. Provide the
        remote version this edit was based on via ``base_version_id`` (defaults to
        this dataset's pinned version) for optimistic concurrency; a stale base raises
        ``DatasetConflictError``. ``on_existing`` overrides the double-check before
        adding a version to an already-existing dataset (default: the transport's own,
        an interactive prompt). Imported lazily to avoid a types <-> dataset_sync cycle.
        """
        if transport is not None:
            sync = transport
        else:
            import os

            import traceroot
            from traceroot.eval.dataset_sync import PlatformDatasetSync

            # Check for credentials WITHOUT auto-creating the global client. Constructing
            # PlatformDatasetSync() resolves creds via get_client(), which auto-initializes a
            # keyless client from the environment when none exists -- after which
            # traceroot.initialize() becomes a no-op, so a credential-less push() could never be
            # recovered. Reading the existing client / env avoids that side effect, and turns only
            # the genuinely-missing-key case into the actionable error; any other init error (e.g.
            # a malformed env setting) propagates from PlatformDatasetSync() unchanged.
            client = traceroot._client
            api_key = (client.api_key if client is not None else "") or os.environ.get(
                "TRACEROOT_API_KEY", ""
            )
            if not api_key:
                raise ValueError(
                    "Dataset.push() publishes to the TraceRoot platform, but no credentials are "
                    "configured. Call traceroot.initialize(api_key=..., host_url=...) first (or set "
                    "TRACEROOT_API_KEY), or pass an explicit transport (transport=LocalDatasetSync() "
                    "keeps it local)."
                )
            sync = PlatformDatasetSync()
        snapshot = self.snapshot()
        base = base_version_id if base_version_id is not None else self.base_version_id
        # Only forwarded when set, so a duck-typed transport without the keyword still works.
        extra = {"on_existing": on_existing} if on_existing is not None else {}
        result = sync.push_dataset(snapshot, base, **extra)
        if result.status == "uploaded" and result.dataset_version_id is not None:
            self.dataset_version_id = result.dataset_version_id
            self.base_version_id = result.dataset_version_id
        return result
