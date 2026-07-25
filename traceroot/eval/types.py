"""Core data types for offline evaluation.

Local-first: dataset construction, mutation, snapshotting, and serialization
perform NO network I/O. See ``offline-eval/architecture-v2-rebaseline.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from traceroot.eval.ids import new_dataset_id, new_test_case_id
from traceroot.utils import serialize_value


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
_CONTENT_FIELDS = (
    "id",
    "input",
    "expected",
    "metadata",
    "source_trace_id",
    "source_span_id",
    "score_target_span_id",
)


def _content_revision(cases: tuple[EvalCase, ...]) -> str:
    content = [{k: getattr(c, k) for k in _CONTENT_FIELDS} for c in cases]
    canonical = json.dumps(
        serialize_value(content), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return "rev_" + hashlib.sha256(canonical.encode()).hexdigest()[:16]


@dataclasses.dataclass(frozen=True)
class DatasetSnapshot:
    """An immutable, content-addressed snapshot of a dataset's active cases."""

    dataset_id: str
    name: str
    description: str | None
    revision: str
    cases: tuple[EvalCase, ...]
    base_version_id: str | None = None

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
            "description": self.description,
            "revision": self.revision,
            "base_version_id": self.base_version_id,
            "cases": [c.to_dict() for c in self.cases],
        }


class Dataset:
    """A local, mutable, ordered collection of :class:`EvalCase` keyed by stable id.

    Construction and mutation perform no network I/O. A client-generated
    ``dataset_id`` (``ds_``+ULID) is assigned at creation so local work can begin
    without a server round-trip; ``dataset_version_id`` is set only when this
    Dataset mirrors a pushed/pulled remote version.
    """

    def __init__(self, name: str, description: str | None = None) -> None:
        self.name = name
        self.description = description
        self.dataset_id = new_dataset_id()
        self.dataset_version_id: str | None = None
        self.base_version_id: str | None = None
        self._cases: dict[str, EvalCase] = {}

    # --- authoring / mutation (network-free) ---
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
        """Add a new case (strict: a duplicate id raises)."""
        cid = id or new_test_case_id()
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
        """Add or replace by id; anonymous cases get a stable ULID id."""
        if case.id is None:
            case = dataclasses.replace(case, id=new_test_case_id())
        self._cases[case.id] = case
        return case

    def update(self, id: str, **changes: Any) -> EvalCase:
        """Replace fields of an existing case; raises KeyError if absent."""
        if id not in self._cases:
            raise KeyError(id)
        updated = dataclasses.replace(self._cases[id], **changes)
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
        active = tuple(self.cases())
        return DatasetSnapshot(
            dataset_id=self.dataset_id,
            name=self.name,
            description=self.description,
            revision=_content_revision(active),
            cases=active,
            base_version_id=self.base_version_id,
        )

    # --- serialization (network-free) ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "base_version_id": self.base_version_id,
            "cases": [c.to_dict() for c in self._cases.values()],  # incl. archived
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Dataset:
        ds = cls(name=d["name"], description=d.get("description"))
        ds.dataset_id = d.get("dataset_id", ds.dataset_id)
        ds.base_version_id = d.get("base_version_id")
        ds.dataset_version_id = d.get("dataset_version_id")
        for cd in d.get("cases", []):
            case = EvalCase(**cd)
            ds._cases[case.id] = case  # type: ignore[index]
        return ds

    def to_json(self) -> str:
        return json.dumps(serialize_value(self.to_dict()), ensure_ascii=False)

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
                "description": self.description,
                "base_version_id": self.base_version_id,
                "schema": 1,
            }
            lines = [json.dumps(serialize_value(header), ensure_ascii=False)]
            for c in self._cases.values():
                lines.append(
                    json.dumps(serialize_value({"type": "case", **c.to_dict()}), ensure_ascii=False)
                )
            Path(path).write_text("\n".join(lines) + "\n")
        else:
            Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> Dataset:
        text = Path(path).read_text()
        if path.endswith(".jsonl"):
            records = [json.loads(line) for line in text.strip().splitlines()]
            header = records[0]
            d = {k: header.get(k) for k in ("dataset_id", "name", "description", "base_version_id")}
            d["cases"] = [{k: v for k, v in rec.items() if k != "type"} for rec in records[1:]]
            return cls.from_dict(d)
        return cls.from_json(text)

    def push(self, transport: Any = None, *, base_version_id: str | None = None) -> Any:
        """Explicitly publish this dataset as ONE immutable server version.

        Local mutations never create versions; this is the deliberate publish
        boundary. ``transport`` defaults to a no-op ``LocalDatasetSync`` (local-only,
        no network). Provide the remote version this edit was based on via
        ``base_version_id`` (defaults to this dataset's pinned version) for
        optimistic concurrency; a stale base raises ``DatasetConflictError``.
        Imported lazily to avoid a types <-> dataset_sync cycle.
        """
        from traceroot.eval.dataset_sync import LocalDatasetSync

        sync = transport if transport is not None else LocalDatasetSync()
        snapshot = self.snapshot()
        base = base_version_id if base_version_id is not None else self.base_version_id
        result = sync.push_dataset(snapshot, base)
        if result.status == "uploaded" and result.dataset_version_id is not None:
            self.dataset_version_id = result.dataset_version_id
            self.base_version_id = result.dataset_version_id
        return result
