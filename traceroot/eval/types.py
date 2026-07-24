"""Core data types for offline evaluation.

Local-first: dataset construction, mutation, snapshotting, and serialization
perform NO network I/O. See ``offline-eval/architecture-v2-rebaseline.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any

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
